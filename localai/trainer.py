import random
import json
import platform
import warnings
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from localai import __version__

from localai.config import (
    ModelConfig, TrainConfig, DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH,
    DEFAULT_VOCAB_PATH, ADDITIONAL_DATA_PATH, V02_MODEL_PATH, V02_VOCAB_PATH,
    DEFAULT_DIALOGUE_PATH, DIALOGUE_BLOCK_SIZE,
)
from localai.tokenizer import CharTokenizer
from localai.dataset import load_text, get_batch, split_text, encode_splits, dataset_signature
from localai.model import LocalTransformer, dialogue_training_loss
from localai.checkpoint import save_checkpoint, load_checkpoint, load_optimizer_state, checkpoint_tokenizer
from localai.metrics import evaluate, perplexity, append_log, EarlyStopping
from localai.experiment import create_experiment, write_json
from localai.dialogue import load_dialogues, split_dialogues, dialogue_vocab_texts


def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_training_text():
    text = load_text(DEFAULT_DATA_PATH)
    if ADDITIONAL_DATA_PATH.exists():
        text += "\n" + load_text(ADDITIONAL_DATA_PATH)
        print("[DATA] 추가 학습 데이터를 불러왔습니다.")
    else:
        print("[DATA] 기본 학습 데이터만 사용합니다.")
    return text


def train(steps=None, resume=None, resume_vocab=None, *, train_config=None,
          model_config=None, data_path=None, output_dir=None,
          allow_legacy_resume=False, data_format="text"):
    train_config = replace(train_config) if train_config else TrainConfig()
    if steps is not None:
        train_config.max_steps = steps
    train_config.validate()
    if data_format not in ("text", "dialogue"):
        raise ValueError("데이터 형식은 text 또는 dialogue여야 합니다.")
    if data_format != "dialogue" and (train_config.prompt_loss_weight or train_config.session_augmentation):
        raise ValueError("문맥 손실과 세션 증강은 대화 학습에만 적용합니다.")
    if resume_vocab is not None and resume is None:
        raise ValueError("resume_vocab은 resume과 함께 사용해야 합니다.")
    if resume is not None and model_config is not None:
        raise ValueError("이어 학습에서는 체크포인트의 모델 설정을 사용합니다.")

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    device = select_device()
    if data_format == "dialogue":
        records = load_dialogues(data_path or DEFAULT_DIALOGUE_PATH)
        manifest = split_dialogues(records, train_config.validation_ratio, train_config.test_ratio,
                                   train_config.seed, train_config.near_duplicate_threshold)
    else:
        text = load_text(data_path) if data_path else load_training_text()
        manifest = split_text(text, train_config.validation_ratio, train_config.test_ratio,
                              train_config.seed, train_config.near_duplicate_threshold)
    checkpoint = None
    start_step = 0
    optimizer_restored = False
    contaminated = False
    if resume is not None:
        model, checkpoint = load_checkpoint(resume, device=device)
        previous = checkpoint.get("metadata", {})
        previous_format = previous.get("input_format", "text")
        if previous_format != data_format:
            raise ValueError("체크포인트와 학습 데이터 형식이 다릅니다. 대화 모델은 새 학습으로 시작하세요.")
        if "dataset" not in previous:
            if not allow_legacy_resume:
                raise ValueError("기존 모델은 새 검증/테스트 문장을 이미 학습했을 수 있습니다. "
                                 "새 학습을 권장합니다. 진단용 이어 학습은 --allow-legacy-resume을 사용하세요.")
            contaminated = True
            warnings.warn("기존 모델 이어 학습: 검증/테스트 오염 가능성. 일반화 성능으로 해석할 수 없습니다.", UserWarning)
        else:
            if previous["dataset"] != manifest:
                raise ValueError("이어 학습의 원문/분할 설정이 이전 실험과 다릅니다. 동일한 데이터와 분할 설정을 사용하세요.")
            contaminated = previous.get("heldout_contaminated", False)
        if resume_vocab is None and "tokenizer_vocab" not in checkpoint:
            known = {DEFAULT_MODEL_PATH.resolve(): DEFAULT_VOCAB_PATH,
                     V02_MODEL_PATH.resolve(): V02_VOCAB_PATH}
            resume_vocab = known.get(Path(resume).resolve())
        tokenizer = checkpoint_tokenizer(model, checkpoint, resume_vocab)
        if "tokenizer_vocab" not in checkpoint:
            warnings.warn("기존 형식은 문자-ID 매핑을 내장하지 않아 크기만 검사할 수 있습니다.", UserWarning)
        start_step = checkpoint.get("step", 0)
    else:
        tokenizer = CharTokenizer()
        vocab_texts = dialogue_vocab_texts(manifest["splits"]["train"]) if data_format == "dialogue" else manifest["splits"]["train"]
        tokenizer.build_vocab(vocab_texts)
        config = replace(model_config, vocab_size=tokenizer.vocab_size) if model_config else ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=DIALOGUE_BLOCK_SIZE if data_format == "dialogue" else ModelConfig().block_size)
        model = LocalTransformer(config).to(device)

    data, stats = encode_splits(manifest, tokenizer, model.config.block_size,
                                session_augmentation=train_config.session_augmentation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    if checkpoint is not None:
        optimizer_restored = load_optimizer_state(optimizer, checkpoint)
        # 모멘텀은 복원하되 이번 실행에서 요청한 학습률을 명시적으로 적용합니다.
        for group in optimizer.param_groups:
            group["lr"] = train_config.learning_rate
    experiment_id, directory = create_experiment(output_dir)
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    metadata = {
        "experiment_id": experiment_id, "status": "running",
        "origin": "resumed" if resume else "fresh",
        "resume_from": str(Path(resume).resolve()) if resume else None,
        "optimizer_restored": optimizer_restored,
        "heldout_contaminated": contaminated,
        "model_config": model.config.to_dict(), "train_config": asdict(train_config),
        "dataset": manifest, "dataset_signature": dataset_signature(manifest),
        "split_stats": stats, "parameter_count": model.count_parameters(),
        "torch_version": str(torch.__version__), "device": str(device),
        "python_version": platform.python_version(),
        "software_version": __version__,
        "model_version": "v0.3" if data_format == "dialogue" else "v0.2",
        "input_format": data_format,
        "loss_scope": "assistant_content_and_eos" if data_format == "dialogue" else "all_next_tokens",
        "training_objective": "response_with_context_auxiliary" if train_config.prompt_loss_weight else "response_only" if data_format == "dialogue" else "all_next_tokens",
        "resume_policy": "weights_and_optimizer; rng_and_early_stopping_restart_per_experiment",
        "evaluation": {"method": "full_response_windows_v1" if data_format == "dialogue" else "fixed_random_windows_v1", "seed": train_config.seed,
                       "batches": train_config.eval_batches, "batch_size": train_config.batch_size,
                       "block_size": model.config.block_size},
        "start_step": start_step, "step": start_step, "best_step": None,
        "best_val_loss": None, "stop_reason": None,
        "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
    }
    tokenizer.save(directory / "vocab.json")
    write_json(directory / "dataset.json", manifest)
    write_json(directory / "experiment.json", metadata)
    stopper = EarlyStopping(train_config.patience, train_config.min_delta)
    training_generator = torch.Generator().manual_seed(train_config.seed)
    print(f"[LocalAI] 실험: {experiment_id} / {device}")
    print(f"모델 파라미터: {model.count_parameters():,}; 분할 통계: {stats}")
    print(f"[기록] {directory}")

    def record(step):
        metrics = evaluate(model, data["train"], data["val"],
                           batch_size=train_config.batch_size,
                           eval_batches=train_config.eval_batches, seed=train_config.seed)
        improved, should_stop = stopper.update(metrics["val"])
        metadata.update(step=step, metrics=metrics, best_val_loss=stopper.best_loss,
                        bad_evaluations=stopper.bad_evaluations)
        if improved:
            metadata["best_step"] = step
            save_checkpoint(model, best_path, step, optimizer, metrics,
                            metadata=metadata, tokenizer=tokenizer)
        append_log(directory / "metrics.csv", step, metrics, best_val_loss=stopper.best_loss)
        # CSV의 기존 인터페이스를 유지하고 상세 상태는 JSONL에 기록합니다.
        with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"experiment_id": experiment_id, "step": step,
                                    "train_loss": metrics["train"], "val_loss": metrics["val"],
                                    "best_val_loss": stopper.best_loss, "best_step": metadata["best_step"],
                                    "best_checkpoint": str(best_path)}, allow_nan=False) + "\n")
        write_json(directory / "experiment.json", metadata)
        print(f"Step {step:5d} | Train Loss: {metrics['train']:.4f} "
              f"| Val Loss: {metrics['val']:.4f} | PPL: {perplexity(metrics['val']):.2f}")
        return metrics, should_stop

    try:
        training_started = time.monotonic()
        # 이어 학습 전 상태도 후보로 남겨 추가 학습이 성능을 악화해도 보존합니다.
        last_metrics, _ = record(start_step)
        last_evaluated_step = start_step
        model.train()
        stop_reason = "max_steps"
        for local_step in range(1, train_config.max_steps + 1):
            if (train_config.max_seconds is not None
                    and time.monotonic() - training_started >= train_config.max_seconds):
                stop_reason = "time_budget"
                if metadata["step"] != last_evaluated_step:
                    last_metrics, _ = record(metadata["step"])
                break
            global_step = start_step + local_step
            x, y = get_batch(data["train"], model.config.block_size,
                             train_config.batch_size, device, generator=training_generator)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            if data_format == "dialogue" and (train_config.prompt_loss_weight or train_config.label_smoothing):
                loss = dialogue_training_loss(logits, x, y, tokenizer.stoi[tokenizer.PAD],
                                               train_config.prompt_loss_weight, train_config.label_smoothing)
            elif train_config.label_smoothing:
                import torch.nn.functional as F
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                       label_smoothing=train_config.label_smoothing)
            if not torch.isfinite(loss):
                raise ValueError("학습 손실이 유한하지 않습니다.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            metadata["step"] = global_step
            if (local_step % train_config.log_interval == 0
                    or local_step == train_config.max_steps):
                last_metrics, should_stop = record(global_step)
                last_evaluated_step = global_step
                if should_stop:
                    stop_reason = "early_stopping"
                    break
        metadata.update(status="completed", stop_reason=stop_reason,
                        elapsed_seconds=time.monotonic() - training_started)
        save_checkpoint(model, last_path, metadata["step"], optimizer, last_metrics,
                        metadata=metadata, tokenizer=tokenizer)
        write_json(directory / "experiment.json", metadata)
    except (Exception, KeyboardInterrupt) as exc:
        metadata.update(status="failed", stop_reason=type(exc).__name__)
        write_json(directory / "experiment.json", metadata)
        raise
    print(f"[완료] {stop_reason}; best: {best_path}; last: {last_path}")
    return model, tokenizer
