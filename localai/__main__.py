
import argparse
from pathlib import Path

from localai.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VOCAB_PATH,
    V02_MODEL_PATH,
    V02_VOCAB_PATH,
    TrainConfig,
    DEFAULT_DIALOGUE_PATH,
    ModelConfig,
    DIALOGUE_BLOCK_SIZE,
)
from localai import __version__

from localai.trainer import train, select_device
from localai.checkpoint import load_checkpoint, checkpoint_tokenizer
from localai.generator import generate_text, generate_reply


def run_chat(checkpoint_path, prompt=None, length=100, temperature=0.6, history_turns=4,
             *, sample=False, top_p=0.9, repetition_penalty=1.1):
    if history_turns < 0:
        raise ValueError("history_turns는 0 이상이어야 합니다.")
    model, checkpoint = load_checkpoint(checkpoint_path, device=select_device())
    if checkpoint.get("metadata", {}).get("input_format") != "dialogue":
        raise ValueError("chat에는 --format dialogue로 학습한 체크포인트가 필요합니다.")
    tokenizer = checkpoint_tokenizer(model, checkpoint)
    print("[LocalAI v0.3] 실험용 대화 모델입니다. 대화 성능은 별도 검증이 필요합니다.")
    print(f"[모델] {checkpoint_path} | 학습 step {checkpoint.get('step', 0)}")
    if prompt is not None:
        result = generate_reply(model, tokenizer, [{"role": "user", "content": prompt}],
                                max_new_tokens=length, temperature=temperature,
                                greedy=not sample, top_p=top_p, repetition_penalty=repetition_penalty)
        print(result)
        return result
    print("/exit: 종료, /reset: 현재 대화 초기화. 대화는 파일에 저장하지 않습니다.")
    history = []
    while True:
        try:
            message = input("사용자> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화를 종료합니다.")
            break
        if message == "/exit":
            break
        if message == "/reset":
            history.clear()
            print("대화를 초기화했습니다.")
            continue
        if not message:
            continue
        previous = history[-2 * history_turns:] if history_turns else []
        pending = previous + [{"role": "user", "content": message}]
        try:
            reply = generate_reply(model, tokenizer, pending, max_new_tokens=length,
                                   temperature=temperature, greedy=not sample, top_p=top_p,
                                   repetition_penalty=repetition_penalty).strip()
        except ValueError as exc:
            print(f"[오류] {exc}")
            continue
        print(f"도우미> {reply}" if reply else "[생성된 응답이 없습니다.]")
        if reply:
            history = pending + [{"role": "assistant", "content": reply}]


def get_model_paths(version):
    if version == "v0.1":
        return DEFAULT_MODEL_PATH, DEFAULT_VOCAB_PATH

    if version == "v0.2":
        return V02_MODEL_PATH, V02_VOCAB_PATH

    raise ValueError(f"알 수 없는 버전: {version}")


def load_version(version):
    model_path, vocab_path = get_model_paths(version)

    model, checkpoint = load_checkpoint(
        model_path,
        device=select_device()
    )

    tokenizer = checkpoint_tokenizer(model, checkpoint, vocab_path)

    return model, tokenizer, checkpoint


def run_generate(version, prompt, length, temperature, checkpoint_path=None, vocab_path=None):
    if checkpoint_path:
        model, checkpoint = load_checkpoint(checkpoint_path, device=select_device())
        tokenizer = checkpoint_tokenizer(model, checkpoint, vocab_path)
    else:
        model, tokenizer, checkpoint = load_version(version)

    result = generate_text(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=length,
        temperature=temperature
    )

    print(f"\n[LocalAI {checkpoint_path or version}]")
    print(f"학습 Step: {checkpoint.get('step', 0)}")
    print(result)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="LocalAI v0.3 — 로컬 언어·대화 학습"
    )
    parser.add_argument("--version", action="version", version=f"LocalAI {__version__}")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # 학습 명령
    train_parser = subparsers.add_parser("train")

    train_parser.add_argument(
        "--steps",
        type=int,
        default=200
    )

    train_parser.add_argument(
        "--resume",
        type=str,
        default=None
    )

    train_parser.add_argument("--data", type=Path, help="지정한 로컬 텍스트만 사용")
    train_parser.add_argument("--format", choices=["text", "dialogue"], default="text")
    train_parser.add_argument("--output-dir", type=Path, help="존재하지 않는 새 실험 폴더")
    train_parser.add_argument("--patience", type=int, default=5, help="개선 없는 평가 횟수, 0: 비활성화")
    train_parser.add_argument("--min-delta", type=float, default=0.001)
    train_parser.add_argument("--eval-interval", type=int, default=20)
    train_parser.add_argument("--eval-batches", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--validation-ratio", type=float, default=0.1)
    train_parser.add_argument("--test-ratio", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    train_parser.add_argument("--prompt-loss-weight", type=float, default=0.0,
                              help="대화 문맥의 보조 학습 비중 (0: 기존 응답 전용)")
    train_parser.add_argument("--max-seconds", type=float,
                              help="학습 시간 한도, 스텝 경계에서 확인 후 최종 평가·저장")
    train_parser.add_argument("--label-smoothing", type=float, default=0.0)
    train_parser.add_argument("--dropout", type=float, help="새 모델에만 적용; 이어 학습은 기존 설정 유지")
    train_parser.add_argument("--session-augmentation", type=int, default=0,
                              help="학습 대화당 합성 세션 수 (0~4), 검증/테스트에는 미적용")
    train_parser.add_argument("--allow-legacy-resume", action="store_true",
                              help="검증/테스트 오염 가능성을 인정한 기존 모델 진단용 이어 학습")

    train_parser.add_argument(
        "--resume-vocab",
        type=str,
        default=None
    )

    # 생성 명령
    generate_parser = subparsers.add_parser("generate")

    generate_parser.add_argument(
        "--prompt",
        required=True
    )

    generate_parser.add_argument(
        "--length",
        type=int,
        default=100
    )

    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    generate_parser.add_argument(
        "--version",
        choices=["v0.1", "v0.2"],
        default="v0.2"
    )
    generate_parser.add_argument("--checkpoint", type=Path)
    generate_parser.add_argument("--vocab", type=Path)

    evaluate_parser = subparsers.add_parser("evaluate", help="명시적인 고정 분할 평가; 테스트는 모델 선택에 사용하지 마세요")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=["val", "test"], default="val")

    chat_parser = subparsers.add_parser("chat", help="대화 모델의 응답 생성과 현재 세션 문맥")
    chat_parser.add_argument("--checkpoint", type=Path, required=True)
    chat_parser.add_argument("--prompt", help="대화형 입력 대신 한 번만 질문")
    chat_parser.add_argument("--length", type=int, default=100)
    chat_parser.add_argument("--temperature", type=float, default=0.6)
    chat_parser.add_argument("--history-turns", type=int, default=4)
    chat_parser.add_argument("--sample", action="store_true", help="최대 확률 선택 대신 확률 샘플링 사용")
    chat_parser.add_argument("--top-p", type=float, default=0.9)
    chat_parser.add_argument("--repetition-penalty", type=float, default=1.1)

    inspect_parser = subparsers.add_parser("inspect-data", help="대화 JSONL 검사와 분할·어휘 통계")
    inspect_parser.add_argument("--data", type=Path, default=DEFAULT_DIALOGUE_PATH)
    inspect_parser.add_argument("--seed", type=int, default=42)
    inspect_parser.add_argument("--validation-ratio", type=float, default=0.1)
    inspect_parser.add_argument("--test-ratio", type=float, default=0.1)
    inspect_parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)

    assess_parser = subparsers.add_parser("assess-chat", help="실제 응답과 참조 문장 비교; 의미 정확도 검사는 아님")
    assess_parser.add_argument("--checkpoint", type=Path, required=True)
    assess_parser.add_argument("--data", type=Path, help="선택적인 개발용 질문·참조 응답 JSONL")
    assess_parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    assess_parser.add_argument("--max-cases", type=int, default=30)
    assess_parser.add_argument("--length", type=int, default=80)
    assess_parser.add_argument("--output", type=Path)

    # 모델 비교 명령
    compare_parser = subparsers.add_parser("compare")

    compare_parser.add_argument(
        "--prompt",
        required=True
    )

    compare_parser.add_argument(
        "--length",
        type=int,
        default=80
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            steps=args.steps,
            resume=args.resume,
            resume_vocab=args.resume_vocab,
            data_path=args.data,
            output_dir=args.output_dir,
            allow_legacy_resume=args.allow_legacy_resume,
            data_format=args.format,
            model_config=ModelConfig(
                block_size=DIALOGUE_BLOCK_SIZE if args.format == "dialogue" else ModelConfig().block_size,
                dropout=args.dropout) if args.dropout is not None else None,
            train_config=TrainConfig(
                patience=args.patience, min_delta=args.min_delta,
                log_interval=args.eval_interval, eval_batches=args.eval_batches,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                validation_ratio=args.validation_ratio, test_ratio=args.test_ratio,
                seed=args.seed, near_duplicate_threshold=args.near_duplicate_threshold,
                prompt_loss_weight=args.prompt_loss_weight, max_seconds=args.max_seconds,
                label_smoothing=args.label_smoothing, session_augmentation=args.session_augmentation),
        )

    elif args.command == "generate":
        if args.vocab is not None and args.checkpoint is None:
            parser.error("--vocab은 --checkpoint와 함께 사용해야 합니다.")
        run_generate(
            args.version,
            args.prompt,
            args.length,
            args.temperature,
            args.checkpoint,
            args.vocab,
        )

    elif args.command == "evaluate":
        import json
        from localai.metrics import evaluate_checkpoint
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.split), ensure_ascii=False, indent=2))

    elif args.command == "chat":
        run_chat(args.checkpoint, args.prompt, args.length, args.temperature, args.history_turns,
                 sample=args.sample, top_p=args.top_p, repetition_penalty=args.repetition_penalty)

    elif args.command == "inspect-data":
        import json
        from localai.dialogue import load_dialogues, split_dialogues, dialogue_vocab_texts
        from localai.tokenizer import CharTokenizer
        from localai.dataset import encode_splits, dataset_signature
        manifest = split_dialogues(load_dialogues(args.data), args.validation_ratio, args.test_ratio,
                                   args.seed, args.near_duplicate_threshold)
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(dialogue_vocab_texts(manifest["splits"]["train"]))
        _, stats = encode_splits(manifest, tokenizer, DIALOGUE_BLOCK_SIZE)
        report = {key: value for key, value in manifest.items() if key != "splits"}
        report.update(split_stats=stats, vocab_size=tokenizer.vocab_size,
                      dataset_signature=dataset_signature(manifest))
        print(json.dumps(report, ensure_ascii=False, indent=2))

    elif args.command == "assess-chat":
        import json
        from localai.assessment import assess_chat
        from localai.experiment import write_json
        if args.output is not None and args.output.exists():
            raise FileExistsError("평가 출력 경로가 이미 있습니다. 새 경로를 지정하세요.")
        report = assess_chat(args.checkpoint, args.data, args.split, args.max_cases, args.length)
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    elif args.command == "compare":
        for version in ["v0.1", "v0.2"]:
            try:
                run_generate(
                    version,
                    args.prompt,
                    args.length,
                    0.8
                )
            except FileNotFoundError as exc:
                print(f"\n[{version}] 파일 없음: {exc}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"[오류] {exc}") from None
