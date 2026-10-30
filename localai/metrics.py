
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from localai.dataset import get_batch


def split_data(data, ratio=0.1, block_size=64):
    """
    연속된 토큰 데이터를 학습/검증 구간으로 분리합니다.

    데이터 누출을 줄이기 위해 순서를 섞지 않고
    서로 겹치지 않는 구간으로 나눕니다.
    """

    if not 0 < ratio < 1:
        raise ValueError(
            "검증 데이터 비율은 0과 1 사이여야 합니다."
        )

    total = len(data)
    split_index = int(total * (1 - ratio))

    train_data = data[:split_index]
    val_data = data[split_index:]

    if len(train_data) <= block_size:
        raise ValueError(
            "학습 데이터가 block_size보다 짧습니다."
        )

    if len(val_data) <= block_size:
        raise ValueError(
            "검증 데이터가 block_size보다 짧습니다. "
            "데이터를 늘리거나 block_size를 줄이세요."
        )

    return train_data, val_data


@torch.no_grad()
def evaluate(
    model,
    train_data,
    val_data,
    batch_size=16,
    eval_batches=10,
    seed=42,
):
    """
    학습/검증 데이터에 대한 평균 Loss를 측정합니다.
    가중치는 변경하지 않습니다.
    """

    if eval_batches < 1:
        raise ValueError(
            "eval_batches는 1 이상이어야 합니다."
        )

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    block_size = model.config.block_size

    results = {}
    from localai.dialogue import DialogueDataset, IGNORE_INDEX

    try:
        for split_index, (name, data) in enumerate([
            ("train", train_data),
            ("val", val_data)
        ]):
            if isinstance(data, DialogueDataset):
                # 응답 길이가 달라도 모든 정답 토큰을 같은 비중으로 평가합니다.
                total_loss, total_targets = 0.0, 0
                for start in range(0, len(data), batch_size):
                    x, y = data.batch(range(start, min(len(data), start + batch_size)), device)
                    _, loss = model(x, y)
                    count = int(y.ne(IGNORE_INDEX).sum().item())
                    total_loss += loss.item() * count
                    total_targets += count
                results[name] = total_loss / total_targets
                continue
            generator = torch.Generator().manual_seed(seed + split_index)
            losses = []

            for _ in range(eval_batches):
                x, y = get_batch(
                    data,
                    block_size,
                    batch_size,
                    device,
                    generator=generator,
                )

                _, loss = model(x, y)
                losses.append(loss.item())

            results[name] = sum(losses) / len(losses)

    finally:
        model.train(was_training)

    return results


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.001
    best_loss: float = float("inf")
    reference_loss: float = float("inf")
    bad_evaluations: int = 0

    def update(self, loss):
        if not math.isfinite(loss):
            raise ValueError("검증 손실이 유한하지 않습니다.")
        # best 저장은 실제 최저 손실, patience는 min_delta 이상의 개선을 기준으로 합니다.
        improved = loss < self.best_loss
        self.best_loss = min(self.best_loss, loss)
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss = loss
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
        return improved, self.patience > 0 and self.bad_evaluations >= self.patience


def perplexity(loss):
    """
    Loss를 Perplexity로 변환합니다.
    숫자가 낮을수록 예측 성능이 좋은 편입니다.
    """

    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def append_log(path, step, metrics, best_val_loss=None):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = path.exists() and path.stat().st_size > 0

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        if not file_exists:
            columns = [
                "step",
                "train_loss",
                "val_loss",
                "val_perplexity"
            ]
            if best_val_loss is not None:
                columns.append("best_val_loss")
            writer.writerow(columns)

        row = [
            step,
            round(metrics["train"], 6),
            round(metrics["val"], 6),
            round(perplexity(metrics["val"]), 6)
        ]
        if best_val_loss is not None:
            row.append(best_val_loss)
        writer.writerow(row)


def evaluate_checkpoint(path, split="val"):
    from localai.checkpoint import load_checkpoint, checkpoint_tokenizer
    from localai.dataset import encode_splits

    if split not in ("val", "test"):
        raise ValueError("val 또는 test 분할을 선택하세요.")
    model, checkpoint = load_checkpoint(path, device="cpu")
    metadata = checkpoint.get("metadata", {})
    if "dataset" not in metadata:
        raise ValueError("이 체크포인트에는 고정 분할 정보가 없습니다.")
    tokenizer = checkpoint_tokenizer(model, checkpoint)
    data, stats = encode_splits(metadata["dataset"], tokenizer, model.config.block_size)
    settings = metadata["evaluation"]
    metrics = evaluate(model, data["train"], data[split],
                       batch_size=settings["batch_size"], eval_batches=settings["batches"],
                       seed=settings["seed"])
    return {"checkpoint": str(Path(path).resolve()), "step": checkpoint["step"],
            "input_format": metadata.get("input_format", "text"),
            "loss_scope": metadata.get("loss_scope", "all_next_tokens"),
            "origin": metadata["origin"], "split": split, "loss": metrics["val"],
            "perplexity": perplexity(metrics["val"]), "stats": stats[split],
            "heldout_contaminated": metadata["heldout_contaminated"],
            "warning": "기존 모델의 사전 노출 가능성으로 독립된 평가가 아닙니다." if metadata["heldout_contaminated"] else None,
            "evaluation": settings,
            "dataset_signature": metadata["dataset_signature"]}
