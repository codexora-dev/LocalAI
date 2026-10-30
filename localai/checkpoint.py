
from pathlib import Path

import torch

from localai.config import ModelConfig
from localai.model import LocalTransformer
from localai.experiment import atomic_path


def save_checkpoint(
    model,
    path,
    step=0,
    optimizer=None,
    metrics=None,
    metadata=None,
    tokenizer=None,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "format_version": 2,
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "step": step,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = (
            optimizer.state_dict()
        )

    if metadata is not None:
        checkpoint["metadata"] = metadata
    if tokenizer is not None:
        validate_tokenizer(model, tokenizer)
        checkpoint["tokenizer_vocab"] = dict(tokenizer.stoi)

    with atomic_path(path) as temporary:
        torch.save(checkpoint, temporary)


def load_checkpoint(path, device="cpu"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"저장된 모델이 없습니다: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    version = checkpoint.get(
        "format_version", 1
    )

    if version not in (1, 2):
        raise ValueError(
            f"지원하지 않는 체크포인트 버전: {version}"
        )

    config = ModelConfig(
        **checkpoint["model_config"]
    )

    model = LocalTransformer(config)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_optimizer_state(optimizer, checkpoint):
    state = checkpoint.get("optimizer_state")

    if state is None:
        return False

    optimizer.load_state_dict(state)
    return True


def validate_tokenizer(model, tokenizer, checkpoint=None):
    values = list(tokenizer.stoi.values())
    if (tokenizer.vocab_size != model.config.vocab_size
            or any(type(value) is not int for value in values)
            or set(values) != set(range(tokenizer.vocab_size))
            or any(token not in tokenizer.stoi for token in tokenizer.special_tokens)):
        raise ValueError("모델과 토크나이저의 어휘가 일치하지 않습니다.")
    expected = (checkpoint or {}).get("tokenizer_vocab")
    if expected is not None and expected != tokenizer.stoi:
        raise ValueError("체크포인트와 토크나이저의 문자-ID 매핑이 다릅니다.")


def checkpoint_tokenizer(model, checkpoint, vocab_path=None):
    from localai.tokenizer import CharTokenizer

    if vocab_path is not None:
        tokenizer = CharTokenizer.load(vocab_path)
    elif "tokenizer_vocab" in checkpoint:
        tokenizer = CharTokenizer()
        tokenizer.stoi = dict(checkpoint["tokenizer_vocab"])
        tokenizer.itos = {index: token for token, index in tokenizer.stoi.items()}
    else:
        raise ValueError("기존 체크포인트에는 별도의 어휘 파일 경로가 필요합니다.")
    validate_tokenizer(model, tokenizer, checkpoint)
    return tokenizer
