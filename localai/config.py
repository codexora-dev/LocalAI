
from dataclasses import dataclass, asdict
from pathlib import Path
import math

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

# 기존 v0.1 경로 유지
DEFAULT_DATA_PATH = DATA_DIR / "raw" / "starter.txt"
DEFAULT_MODEL_PATH = CHECKPOINT_DIR / "localai_v0.1.pt"
DEFAULT_VOCAB_PATH = MODEL_DIR / "vocab.json"

# v0.2 경로
ADDITIONAL_DATA_PATH = DATA_DIR / "raw" / "additional.txt"

V02_MODEL_PATH = CHECKPOINT_DIR / "localai_v0.2.pt"
V02_VOCAB_PATH = MODEL_DIR / "vocab_v0.2.json"
V02_LOG_PATH = LOG_DIR / "training_v0.2.csv"
DEFAULT_DIALOGUE_PATH = DATA_DIR / "raw" / "dialogue_seed_v03.jsonl"
DIALOGUE_BLOCK_SIZE = 128


@dataclass
class ModelConfig:
    vocab_size: int = 0
    block_size: int = 64
    n_embd: int = 96
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    def validate(self):
        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size는 1 이상이어야 합니다."
            )

        if self.n_head <= 0 or self.n_embd <= 0 or self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd는 n_head의 배수여야 합니다."
            )

        if self.block_size <= 0:
            raise ValueError(
                "block_size는 1 이상이어야 합니다."
            )

        if self.n_layer <= 0:
            raise ValueError(
                "n_layer는 1 이상이어야 합니다."
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout은 0 이상 1 미만이어야 합니다."
            )

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 200
    learning_rate: float = 0.001
    log_interval: int = 20
    seed: int = 42

    # 새 설정
    validation_ratio: float = 0.1
    eval_batches: int = 10
    test_ratio: float = 0.1
    patience: int = 5
    min_delta: float = 0.001
    near_duplicate_threshold: float = 0.85
    prompt_loss_weight: float = 0.0
    max_seconds: float | None = None
    label_smoothing: float = 0.0
    session_augmentation: int = 0

    def validate(self):
        for name in ("batch_size", "max_steps", "log_interval", "eval_batches"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name}는 1 이상이어야 합니다.")
        if self.patience < 0:
            raise ValueError("patience는 0 이상이어야 합니다. (0: 조기 종료 비활성화)")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta는 유한한 0 이상의 값이어야 합니다.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate는 유한한 양수여야 합니다.")
        if not (0 < self.validation_ratio < 1 and 0 < self.test_ratio < 1
                and self.validation_ratio + self.test_ratio < 1):
            raise ValueError("검증/테스트 비율은 양수이고 합이 1보다 작아야 합니다.")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("유사 문장 기준은 0 초과 1 이하여야 합니다.")
        if not math.isfinite(self.prompt_loss_weight) or not 0 <= self.prompt_loss_weight <= 1:
            raise ValueError("prompt_loss_weight는 0 이상 1 이하의 유한한 값이어야 합니다.")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds는 유한한 양수여야 합니다.")
        if not math.isfinite(self.label_smoothing) or not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing은 0 이상 1 미만의 유한한 값이어야 합니다.")
        if type(self.session_augmentation) is not int or not 0 <= self.session_augmentation <= 4:
            raise ValueError("session_augmentation은 0 이상 4 이하의 정수여야 합니다.")
