
import torch

from localai.config import ModelConfig
from localai.model import LocalTransformer
from localai.checkpoint import (
    save_checkpoint,
    load_checkpoint
)


def test_checkpoint_roundtrip(tmp_path):
    config = ModelConfig(
        vocab_size=15,
        block_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        dropout=0.0
    )

    model = LocalTransformer(config)
    model.eval()

    path = tmp_path / "test.pt"

    save_checkpoint(model, path, step=10)

    loaded, checkpoint = load_checkpoint(
        path,
        device="cpu"
    )

    assert checkpoint["step"] == 10
    assert loaded.config.vocab_size == 15

    x = torch.randint(0, 15, (1, 6))

    with torch.no_grad():
        out1, _ = model(x)
        out2, _ = loaded(x)

    assert torch.allclose(out1, out2)
