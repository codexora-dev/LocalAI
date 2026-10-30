
import torch

from localai.config import ModelConfig
from localai.model import LocalTransformer


def make_model():
    config = ModelConfig(
        vocab_size=20,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0
    )

    return LocalTransformer(config)


def test_forward_shape():
    model = make_model()

    x = torch.randint(0, 20, (2, 8))
    logits, loss = model(x)

    assert logits.shape == (2, 8, 20)
    assert loss is None


def test_training_backward():
    model = make_model()

    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))

    logits, loss = model(x, y)

    assert torch.isfinite(loss)

    loss.backward()

    assert model.token_embedding.weight.grad is not None


def test_causal_attention():
    model = make_model()
    model.eval()

    x1 = torch.tensor([[1, 2, 3, 4, 5]])
    x2 = torch.tensor([[1, 2, 3, 8, 9]])

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    # 미래 토큰을 바꿔도 앞쪽 예측은 같아야 합니다.
    assert torch.allclose(
        logits1[:, :3],
        logits2[:, :3],
        atol=1e-6
    )


def test_context_limit():
    model = make_model()

    x = torch.randint(0, 20, (1, 17))

    try:
        model(x)
        assert False, "길이 제한 검사가 작동하지 않았습니다."
    except ValueError:
        pass
