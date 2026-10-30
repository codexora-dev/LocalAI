
import csv

import pytest
import torch

from localai.config import ModelConfig
from localai.model import LocalTransformer
from localai.metrics import (
    split_data,
    evaluate,
    append_log,
)
from localai.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_optimizer_state,
)


def make_small_model():
    config = ModelConfig(
        vocab_size=20,
        block_size=8,
        n_embd=16,
        n_head=4,
        n_layer=1,
        dropout=0.0
    )
    return LocalTransformer(config)


def test_split_data():
    data = torch.arange(200)

    train_data, val_data = split_data(
        data,
        ratio=0.2,
        block_size=8
    )

    assert len(train_data) == 160
    assert len(val_data) == 40
    assert train_data[-1].item() == 159
    assert val_data[0].item() == 160


def test_split_data_too_small():
    data = torch.arange(20)

    with pytest.raises(ValueError):
        split_data(
            data,
            ratio=0.1,
            block_size=8
        )


def test_evaluate():
    model = make_small_model()

    data = torch.randint(
        0, 20, (200,)
    )

    train_data, val_data = split_data(
        data,
        ratio=0.2,
        block_size=8
    )

    model.train()

    results = evaluate(
        model,
        train_data,
        val_data,
        batch_size=2,
        eval_batches=2
    )

    assert "train" in results
    assert "val" in results
    assert results["train"] > 0
    assert results["val"] > 0
    assert model.training


def test_optimizer_checkpoint(tmp_path):
    model = make_small_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))

    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

    path = tmp_path / "v02.pt"

    save_checkpoint(
        model,
        path,
        step=123,
        optimizer=optimizer,
        metrics={"val": 1.5}
    )

    loaded_model, checkpoint = load_checkpoint(path)

    new_optimizer = torch.optim.AdamW(
        loaded_model.parameters(),
        lr=0.001
    )

    restored = load_optimizer_state(
        new_optimizer,
        checkpoint
    )

    assert restored is True
    assert checkpoint["step"] == 123
    assert checkpoint["metrics"]["val"] == 1.5
    assert len(new_optimizer.state) > 0


def test_csv_log(tmp_path):
    path = tmp_path / "training.csv"

    append_log(
        path,
        step=10,
        metrics={
            "train": 1.2,
            "val": 1.5
        }
    )

    with open(
        path,
        encoding="utf-8",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["step"] == "10"
    assert float(rows[0]["val_loss"]) == 1.5
