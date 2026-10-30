import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from localai import trainer
from localai.checkpoint import load_checkpoint, checkpoint_tokenizer, save_checkpoint, load_optimizer_state
from localai.config import ModelConfig, TrainConfig, ROOT_DIR
from localai.dataset import split_text, encode_splits
from localai.experiment import write_json
from localai.metrics import EarlyStopping, evaluate, evaluate_checkpoint
from localai.model import LocalTransformer
from localai.tokenizer import CharTokenizer


def corpus():
    rng = random.Random(7)
    return '\n'.join(''.join(rng.choices('abcdefghijklmnopqrstuvxyz0123456789 ', k=90)) for _ in range(30))


def tiny_config():
    return ModelConfig(block_size=8, n_embd=8, n_head=2, n_layer=1, dropout=0.0)


def tiny_train_config(**changes):
    return replace(TrainConfig(batch_size=1, max_steps=2, log_interval=1,
                               eval_batches=1, patience=0), **changes)


def test_sentence_groups_and_duplicates():
    text = 'The little cat is sleeping.\nThe little cat is sleeping.\nThe little cat is sleeping!\n' + corpus()
    manifest = split_text(text)
    assert manifest == split_text(text)
    assert manifest['exact_duplicates_removed'] == 1
    assert manifest['near_duplicate_pairs'] >= 1
    splits = manifest['splits']
    assert not set(splits['train']) & set(splits['val'])
    assert not set(splits['train']) & set(splits['test'])
    assert not set(splits['val']) & set(splits['test'])
    assert sum('The little cat is sleeping.' in units or 'The little cat is sleeping!' in units
               for units in splits.values()) == 1
    assert sum(len(units) for units in splits.values()) == manifest['unique_sentences']
    assert split_text(text, seed=1)['splits'] != splits


def test_small_dataset_rejected():
    with pytest.raises(ValueError):
        split_text('same.\nsame.\nsame.')
    manifest = split_text('a.\nb.\nc.', near_duplicate_threshold=1)
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(manifest['splits']['train'])
    with pytest.raises(ValueError):
        encode_splits(manifest, tokenizer, block_size=64)


def test_train_only_vocabulary():
    manifest = split_text(corpus())
    manifest['splits']['test'].append('한글전용문자')
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(manifest['splits']['train'])
    assert '한' not in tokenizer.stoi
    _, stats = encode_splits(manifest, tokenizer, 8)
    assert stats['test']['unknown_tokens'] > 0
    assert stats['train']['unknown_tokens'] == 0


def test_evaluation_reproducible_and_rng_unchanged():
    model = LocalTransformer(replace(tiny_config(), vocab_size=12))
    data = torch.arange(150) % 12
    state = torch.get_rng_state().clone()
    first = evaluate(model, data, data, batch_size=1, eval_batches=2)
    assert torch.equal(state, torch.get_rng_state())
    assert model.training
    assert first == evaluate(model, data, data, batch_size=1, eval_batches=2)


def test_early_stopping_minimum_and_patience():
    stop = EarlyStopping(patience=2, min_delta=0.1)
    assert stop.update(3.0) == (True, False)
    assert stop.update(2.95) == (True, False)
    assert stop.update(2.94) == (True, True)
    assert stop.best_loss == 2.94
    assert EarlyStopping(patience=0).update(3.0) == (True, False)
    with pytest.raises(ValueError):
        stop.update(float('nan'))


@pytest.mark.parametrize('changes', [dict(patience=-1), dict(min_delta=float('nan')),
    dict(log_interval=0), dict(eval_batches=0), dict(batch_size=0),
    dict(validation_ratio=0.9, test_ratio=0.2), dict(learning_rate=float('inf')),
    dict(near_duplicate_threshold=0)])
def test_config_validation(changes):
    with pytest.raises(ValueError):
        tiny_train_config(**changes).validate()


def test_training_best_last_resume_and_no_test_selection(tmp_path, monkeypatch):
    source = tmp_path / 'source.txt'
    source.write_text(corpus(), encoding='utf-8')
    output = tmp_path / 'run'
    losses = iter([3.0, 2.0, 2.1, 2.2])
    original_evaluate = trainer.evaluate
    seen = []
    manifest = split_text(corpus())
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(manifest['splits']['train'])
    encoded, _ = encode_splits(manifest, tokenizer, 8)

    def controlled_evaluate(model, train_data, val_data, **kwargs):
        assert torch.equal(train_data, encoded['train'])
        assert torch.equal(val_data, encoded['val'])
        seen.append(True)
        return {'train': 1.0, 'val': next(losses)}

    monkeypatch.setattr(trainer, 'evaluate', controlled_evaluate)
    trainer.train(data_path=source, output_dir=output, model_config=tiny_config(),
                  train_config=tiny_train_config(max_steps=20, patience=2))
    meta = json.loads((output / 'experiment.json').read_text(encoding='utf-8'))
    assert meta['stop_reason'] == 'early_stopping'
    assert meta['step'] == 3 and meta['best_step'] == 1
    assert len(seen) == 4
    best_model, best = load_checkpoint(output / 'best.pt')
    _, last = load_checkpoint(output / 'last.pt')
    assert best['step'] == 1 and last['step'] == 3
    assert best['metrics']['val'] == 2.0 and last['metrics']['val'] == 2.2
    assert any(not torch.equal(best['model_state'][key], last['model_state'][key])
               for key in best['model_state'])
    assert len(last['optimizer_state']['state']) > 0
    assert all(int(value['step']) == 3 for value in last['optimizer_state']['state'].values())
    old_hash = hashlib.sha256((output / 'last.pt').read_bytes()).hexdigest()
    monkeypatch.setattr(trainer, 'evaluate', original_evaluate)
    trainer.train(resume=output / 'last.pt', data_path=source, output_dir=tmp_path / 'resume',
                  train_config=tiny_train_config(max_steps=1))
    resumed = json.loads((tmp_path / 'resume' / 'experiment.json').read_text(encoding='utf-8'))
    assert resumed['start_step'] == 3 and resumed['step'] == 4
    assert resumed['optimizer_restored'] is True
    _, resumed_cp = load_checkpoint(tmp_path / 'resume' / 'last.pt')
    assert all(int(value['step']) == 4 for value in resumed_cp['optimizer_state']['state'].values())
    assert hashlib.sha256((output / 'last.pt').read_bytes()).hexdigest() == old_hash
    assert evaluate_checkpoint(output / 'best.pt', 'test')['split'] == 'test'
    with pytest.raises(ValueError, match='다릅니다'):
        trainer.train(resume=output / 'last.pt', data_path=source,
                      train_config=tiny_train_config(seed=99))
    with pytest.raises(FileExistsError):
        trainer.train(data_path=source, output_dir=output, model_config=tiny_config(),
                      train_config=tiny_train_config())
    # 동일한 어휘 크기라도 문자-ID 매핑이 바뀌면 거부합니다.
    wrong_vocab = dict(best['tokenizer_vocab'])
    a, b = list(wrong_vocab)[-2:]
    wrong_vocab[a], wrong_vocab[b] = wrong_vocab[b], wrong_vocab[a]
    write_json(tmp_path / 'wrong.json', wrong_vocab)
    with pytest.raises(ValueError, match='매핑'):
        checkpoint_tokenizer(best_model, best, tmp_path / 'wrong.json')


def test_legacy_resume_requires_explicit_acknowledgement(tmp_path):
    source = tmp_path / 'data.txt'
    source.write_text(corpus(), encoding='utf-8')
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([corpus()])
    model = LocalTransformer(replace(tiny_config(), vocab_size=tokenizer.vocab_size))
    save_checkpoint(model, tmp_path / 'old.pt', step=5)
    tokenizer.save(tmp_path / 'old.json')
    with pytest.raises(ValueError, match='allow-legacy-resume'):
        trainer.train(resume=tmp_path / 'old.pt', resume_vocab=tmp_path / 'old.json', data_path=source)
    with pytest.warns(UserWarning):
        trainer.train(resume=tmp_path / 'old.pt', resume_vocab=tmp_path / 'old.json', data_path=source,
                      output_dir=tmp_path / 'new', allow_legacy_resume=True,
                      train_config=tiny_train_config(max_steps=1))
    assert evaluate_checkpoint(tmp_path / 'new' / 'last.pt', 'test')['heldout_contaminated']


def test_v1_format_loading(tmp_path):
    model = LocalTransformer(replace(tiny_config(), vocab_size=10))
    path = tmp_path / 'v1.pt'
    torch.save({'model_config': model.config.to_dict(), 'model_state': model.state_dict(), 'step': 7}, path)
    loaded, checkpoint = load_checkpoint(path)
    assert loaded.config == model.config and checkpoint['step'] == 7


def test_atomic_checkpoint_preserves_existing_on_failure(tmp_path, monkeypatch):
    model = LocalTransformer(replace(tiny_config(), vocab_size=10))
    path = tmp_path / 'existing.pt'
    save_checkpoint(model, path)
    original = path.read_bytes()

    def interrupted_save(value, temporary):
        Path(temporary).write_bytes(b'incomplete')
        raise OSError('simulated disk error')

    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError):
        save_checkpoint(model, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_optimizer_moments_restored_exactly(tmp_path):
    model = LocalTransformer(replace(tiny_config(), vocab_size=10))
    optimizer = torch.optim.AdamW(model.parameters())
    tokens = torch.tensor([[1, 2, 3, 4]])
    _, loss = model(tokens, tokens)
    loss.backward()
    optimizer.step()
    save_checkpoint(model, tmp_path / 'opt.pt', optimizer=optimizer)
    loaded, cp = load_checkpoint(tmp_path / 'opt.pt')
    restored = torch.optim.AdamW(loaded.parameters())
    assert load_optimizer_state(restored, cp)
    original_state, restored_state = optimizer.state_dict(), restored.state_dict()
    assert original_state['param_groups'] == restored_state['param_groups']
    for key, values in original_state['state'].items():
        for name, tensor in values.items():
            assert torch.equal(tensor, restored_state['state'][key][name])


def cli(*args):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.run([sys.executable, '-m', 'localai', *map(str, args)],
                          cwd=ROOT_DIR, capture_output=True, text=True, encoding='utf-8',
                          timeout=90, env=env)


def test_cli_train_resume_generate_evaluate(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text(corpus(), encoding='utf-8')
    output = tmp_path / 'cli'
    result = cli('train', '--steps', 1, '--data', source, '--output-dir', output,
                 '--batch-size', 1, '--eval-batches', 1)
    assert result.returncode == 0, result.stderr
    result = cli('train', '--steps', 1, '--data', source, '--resume', output / 'last.pt',
                 '--output-dir', tmp_path / 'resumed', '--batch-size', 1, '--eval-batches', 1)
    assert result.returncode == 0, result.stderr
    _, checkpoint = load_checkpoint(tmp_path / 'resumed' / 'last.pt')
    assert checkpoint['step'] == 2
    result = cli('generate', '--checkpoint', output / 'best.pt', '--prompt', 'abc', '--length', 2)
    assert result.returncode == 0 and 'abc' in result.stdout, result.stderr
    result = cli('evaluate', '--checkpoint', output / 'best.pt', '--split', 'test')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['split'] == 'test'
    assert cli('train', '--steps', 0, '--output-dir', tmp_path / 'invalid').returncode != 0
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('version', ['v0.1', 'v0.2'])
def test_existing_cli_generation(version):
    if not (ROOT_DIR / 'models' / 'checkpoints' / f'localai_{version}.pt').exists():
        pytest.skip('선택적인 로컬 기존 모델 자산이 없습니다.')
    result = cli('generate', '--version', version, '--prompt', '인공지능은', '--length', 2)
    assert result.returncode == 0, result.stderr
    assert '인공지능은' in result.stdout


def test_cli_compare_existing_models():
    if not all((ROOT_DIR / 'models' / 'checkpoints' / f'localai_{version}.pt').exists()
               for version in ('v0.1', 'v0.2')):
        pytest.skip('선택적인 로컬 기존 모델 자산이 없습니다.')
    result = cli('compare', '--prompt', '인공지능은', '--length', 1)
    assert result.returncode == 0, result.stderr
    assert '[LocalAI v0.1]' in result.stdout and '[LocalAI v0.2]' in result.stdout
