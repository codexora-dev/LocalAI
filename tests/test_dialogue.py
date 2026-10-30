import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from localai import __version__, trainer, generator
from localai.checkpoint import load_checkpoint
from localai.config import ModelConfig, TrainConfig, DEFAULT_DIALOGUE_PATH, ROOT_DIR
from localai.dataset import encode_splits, get_batch
from localai.dialogue import (
    ASSISTANT_PREFIX, USER_PREFIX, DialogueDataset, IGNORE_INDEX,
    load_dialogues, split_dialogues, dialogue_vocab_texts, encode_messages,
)
from localai.metrics import evaluate, evaluate_checkpoint
from localai.model import LocalTransformer
from localai.tokenizer import CharTokenizer


def seed_records():
    return load_dialogues(DEFAULT_DIALOGUE_PATH)


def synthetic_records():
    rng = random.Random(73)
    def content():
        return ''.join(rng.choices('abcdefghijklmnopqrstuvxyz 0123456789', k=40))
    return [{'id': f'test-{i}', 'source': 'synthetic_test', 'messages': [
        {'role': 'user', 'content': content()}, {'role': 'assistant', 'content': content()}]}
        for i in range(12)]


def write_synthetic_data(path):
    path.write_text('\n'.join(json.dumps(record) for record in synthetic_records()), encoding='utf-8')


def tokenizer_for(records):
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(dialogue_vocab_texts(records))
    return tokenizer


def tiny_model_config():
    return ModelConfig(block_size=16, n_embd=8, n_head=2, n_layer=1, dropout=0.0)


def tiny_train_config(**changes):
    return replace(TrainConfig(batch_size=2, max_steps=1, eval_batches=1,
                               log_interval=1, patience=0), **changes)


def test_seed_schema_and_reproducible_group_split():
    records = seed_records()
    assert len(records) == 32
    assert any(len(record['messages']) > 2 for record in records)
    manifest = split_dialogues(records)
    assert manifest == split_dialogues(records)
    sets = {name: {record['id'] for record in values} for name, values in manifest['splits'].items()}
    assert not sets['train'] & sets['val']
    assert not sets['train'] & sets['test']
    assert not sets['test'] & sets['val']
    assert sum(len(value) for value in sets.values()) == 32
    assert split_dialogues(records, seed=123)['splits'] != manifest['splits']


@pytest.mark.parametrize('record', [
    {}, [], {'id': 'x', 'source': 'test', 'messages': []},
    {'id': 'x', 'source': 'test', 'messages': [{'role': 'assistant', 'content': 'a'}]},
    {'id': 'x', 'source': 'test', 'messages': [{'role': 'user', 'content': 'a'}]},
    {'id': 'x', 'source': 'test', 'messages': [
        {'role': 'user', 'content': ''}, {'role': 'assistant', 'content': 'b'}]},
    {'id': 'x', 'source': 'test', 'messages': [
        {'role': 'user', 'content': 'a'}, {'role': 'user', 'content': 'b'}]},
    {'id': 'x', 'source': 'test', 'messages': [
        {'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 123}]},
    {'id': 'x', 'source': '', 'messages': [
        {'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}]},
])
def test_invalid_dialogue_rejected_with_line_number(tmp_path, record):
    path = tmp_path / 'invalid.jsonl'
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='1행'):
        load_dialogues(path)


def test_duplicate_ids_invalid_json_and_empty_file(tmp_path):
    path = tmp_path / 'input.jsonl'
    text = json.dumps(seed_records()[0])
    for content in (text + '\n' + text, '{bad json', ''):
        path.write_text(content, encoding='utf-8')
        with pytest.raises(ValueError):
            load_dialogues(path)


def test_duplicate_and_shared_prompt_grouping():
    records = seed_records()
    duplicate = copy.deepcopy(records[0])
    duplicate['id'] = 'copy'
    duplicate['group_id'] = records[1]['group_id']
    paraphrase = copy.deepcopy(records[0])
    paraphrase['id'] = 'paraphrase'
    paraphrase['messages'][0]['content'] += '!'
    paraphrase['messages'][1]['content'] = '반갑습니다.'
    manifest = split_dialogues(records + [duplicate, paraphrase], near_duplicate_threshold=0.8)
    assert manifest['exact_duplicates_removed'] == 1
    partitions = {record['id']: name for name, values in manifest['splits'].items() for record in values}
    # 제거된 중복의 다른 group_id도 연결 정보로 남아 있어야 합니다.
    assert partitions[records[0]['id']] == partitions[records[1]['id']] == partitions['paraphrase']


def test_insufficient_groups_rejected():
    with pytest.raises(ValueError):
        split_dialogues(seed_records()[:2])
    records = seed_records()
    for record in records:
        record['group_id'] = 'one-source-document'
    with pytest.raises(ValueError):
        split_dialogues(records)


def test_masks_only_response_and_covers_every_target_once():
    records = seed_records()[2:4]
    records[0]['messages'][-1]['content'] *= 15
    tokenizer = tokenizer_for(records)
    dataset = DialogueDataset(records, tokenizer, block_size=16)
    expected = []
    for record in records:
        ids, spans = encode_messages(record['messages'], tokenizer)
        expected.extend(token for start, end in spans for token in ids[start:end])
    actual = [token for _, targets in dataset.windows for token in targets if token != IGNORE_INDEX]
    assert actual == expected
    assert actual.count(tokenizer.eos_id) == sum(len(record['messages']) // 2 for record in records)
    assert dataset.stats['cropped_windows'] > 0
    for inputs, targets in dataset.windows:
        assert 0 < len(inputs) <= 16 and len(inputs) == len(targets)
        assert any(token != IGNORE_INDEX for token in targets)
    x, y = dataset.batch(range(min(4, len(dataset))), 'cpu')
    assert x.shape == y.shape
    assert y.eq(IGNORE_INDEX).any()
    assert y[x.eq(tokenizer.stoi[tokenizer.PAD])].eq(IGNORE_INDEX).all()


def test_prompt_prefix_consistent_and_training_vocab_only():
    record = seed_records()[2]
    tokenizer = tokenizer_for([record])
    ids, spans = encode_messages(record['messages'][:2], tokenizer)
    prompt, _ = encode_messages(record['messages'][:1], tokenizer, pending_user=True)
    assert prompt == ids[:spans[0][0]]
    assert tokenizer.decode(prompt).startswith(USER_PREFIX)
    assert tokenizer.decode(prompt).endswith(ASSISTANT_PREFIX)
    manifest = split_dialogues(seed_records())
    manifest['splits']['test'][0]['messages'][1]['content'] += 'Ω'
    tokenizer = tokenizer_for(manifest['splits']['train'])
    assert 'Ω' not in tokenizer.stoi
    data, stats = encode_splits(manifest, tokenizer, 16)
    assert stats['train']['unknown_tokens'] == 0
    assert stats['test']['response_unknown_tokens'] > 0
    x, y = get_batch(data['train'], 16, 2, 'cpu', torch.Generator().manual_seed(7))
    assert x.shape == y.shape and y.ne(IGNORE_INDEX).any()


def test_masked_loss_and_full_token_weighted_evaluation():
    records = seed_records()[:3]
    tokenizer = tokenizer_for(records)
    dataset = DialogueDataset(records, tokenizer, 16)
    model = LocalTransformer(replace(tiny_model_config(), vocab_size=tokenizer.vocab_size))
    x, y = dataset.batch([0, 1], 'cpu')
    logits, loss = model(x, y)
    mask = y.ne(IGNORE_INDEX)
    expected = F.cross_entropy(logits[mask], y[mask])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert model.lm_head.weight.grad is not None
    state = torch.get_rng_state().clone()
    first = evaluate(model, dataset, dataset, batch_size=1, eval_batches=1)
    second = evaluate(model, dataset, dataset, batch_size=3, eval_batches=9)
    assert first['val'] == pytest.approx(second['val'], abs=1e-5)
    assert torch.equal(state, torch.get_rng_state()) and model.training
    assert math.isfinite(first['val'])


def test_dialogue_history_eos_and_context_trimming(monkeypatch):
    records = seed_records()[2:4]
    tokenizer = tokenizer_for(records)
    messages = records[0]['messages'][:2] + [records[0]['messages'][2]]
    prompt = generator.dialogue_prompt(messages, tokenizer, 200)
    assert tokenizer.eos_id in prompt
    model = LocalTransformer(replace(tiny_model_config(), vocab_size=tokenizer.vocab_size, block_size=200))
    captured = []

    def fixed_sampling(model, tokenizer, ids, *args, **kwargs):
        captured.append(ids)
        return tokenizer.encode('다섯 개') + [tokenizer.eos_id]

    monkeypatch.setattr(generator, 'sample_tokens', fixed_sampling)
    assert generator.generate_reply(model, tokenizer, messages) == '다섯 개'
    assert tokenizer.eos_id in captured[0]
    newest, _ = encode_messages(messages[-1:], tokenizer, pending_user=True)
    assert generator.dialogue_prompt(messages, tokenizer, len(newest)) == newest
    with pytest.raises(ValueError, match='문맥'):
        generator.dialogue_prompt(messages, tokenizer, len(newest) - 1)


def test_sample_eos_and_empty_prompt_generation():
    tokenizer = tokenizer_for(seed_records()[:1])

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=16)
            self.calls = 0

        def forward(self, tokens):
            self.calls += 1
            logits = torch.full((*tokens.shape, tokenizer.vocab_size), -1e9)
            token = tokenizer.stoi['안'] if self.calls == 1 else tokenizer.eos_id
            logits[:, :, token] = 0
            return logits, None

    model = FixedModel()
    assert generator.generate_text(model, tokenizer, '', max_new_tokens=8) == '안'
    assert model.calls == 2
    with pytest.raises(ValueError):
        generator.sample_tokens(model, tokenizer, [], temperature=float('nan'))


def test_dialogue_training_early_stopping_resume_and_evaluation(tmp_path, monkeypatch):
    output = tmp_path / 'dialogue'
    source = tmp_path / 'synthetic.jsonl'
    write_synthetic_data(source)
    original_evaluate = trainer.evaluate
    val_losses = iter([3.0, 2.0, 2.1])
    seen_train = []
    expected_manifest = split_dialogues(load_dialogues(source))
    expected_tokenizer = tokenizer_for(expected_manifest['splits']['train'])
    expected_data, _ = encode_splits(expected_manifest, expected_tokenizer, tiny_model_config().block_size)

    def controlled_evaluate(model, train_data, val_data, **kwargs):
        assert isinstance(train_data, DialogueDataset) and isinstance(val_data, DialogueDataset)
        assert train_data.windows == expected_data['train'].windows
        assert val_data.windows == expected_data['val'].windows
        seen_train.append(train_data)
        return {'train': 1.5, 'val': next(val_losses)}

    monkeypatch.setattr(trainer, 'evaluate', controlled_evaluate)
    trainer.train(data_format='dialogue', data_path=source, output_dir=output, model_config=tiny_model_config(),
                  train_config=tiny_train_config(max_steps=10, patience=1))
    meta = json.loads((output / 'experiment.json').read_text(encoding='utf-8'))
    assert meta['input_format'] == 'dialogue' and meta['model_version'] == 'v0.3'
    assert meta['stop_reason'] == 'early_stopping' and meta['best_step'] == 1 and meta['step'] == 2
    assert meta['loss_scope'] == 'assistant_content_and_eos'
    best_model, best = load_checkpoint(output / 'best.pt')
    assert best['step'] == 1
    digest = hashlib.sha256((output / 'last.pt').read_bytes()).hexdigest()
    monkeypatch.setattr(trainer, 'evaluate', original_evaluate)
    trainer.train(data_format='dialogue', data_path=source, resume=output / 'last.pt', output_dir=tmp_path / 'resumed',
                  train_config=tiny_train_config())
    _, resumed = load_checkpoint(tmp_path / 'resumed' / 'last.pt')
    assert resumed['step'] == 3 and resumed['metadata']['optimizer_restored']
    assert all(int(value['step']) == 3 for value in resumed['optimizer_state']['state'].values())
    assert hashlib.sha256((output / 'last.pt').read_bytes()).hexdigest() == digest
    report = evaluate_checkpoint(output / 'best.pt', 'val')
    assert report['input_format'] == 'dialogue' and report['loss_scope'] == 'assistant_content_and_eos'
    assert report['evaluation']['method'] == 'full_response_windows_v1'
    assert evaluate_checkpoint(output / 'best.pt', 'test')['split'] == 'test'
    with pytest.raises(ValueError, match='형식'):
        trainer.train(resume=output / 'last.pt', output_dir=tmp_path / 'invalid-format')


def test_text_checkpoint_cannot_be_used_as_dialogue(tmp_path):
    from localai.checkpoint import save_checkpoint
    from localai.__main__ import run_chat

    path = tmp_path / 'text.pt'
    model = LocalTransformer(replace(tiny_model_config(), vocab_size=10))
    save_checkpoint(model, path)
    with pytest.raises(ValueError, match='형식'):
        trainer.train(data_format='dialogue', resume=path)
    with pytest.raises(ValueError, match='dialogue'):
        run_chat(path, prompt='안녕')


def cli(*args, input_text=None):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.run([sys.executable, '-m', 'localai', *map(str, args)], cwd=ROOT_DIR,
                          input=input_text, capture_output=True, text=True, encoding='utf-8',
                          timeout=90, env=env)


def test_cli_v03_inspect_train_chat_resume_and_evaluate(tmp_path):
    version = cli('--version')
    assert version.returncode == 0 and __version__ in version.stdout
    inspection = cli('inspect-data')
    assert inspection.returncode == 0, inspection.stderr
    report = json.loads(inspection.stdout)
    assert report['kind'] == 'dialogue' and report['raw_conversations'] == 32
    output = tmp_path / 'run'
    source = tmp_path / 'synthetic.jsonl'
    write_synthetic_data(source)
    result = cli('train', '--format', 'dialogue', '--data', source, '--steps', 1, '--batch-size', 1,
                 '--output-dir', output, '--dropout', .2, '--prompt-loss-weight', .1,
                 '--label-smoothing', .1, '--max-seconds', 60, '--session-augmentation', 1)
    assert result.returncode == 0, result.stderr
    result = cli('chat', '--checkpoint', output / 'best.pt', '--prompt', '안녕', '--length', 2)
    assert result.returncode == 0 and 'v0.3' in result.stdout, result.stderr
    result = cli('chat', '--checkpoint', output / 'best.pt', '--length', 2,
                 input_text='안녕\n/reset\n안녕\n/exit\n')
    assert result.returncode == 0 and '초기화' in result.stdout, result.stderr
    result = cli('train', '--format', 'dialogue', '--data', source, '--resume', output / 'last.pt', '--steps', 1,
                 '--batch-size', 1, '--output-dir', tmp_path / 'resume')
    assert result.returncode == 0, result.stderr
    _, checkpoint = load_checkpoint(tmp_path / 'resume' / 'last.pt')
    assert checkpoint['step'] == 2
    assert checkpoint['model_config']['dropout'] == .2
    result = cli('evaluate', '--checkpoint', output / 'best.pt', '--split', 'test')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['loss_scope'] == 'assistant_content_and_eos'
    report_path = tmp_path / 'assessment.json'
    result = cli('assess-chat', '--checkpoint', output / 'best.pt', '--max-cases', 1,
                 '--length', 2, '--output', report_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(report_path.read_text(encoding='utf-8'))['scope'] == 'val'
    result = cli('assess-chat', '--checkpoint', output / 'best.pt', '--output', report_path)
    assert result.returncode != 0
