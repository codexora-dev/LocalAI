import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from localai import trainer
from localai.config import ModelConfig, TrainConfig
from localai.model import dialogue_training_loss
from localai.generator import sample_tokens
from localai.tokenizer import CharTokenizer
from localai.checkpoint import save_checkpoint
from localai.model import LocalTransformer


def test_context_auxiliary_loss_matches_manual_calculation():
    inputs = torch.tensor([[3, 4, 5, 6], [3, 4, 0, 0]])
    targets = torch.tensor([[-100, -100, 6, 1], [-100, 1, -100, -100]])
    logits = torch.randn(2, 4, 8, requires_grad=True)
    response_only = dialogue_training_loss(logits, inputs, targets, 0, 0.0)
    assert torch.allclose(response_only, F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
    labels = torch.tensor([[4, 5, 6, 1], [4, 1, -100, -100]])
    weights = torch.tensor([[.25, .25, 1, 1], [.25, 1, 0, 0]])
    expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').reshape(2, 4)
    expected = (expected * weights).sum() / weights.sum()
    actual = dialogue_training_loss(logits, inputs, targets, 0, .25)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad[1, 2:].eq(0).all()


def test_label_smoothing_uses_response_mask():
    inputs = torch.tensor([[3, 4, 5]])
    targets = torch.tensor([[-100, 5, 1]])
    logits = torch.randn(1, 3, 8)
    expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), label_smoothing=.1)
    assert torch.allclose(dialogue_training_loss(logits, inputs, targets, 0, 0, .1), expected)


def test_greedy_and_nucleus_sampling():
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(['가나'])

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(block_size=16)

        def forward(self, inputs):
            logits = torch.full((*inputs.shape, tokenizer.vocab_size), -10.)
            logits[..., tokenizer.stoi['가']] = 2
            logits[..., tokenizer.stoi['나']] = 1
            return logits, None

    model = FixedModel()
    first = sample_tokens(model, tokenizer, [], max_new_tokens=3, greedy=True)
    torch.manual_seed(999)
    assert sample_tokens(model, tokenizer, [], max_new_tokens=3, greedy=True) == first
    assert sample_tokens(model, tokenizer, [], max_new_tokens=3, top_p=.01) == [tokenizer.stoi['가']] * 3
    changed = sample_tokens(model, tokenizer, [], max_new_tokens=2, greedy=True, repetition_penalty=3)
    assert changed == [tokenizer.stoi['가'], tokenizer.stoi['나']]
    with pytest.raises(ValueError):
        sample_tokens(model, tokenizer, [], top_p=0)
    with pytest.raises(ValueError):
        sample_tokens(model, tokenizer, [], repetition_penalty=.5)


@pytest.mark.parametrize('changes', [dict(prompt_loss_weight=-.1), dict(prompt_loss_weight=1.1),
    dict(prompt_loss_weight=float('nan')), dict(max_seconds=0), dict(max_seconds=float('inf'))])
def test_quality_configuration_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        replace(TrainConfig(), **changes).validate()


def test_training_time_budget_saves_step_zero(tmp_path, monkeypatch):
    records = [{'id':str(i), 'source':'synthetic', 'messages':[
        {'role':'user','content':question}, {'role':'assistant','content':answer}]}
        for i,(question,answer) in enumerate([
            ('orange apples fruit', 'a short answer'),
            ('programming loops syntax', 'another response'),
            ('weather clouds sunshine', 'third synthetic response')])]
    path = tmp_path / 'data.jsonl'
    path.write_text('\n'.join(json.dumps(record) for record in records), encoding='utf-8')
    values = iter([0.0, 2.0, 3.0])
    monkeypatch.setattr(trainer, 'time', SimpleNamespace(monotonic=lambda: next(values)))
    trainer.train(data_format='dialogue', data_path=path, output_dir=tmp_path/'budget',
                  model_config=ModelConfig(block_size=8,n_embd=8,n_head=2,n_layer=1,dropout=0),
                  train_config=TrainConfig(batch_size=1,max_steps=10,max_seconds=1))
    metadata=json.loads((tmp_path/'budget'/'experiment.json').read_text(encoding='utf-8'))
    assert metadata['stop_reason']=='time_budget' and metadata['step']==0
    assert (tmp_path/'budget'/'best.pt').exists() and (tmp_path/'budget'/'last.pt').exists()


def test_assessment_labels_overlap_and_keeps_reference_out_of_prompt(tmp_path, monkeypatch):
    from localai import assessment
    question = 'synthetic question'
    reference = 'secret reference answer'
    record = {'id':'one', 'source':'synthetic', 'messages':[
        {'role':'user','content':question}, {'role':'assistant','content':reference}]}
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([question, reference])
    model = LocalTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, block_size=32,
                                         n_embd=8,n_head=2,n_layer=1,dropout=0))
    path = tmp_path/'model.pt'
    metadata = {'input_format':'dialogue', 'dataset_signature':'synthetic',
                'dataset':{'splits':{'train':[record], 'val':[record], 'test':[]}}}
    save_checkpoint(model,path,metadata=metadata,tokenizer=tokenizer)
    def prediction(model, tokenizer, messages, **kwargs):
        assert messages == [{'role':'user','content':question}]
        return reference
    monkeypatch.setattr(assessment,'generate_reply',prediction)
    report=assessment.assess_chat(path,max_cases=1)
    assert report['scope']=='val' and report['exact_match_rate']==1
    assert report['cases'][0]['exact_training_question']
    assert report['cases'][0]['copies_training_answer']


def test_session_augmentation_uses_training_partition_only():
    from localai.dialogue import augment_training_sessions, split_dialogues, dialogue_vocab_texts
    from localai.dataset import encode_splits
    records = [{'id': str(i), 'source': 'synthetic', 'messages':[
        {'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]}
        for i, (question, answer) in enumerate([
            ('apple orange food', 'fruit answer'), ('loops python syntax', 'code answer'),
            ('cloud sky weather', 'weather answer'), ('book page reading', 'reading answer')])]
    manifest = split_dialogues(records)
    original = json.dumps(manifest, sort_keys=True)
    training = manifest['splits']['train']
    augmented = augment_training_sessions(training, 2)
    assert augmented == augment_training_sessions(training, 2)
    assert len(augmented) == len(training) * 3
    allowed = {json.dumps(message, sort_keys=True) for record in training for message in record['messages']}
    assert all(json.dumps(message, sort_keys=True) in allowed
               for record in augmented for message in record['messages'])
    assert any(len(record['messages']) > 2 for record in augmented)
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(dialogue_vocab_texts(training))
    plain, _ = encode_splits(manifest, tokenizer, 16)
    mixed, _ = encode_splits(manifest, tokenizer, 16, session_augmentation=2)
    assert plain['val'].windows == mixed['val'].windows
    assert plain['test'].windows == mixed['test'].windows
    assert len(mixed['train']) > len(plain['train'])
    assert json.dumps(manifest, sort_keys=True) == original
