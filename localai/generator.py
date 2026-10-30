import math

import torch
import torch.nn.functional as F


def _validate_sampling(max_new_tokens, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature는 유한한 양수여야 합니다.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens는 음수가 될 수 없습니다.")


@torch.no_grad()
def sample_tokens(model, tokenizer, prompt_ids, max_new_tokens=100,
                  temperature=0.8, top_k=20, *, greedy=False, top_p=1.0,
                  repetition_penalty=1.0):
    _validate_sampling(max_new_tokens, temperature)
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p는 0 초과 1 이하의 유한한 값이어야 합니다.")
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError("repetition_penalty는 1 이상의 유한한 값이어야 합니다.")
    model.eval()
    device = next(model.parameters()).device
    generated = torch.tensor([prompt_ids or [tokenizer.eos_id]], dtype=torch.long, device=device)
    new_ids = []
    for _ in range(max_new_tokens):
        context = generated[:, -model.config.block_size:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        if new_ids and repetition_penalty != 1:
            previous = list(set(new_ids))
            scores = logits[:, previous]
            logits[:, previous] = torch.where(scores >= 0, scores / repetition_penalty,
                                              scores * repetition_penalty)
        logits[:, tokenizer.stoi[tokenizer.PAD]] = float("-inf")
        logits[:, tokenizer.stoi[tokenizer.UNK]] = float("-inf")
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        if greedy:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            if top_p < 1:
                sorted_logits, indices = torch.sort(logits, descending=True)
                probabilities = F.softmax(sorted_logits, dim=-1)
                remove = probabilities.cumsum(dim=-1) - probabilities > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.empty_like(logits).scatter(1, indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        new_ids.append(next_token.item())
        if next_token.item() == tokenizer.eos_id:
            break
    return new_ids


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100,
                  temperature=0.8, top_k=20):
    ids = tokenizer.encode(prompt)
    generated = sample_tokens(model, tokenizer, ids, max_new_tokens, temperature, top_k)
    return tokenizer.decode(ids + generated)


def dialogue_prompt(messages, tokenizer, block_size):
    from localai.dialogue import encode_messages, validate_messages

    messages = validate_messages(messages, pending_user=True)
    ids, _ = encode_messages(messages, tokenizer, pending_user=True)
    # 오래된 대화 쌍부터 제외하고 이번 질문과 응답 시작 표시는 보존합니다.
    while len(ids) > block_size and len(messages) > 1:
        messages = messages[2:]
        ids, _ = encode_messages(messages, tokenizer, pending_user=True)
    if len(ids) > block_size:
        raise ValueError(f"이번 질문이 모델의 문맥 길이 {block_size}를 초과합니다. 질문을 줄여 주세요.")
    return ids


@torch.no_grad()
def generate_reply(model, tokenizer, messages, max_new_tokens=100,
                   temperature=0.8, top_k=20, *, greedy=False, top_p=1.0,
                   repetition_penalty=1.0):
    ids = dialogue_prompt(messages, tokenizer, model.config.block_size)
    generated = sample_tokens(model, tokenizer, ids, max_new_tokens, temperature, top_k,
                              greedy=greedy, top_p=top_p, repetition_penalty=repetition_penalty)
    # 과거 응답의 EOS에 막히지 않도록 새로 생성한 부분만 디코딩합니다.
    return tokenizer.decode(generated)
