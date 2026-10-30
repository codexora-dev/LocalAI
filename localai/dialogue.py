"""검증된 로컬 대화 데이터, 분할 및 응답 전용 학습 표본."""

import hashlib
import json
import random
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import torch

USER_PREFIX = "사용자: "
ASSISTANT_PREFIX = "도우미: "
DIALOGUE_FORMAT = "korean_roles_v1"
IGNORE_INDEX = -100


def validate_messages(messages, *, pending_user=False):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages는 비어 있지 않은 목록이어야 합니다.")
    expected_end = 1 if pending_user else 0
    if len(messages) % 2 != expected_end:
        raise ValueError("대화는 사용자로 시작하고 학습 자료는 응답으로 끝나야 합니다.")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("각 메시지에는 role과 content만 있어야 합니다.")
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise ValueError("role은 user와 assistant가 번갈아 나와야 합니다.")
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content는 비어 있지 않은 문자열이어야 합니다.")
        normalized.append({"role": expected_role, "content": unicodedata.normalize("NFC", content).strip()})
    return normalized


def load_dialogues(path):
    records, identifiers = [], set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or not {"id", "source", "messages"} <= set(record):
                raise ValueError("id, source, messages가 필요합니다.")
            if set(record) - {"id", "source", "group_id", "messages"}:
                raise ValueError("지원하지 않는 대화 필드가 있습니다.")
            for field in ("id", "source", "group_id"):
                if field in record and (not isinstance(record[field], str) or not record[field].strip()):
                    raise ValueError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
            record = dict(record, messages=validate_messages(record["messages"]))
            if record["id"] in identifiers:
                raise ValueError("대화 id가 중복됩니다.")
            identifiers.add(record["id"])
            records.append(record)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"대화 데이터 {line_number}행: {exc}") from exc
    if not records:
        raise ValueError("대화 데이터가 비어 있습니다.")
    return records


def split_dialogues(records, validation_ratio=0.1, test_ratio=0.1, seed=42,
                    near_duplicate_threshold=0.85):
    if not (0 < validation_ratio < 1 and 0 < test_ratio < 1
            and validation_ratio + test_ratio < 1):
        raise ValueError("검증/테스트 분할 비율이 잘못되었습니다.")
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("유사 문장 기준은 0 초과 1 이하여야 합니다.")
    # 중복 사본의 group_id까지 연결한 뒤 제거하여 그룹 연결 정보는 보존합니다.
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    prompts = [[" ".join(message["content"].split()).casefold()
                for message in record["messages"] if message["role"] == "user"]
               for record in records]
    canonical = [json.dumps(record["messages"], ensure_ascii=False, sort_keys=True) for record in records]
    similar_pairs = 0
    for i, record in enumerate(records):
        for j in range(i):
            same_group = record.get("group_id") is not None and record["group_id"] == records[j].get("group_id")
            similar = any(SequenceMatcher(None, left, right, autojunk=False).ratio() >= near_duplicate_threshold
                          for left in prompts[i] for right in prompts[j])
            if similar:
                similar_pairs += 1
            if same_group or similar or canonical[i] == canonical[j]:
                parents[root(i)] = root(j)
    groups, seen = {}, set()
    for i, record in enumerate(records):
        if canonical[i] in seen:
            continue
        seen.add(canonical[i])
        groups.setdefault(root(i), []).append(record)
    groups = list(groups.values())
    if len(groups) < 3:
        raise ValueError("독립된 대화 그룹이 3개 이상 필요합니다. 관련 질문을 복제하지 말고 데이터를 늘리세요.")
    random.Random(seed).shuffle(groups)
    val_count = max(1, round(len(groups) * validation_ratio))
    test_count = max(1, round(len(groups) * test_ratio))
    if val_count + test_count >= len(groups):
        raise ValueError("학습 대화 그룹이 남지 않습니다.")
    partitions = {"val": groups[:val_count], "test": groups[val_count:val_count+test_count],
                  "train": groups[val_count+test_count:]}
    source = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"version": 1, "kind": "dialogue", "format": DIALOGUE_FORMAT,
            "source_sha256": hashlib.sha256(source).hexdigest(), "seed": seed,
            "validation_ratio": validation_ratio, "test_ratio": test_ratio,
            "near_duplicate_threshold": near_duplicate_threshold,
            "raw_conversations": len(records), "unique_conversations": len(seen),
            "exact_duplicates_removed": len(records) - len(seen),
            "near_duplicate_pairs": similar_pairs, "groups": len(groups),
            "splits": {name: [record for group in value for record in group]
                       for name, value in partitions.items()}}


def dialogue_vocab_texts(records):
    return [USER_PREFIX, ASSISTANT_PREFIX, "\n"] + [
        message["content"] for record in records for message in record["messages"]]


def augment_training_sessions(records, count_per_record=0, seed=42):
    """분할이 확정된 학습 자료만 사용해 주제가 전환되는 합성 세션을 만듭니다."""
    if not 0 <= count_per_record <= 4:
        raise ValueError("세션 증강 수는 대화당 0 이상 4 이하여야 합니다.")
    if not count_per_record:
        return list(records)
    if len(records) < 2:
        raise ValueError("세션 증강에는 학습 대화가 두 개 이상 필요합니다.")
    rng = random.Random(seed)
    augmented = list(records)
    for index in range(len(records) * count_per_record):
        selected = rng.sample(records, k=min(len(records), rng.choice([2, 3])))
        augmented.append({"id": f"augmented-session-{index}", "source": "training_split_session_augmentation",
                          "messages": [dict(message) for record in selected for message in record["messages"]]})
    return augmented


def encode_messages(messages, tokenizer, *, pending_user=False):
    """전체 입력과 정답 응답의 시작/끝 위치를 반환합니다."""
    messages = validate_messages(messages, pending_user=pending_user)
    ids, spans = [], []
    for message in messages:
        prefix = USER_PREFIX if message["role"] == "user" else ASSISTANT_PREFIX
        ids.extend(tokenizer.encode(prefix))
        start = len(ids)
        ids.extend(tokenizer.encode(message["content"]))
        if message["role"] == "assistant":
            ids.append(tokenizer.eos_id)
            spans.append((start, len(ids)))
        ids.extend(tokenizer.encode("\n"))
    if pending_user:
        ids.extend(tokenizer.encode(ASSISTANT_PREFIX))
    return ids, spans


class DialogueDataset:
    def __init__(self, records, tokenizer, block_size):
        self.windows = []
        self.block_size = block_size
        self.pad_id = tokenizer.stoi[tokenizer.PAD]
        total_tokens = unknown = response_unknown = supervised = cropped = 0
        for record in records:
            ids, spans = encode_messages(record["messages"], tokenizer)
            total_tokens += len(ids)
            unknown += ids.count(tokenizer.stoi[tokenizer.UNK])
            for response_start, response_end in spans:
                response_unknown += ids[response_start:response_end].count(tokenizer.stoi[tokenizer.UNK])
                # 긴 응답도 빠뜨리지 않고 나누어 각 정답 토큰을 한 번씩 학습합니다.
                for target_start in range(response_start, response_end, max(1, block_size // 2)):
                    target_end = min(response_end, target_start + max(1, block_size // 2))
                    input_start = max(0, target_end - 1 - block_size)
                    x = ids[input_start:target_end-1]
                    y = [ids[position] if position >= target_start else IGNORE_INDEX
                         for position in range(input_start+1, target_end)]
                    self.windows.append((x, y))
                    supervised += target_end - target_start
                    cropped += input_start > 0
        if not self.windows:
            raise ValueError("학습할 AI 응답 토큰이 없습니다.")
        self.stats = {"conversations": len(records), "tokens": total_tokens,
                      "windows": len(self.windows), "supervised_tokens": supervised,
                      "cropped_windows": cropped, "unknown_tokens": unknown,
                      "unknown_ratio": unknown / total_tokens,
                      "response_unknown_tokens": response_unknown,
                      "response_unknown_ratio": response_unknown / supervised}

    def __len__(self):
        return len(self.windows)

    def batch(self, indices, device):
        selected = [self.windows[index] for index in indices]
        length = max(len(x) for x, _ in selected)
        x = torch.full((len(selected), length), self.pad_id, dtype=torch.long)
        y = torch.full_like(x, IGNORE_INDEX)
        for row, (inputs, targets) in enumerate(selected):
            x[row, :len(inputs)] = torch.tensor(inputs)
            y[row, :len(targets)] = torch.tensor(targets)
        return x.to(device), y.to(device)

    def sample(self, block_size, batch_size, device, generator=None):
        if block_size != self.block_size:
            raise ValueError("대화 데이터와 모델의 문맥 길이가 다릅니다.")
        indices = torch.randint(len(self), (batch_size,), generator=generator).tolist()
        return self.batch(indices, device)
