from pathlib import Path
import hashlib
import json
import random
import re
import unicodedata
from difflib import SequenceMatcher
import torch

def load_text(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"학습 데이터 파일이 없습니다: {path}"
        )

    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("학습 데이터가 비어 있습니다")

    return text


def prepare_data(text, tokenizer):
    token_ids = tokenizer.encode(
        text, add_eos=True
    )

    return torch.tensor(
        token_ids,
        dtype=torch.long
    )


def get_batch(data, block_size, batch_size, device, generator=None):
    from localai.dialogue import DialogueDataset
    if isinstance(data, DialogueDataset):
        return data.sample(block_size, batch_size, device, generator)
    if len(data) <= block_size:
        raise ValueError(
            "학습 데이터의 토큰 수가 block_size보다 커야 합니다."
        )

    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
        generator=generator,
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts.tolist()
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts.tolist()
    ])

    return x.to(device), y.to(device)


def split_text(text, validation_ratio=0.1, test_ratio=0.1, seed=42,
               near_duplicate_threshold=0.85):
    """문장을 정규화하고 유사 문장 연결 그룹을 같은 분할에 배치합니다.

    짧은 로컬 말뭉치용 O(n²) 비교이며 의미 기반 중복 검사는 아닙니다.
    """
    if not (0 < validation_ratio < 1 and 0 < test_ratio < 1
            and validation_ratio + test_ratio < 1):
        raise ValueError("검증/테스트 비율의 합은 1보다 작아야 합니다.")
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("유사 문장 기준은 0 초과 1 이하여야 합니다.")
    raw = [" ".join(part.split()) for part in re.split(
        r"(?<=[.!?。！？])\s+|[\r\n]+", unicodedata.normalize("NFC", text))
        if part.strip()]
    units = list(dict.fromkeys(raw))
    parents = list(range(len(units)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    near_pairs = 0
    for i, left in enumerate(units):
        for j in range(i):
            if SequenceMatcher(None, left, units[j], autojunk=False).ratio() >= near_duplicate_threshold:
                parents[root(i)] = root(j)
                near_pairs += 1
    groups = {}
    for i, unit in enumerate(units):
        groups.setdefault(root(i), []).append(unit)
    groups = list(groups.values())
    if len(groups) < 3:
        raise ValueError("독립된 문장 그룹이 3개 이상 필요합니다. 데이터를 늘리세요.")
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * validation_ratio))
    n_test = max(1, round(len(groups) * test_ratio))
    if n_val + n_test >= len(groups):
        raise ValueError("학습 그룹이 남지 않습니다. 분할 비율을 줄이세요.")
    partitions = {"val": groups[:n_val], "test": groups[n_val:n_val+n_test],
                  "train": groups[n_val+n_test:]}
    splits = {name: [unit for group in value for unit in group]
              for name, value in partitions.items()}
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"version": 1, "source_sha256": fingerprint, "seed": seed,
            "validation_ratio": validation_ratio, "test_ratio": test_ratio,
            "near_duplicate_threshold": near_duplicate_threshold,
            "raw_sentences": len(raw), "unique_sentences": len(units),
            "exact_duplicates_removed": len(raw) - len(units),
            "near_duplicate_pairs": near_pairs, "groups": len(groups),
            "splits": splits}


def encode_splits(manifest, tokenizer, block_size, session_augmentation=0):
    if manifest.get("kind") == "dialogue":
        from localai.dialogue import DialogueDataset, augment_training_sessions
        data = {name: DialogueDataset(
                    augment_training_sessions(records, session_augmentation, manifest["seed"])
                    if name == "train" else records, tokenizer, block_size)
                for name, records in manifest["splits"].items()}
        return data, {name: value.stats for name, value in data.items()}
    data = {}
    stats = {}
    for name, units in manifest["splits"].items():
        # 문장 경계를 EOS로 표시하되 서로 다른 분할은 절대 연결하지 않습니다.
        ids = [token for unit in units for token in tokenizer.encode(unit, add_eos=True)]
        if len(ids) <= block_size:
            raise ValueError(f"{name} 분할이 block_size보다 짧습니다. 데이터를 늘리거나 문맥 길이를 줄이세요.")
        data[name] = torch.tensor(ids, dtype=torch.long)
        unknown = sum(token == tokenizer.stoi[tokenizer.UNK] for token in ids)
        stats[name] = {"sentences": len(units), "tokens": len(ids),
                       "unknown_tokens": unknown, "unknown_ratio": unknown / len(ids)}
    return data, stats


def dataset_signature(manifest):
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
