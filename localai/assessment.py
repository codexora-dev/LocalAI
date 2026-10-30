"""고정 질문의 실제 생성 결과를 기록합니다. 문자열 점수는 의미 정확도가 아닙니다."""

from difflib import SequenceMatcher
from pathlib import Path

from localai.checkpoint import load_checkpoint, checkpoint_tokenizer
from localai.dialogue import load_dialogues
from localai.generator import generate_reply


def assess_chat(checkpoint_path, data_path=None, split="val", max_cases=30, length=80):
    if split not in ("train", "val", "test") or max_cases < 1 or length < 1:
        raise ValueError("분할, 문항 수, 생성 길이를 확인하세요.")
    model, checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    tokenizer = checkpoint_tokenizer(model, checkpoint)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("input_format") != "dialogue":
        raise ValueError("대화 체크포인트가 필요합니다.")
    records = load_dialogues(data_path) if data_path else metadata["dataset"]["splits"][split]
    training = metadata["dataset"]["splits"]["train"]
    training_questions = {message["content"].strip() for record in training
                          for message in record["messages"] if message["role"] == "user"}
    training_answers = {message["content"].strip() for record in training
                        for message in record["messages"] if message["role"] == "assistant"}
    cases = []
    for record in records:
        for index in range(1, len(record["messages"]), 2):
            pending = record["messages"][:index]
            reference = record["messages"][index]["content"].strip()
            error = None
            try:
                generated = generate_reply(model, tokenizer, pending, max_new_tokens=length,
                                           greedy=True, repetition_penalty=1.1).strip()
            except ValueError as exc:
                generated, error = "", str(exc)
            cases.append({"id": record["id"], "turn": index // 2 + 1,
                          "prompt": pending[-1]["content"], "reference": reference,
                          "generated": generated, "error": error,
                          "exact_match": generated == reference,
                          "character_similarity": SequenceMatcher(None, reference, generated, autojunk=False).ratio(),
                          "exact_training_question": pending[-1]["content"].strip() in training_questions,
                          "copies_training_answer": bool(generated) and generated in training_answers})
            if len(cases) >= max_cases:
                break
        if len(cases) >= max_cases:
            break
    if not cases:
        raise ValueError("평가할 대화가 없습니다.")
    return {"checkpoint": str(Path(checkpoint_path).resolve()), "step": checkpoint["step"],
            "dataset_signature": metadata["dataset_signature"],
            "scope": "development_probes" if data_path else split,
            "source": str(Path(data_path).resolve()) if data_path else "checkpoint_manifest",
            "warning": "문자열 일치도는 의미 정확도가 아닙니다. 개발 문항은 학습 중복을 확인하세요.",
            "decoding": {"greedy": True, "repetition_penalty": 1.1, "max_new_tokens": length},
            "case_count": len(cases),
            "exact_match_rate": sum(case["exact_match"] for case in cases) / len(cases),
            "mean_character_similarity": sum(case["character_similarity"] for case in cases) / len(cases),
            "empty_rate": sum(not case["generated"] for case in cases) / len(cases),
            "error_count": sum(case["error"] is not None for case in cases), "cases": cases}
