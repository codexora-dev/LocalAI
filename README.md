# LocalAI

PyTorch로 직접 구현한 Decoder-only Transformer를 무작위 가중치부터 학습하는 로컬 AI 연구 프로젝트입니다.
현재 소프트웨어 버전은 **v0.3.1**입니다.

저장소: [codexora-dev/LocalAI](https://github.com/codexora-dev/LocalAI)

사전학습 LLM 가중치나 외부 AI 추론 API를 사용하지 않습니다.
개발에는 Codex를 사용했지만 LocalAI 실행에는 Codex가 필요하지 않습니다.
필요한 패키지가 설치된 뒤 핵심 학습·생성·평가 기능은 인터넷 없이 실행됩니다.
실행 시 모델 다운로드나 대화 업로드를 하지 않습니다.

## 현재 가능한 것과 한계

- 한국어 문자 단위 토크나이저와 자체 causal Transformer.
- 텍스트 및 대화 JSONL 학습, 검증·테스트 분리, early stopping.
- 실험별 best/last 체크포인트와 optimizer 상태 저장·이어 학습.
- 응답 전용 손실, 선택적 문맥 보조 손실·label smoothing·합성 세션 증강.
- CLI 텍스트 생성, 현재 세션 문맥을 전달하는 대화, 손실·실제 응답 평가.

**실용적인 자유 대화 모델은 아직 아닙니다.**
작은 합성 자료에서 학습 예문을 재현하는 성향이 강하고,
새 질문이나 긴 문맥에서는 어색하거나 틀린 답을 생성합니다.
테스트 코드가 통과한다는 사실은 대화 정확도나 언어 이해를 입증하지 않습니다.
Windows 제어, 웹 검색, 장기 기억, GUI는 아직 구현하지 않았습니다.

## 실행 환경

개발·검증 환경은 Windows 11, Python 3.14.7, PyTorch 2.12.1+cpu입니다.
CUDA나 NVIDIA GPU는 필수가 아닙니다. 가상환경도 필수로 요구하지 않습니다.
의존성 설치에는 인터넷이 필요할 수 있으며 런타임의 오프라인 실행과는 별개입니다.

프로젝트 루트의 PowerShell에서:

```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 -m localai --version
py -3.14 -m localai inspect-data
```

다른 OS나 Python 버전은 별도로 검증하지 않았습니다.

## 새 모델 학습과 대화

**학습된 체크포인트와 로컬 실험 로그는 이 저장소에 포함하지 않습니다.**
새로 내려받은 저장소에서 대화하려면 먼저 로컬 학습을 실행해야 합니다.

```powershell
py -3.14 -m localai train --format dialogue --data data/raw/dialogue_basic_v031.jsonl --steps 200 --output-dir logs/experiments/first_run
py -3.14 -m localai chat --checkpoint logs/experiments/first_run/best.pt
```

`first_run`은 존재하지 않는 새 디렉터리여야 합니다.
이 명령은 학습과 실행 경로를 확인하는 예시이며 200스텝으로 자연스러운 대화가 보장되지 않습니다.
`chat`에서 `/exit`로 종료하고 `/reset`으로 현재 문맥을 초기화합니다.
대화는 자동 저장되거나 신경망에 자동 학습되지 않습니다.

검증 손실과 실제 응답 확인:

```powershell
py -3.14 -m localai evaluate --checkpoint logs/experiments/first_run/best.pt --split val
py -3.14 -m localai assess-chat --checkpoint logs/experiments/first_run/best.pt --split val
```

모델 선택에는 검증 자료를 사용하고 테스트 자료는 최종 평가용으로 남겨 두세요.
예측 문장과 참조 문장의 문자열 일치도는 의미 정확도와 다릅니다.
과거 문서의 버전별 모델·시연 모델 경로는 개발자의 로컬 자산을 가리킬 수 있으며 복제본에는 없습니다.

## 테스트

```powershell
py -3.14 -m pytest tests -v
```

기존 v0.1/v0.2 로컬 모델을 요구하는 선택적 테스트는 해당 파일이 없으면 건너뜁니다.
나머지 학습·복원·CLI 검사는 작은 임시 모델과 합성 데이터를 사용합니다.
기본 테스트는 모델을 다운로드하지 않습니다.

## 데이터

동봉 자료는 소규모 개발 예제입니다. 실제 사용자 대화나 대규모 독립 평가 자료가 아닙니다.
`dialogue_seed_v03.jsonl`과 `dialogue_basic_v031.jsonl`은 개발 중 작성한 합성 대화입니다.
개발 문항 `dialogue_development_v031.jsonl`은 학습 예제와 겹치므로 독립 테스트로 사용하지 마세요.
출처와 사용 범위를 기록하고, 검증·테스트와 중복되는 예제를 학습에 넣지 않아야 합니다.
추가 로컬 데이터는 기본적으로 Git에서 제외하며 공개할 자료는 별도로 검토해야 합니다.

## 코드 구조와 문서

- `localai/`: 모델, 토크나이저, 데이터 처리, 학습·생성·평가와 CLI.
- `tests/`: 단위 및 CLI 회귀 테스트.
- `data/`: 공개용 예제 자료.
- [TRAINING.md](TRAINING.md): 기존 텍스트 학습과 실험 관리.
- [V03.md](V03.md): 대화 JSONL 형식과 대화 학습 사용법.
- [QUALITY_REPORT.md](QUALITY_REPORT.md): v0.3.1 실험 결과와 미해결 한계.
- [DEVELOPMENT_REPORT.md](DEVELOPMENT_REPORT.md): 초기 학습·평가 파이프라인 개선 기록.
- [AGENTS.md](AGENTS.md): 개발 지침. 과거 상태 설명은 당시 관찰 기록입니다.

## Git에 포함하지 않는 것

모델·어휘·로그·백업, 로컬 피드백과 DB, 환경 변수·키 파일, 캐시는 `.gitignore`로 제외합니다.
공개하는 예제 데이터만 명시적으로 포함합니다.
로컬 실험 자산은 원본 위치에 유지되며 Git 설정이 파일을 삭제하지 않습니다.

프로젝트 루트에서 변경 사항을 올립니다.

```powershell
git add .
git commit -m "Update LocalAI"
git push
```

공개 전 `git status`로 포함 파일을 확인하세요. 모델과 대화 기록은 기본적으로 제외됩니다.

## 라이선스

프로젝트 코드와 동봉 예제는 [MIT License](LICENSE)로 제공합니다.
새 외부 자료나 의존성을 추가할 때는 해당 자료의 이용 조건도 별도로 확인하세요.
