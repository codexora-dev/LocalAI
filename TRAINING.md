# LocalAI v0.2 학습·평가 사용법

v0.3 대화 학습·`chat` 명령은 [V03.md](V03.md)를 참고하세요.
아래의 기존 텍스트 학습 명령은 계속 사용할 수 있습니다.

기존 Decoder-only Transformer와 문자 토크나이저를 그대로 사용합니다.
사전학습 가중치, 외부 추론 API, 네트워크 연결, 추가 의존성을 도입하지 않았습니다.

## 새 학습

```powershell
py -3.14 -m localai train --steps 200 --patience 5 --min-delta 0.001 --eval-interval 20
```

기본 입력은 `data/raw/starter.txt`와 존재할 경우 `additional.txt`입니다.
`--data 경로`를 지정하면 해당 파일만 읽습니다.
`--output-dir 경로`는 **존재하지 않는 새 폴더**여야 합니다.
생략하면 `logs/experiments/<UTC시간_UUID>/`가 생성됩니다. ID의 시간은 UTC입니다.
기존 버전별 모델·어휘·CSV 경로에는 학습 결과를 쓰지 않습니다.

실험 폴더의 파일:

| 파일 | 내용 |
|---|---|
| `best.pt` | 이번 실험에서 실제 최저 검증 손실을 기록한 모델과 optimizer |
| `last.pt` | 종료 시점 모델과 optimizer |
| `vocab.json` | 학습 분할로만 만든 어휘, 이어 학습은 기존 어휘 고정 |
| `dataset.json` | 원문 SHA-256, 분할 설정, 문장별 고정 분할 |
| `experiment.json` | 고유 ID, 모델·학습 설정, 분할 통계, 평가 방식, best/최종 step, 종료 이유 |
| `metrics.csv` | step별 train/val loss, val perplexity, best val loss |
| `history.jsonl` | 평가별 실험 ID, 손실, best step과 체크포인트 경로 |

체크포인트에는 어휘 매핑, 실험 설정과 데이터 분할도 들어 있습니다.
문장 텍스트를 포함하므로 실험 폴더도 로컬 데이터로 취급하세요.
새 JSON, 어휘, 체크포인트는 같은 디렉터리의 임시 파일에 먼저 저장하고 교체합니다.
실패한 실행은 `experiment.json`에 `failed`와 예외 종류를 기록합니다.
실패·중단한 실행은 `last.pt`가 없을 수 있으며, 이미 저장된 `best.pt`는 남습니다.

## 분할과 평가

NFC와 공백을 정규화하고 줄바꿈 또는 문장 종결부호 뒤 공백을 경계로 문장을 나눕니다.
정확한 중복은 제거하고, 문자열 유사도(기본 0.85)로 연결된 문장은 같은 그룹에 묶습니다.
그룹을 고정 seed로 섞어 기본 학습/검증/테스트 약 80/10/10으로 나눕니다.
분할 비율은 문장 수·토큰 수가 아닌 **그룹 수** 기준이며 반올림됩니다.
데이터가 짧으면 오류를 내며 다른 분할을 복사해서 보충하지 않습니다.
각 분할 안에서 문장을 EOS로 연결해 토큰 창을 만듭니다.

평가는 고정 seed의 무작위 창 표본 평균입니다. 평가가 학습용 난수열을 소모하지 않습니다.
기본 표본은 분할당 `batch_size × eval_batches`개 창입니다.
모든 토큰을 한 번씩 평가하는 전체 말뭉치 손실은 아니며 창은 겹칠 수 있습니다.
비교할 때 데이터 서명, 어휘 매핑, 문맥 길이, seed, batch size, 평가 창 수를 맞추세요.
어휘가 다른 모델의 토큰 perplexity는 직접적인 순위 근거로 삼을 수 없습니다.
검증·테스트의 미등록 문자는 `<unk>`로 처리하며 비율을 별도로 기록합니다.

`patience`는 검증 평가 횟수 기준입니다. 0이면 조기 종료를 끕니다.
`min_delta`보다 큰 누적 개선이 있어야 patience를 초기화합니다.
작은 개선도 실제 최저 손실이면 best 모델로 저장합니다.
학습 전 step 0(이어 학습이면 시작 step)도 평가하므로 최초 상태가 best일 수 있습니다.
마지막 step과 지정 평가 주기에 평가합니다.
현재 실험의 **테스트 손실은 학습 루프에서 계산하지 않습니다.**

## 새 실험의 생성과 이어 학습

```powershell
py -3.14 -m localai generate --checkpoint logs/experiments/<실험ID>/best.pt --prompt "인공지능은" --length 80
py -3.14 -m localai train --resume logs/experiments/<실험ID>/last.pt --steps 100
```

이어 학습도 새 실험을 만듭니다. 원문과 분할 설정(seed, 비율, 유사도 기준)이 같아야 합니다.
원래 실행에서 `--data`나 분할 설정을 지정했다면 이어 학습에도 같은 값을 전달하세요.
가중치·optimizer 모멘텀·누적 step을 복원하고, 학습률은 이번 실행의 설정을 적용합니다.
기본 학습률은 0.001입니다. 다른 값을 원하면 `--learning-rate`를 지정하세요.
난수열과 early stopping 카운터는 새 실험 기준으로 시작합니다.
중단 없이 계속 학습했을 때와 동일한 수치 궤적을 보장하는 재개 기능은 아닙니다.

기존 `generate --version v0.1/v0.2`와 `compare`는 보존한 기존 모델을 사용합니다.
`compare`는 출력 샘플 비교이며 정량 성능 순위 평가가 아닙니다.

기존 모델의 이어 학습은 다음처럼 명시적으로 진단용임을 지정해야 합니다.

```powershell
py -3.14 -m localai train --resume models/checkpoints/localai_v0.2.pt --allow-legacy-resume --steps 10
```

과거 모델은 새 검증/테스트 문장을 이미 보았을 수 있습니다.
이 경우 `heldout_contaminated=true`가 이후 실험에도 유지되며 평가에도 경고가 나옵니다.
원래 체크포인트는 덮어쓰지 않습니다.
사용자 지정 기존 체크포인트에는 `--resume-vocab`을 지정하세요.
기존 형식에는 어휘 매핑이 내장돼 있지 않아 같은 크기의 잘못된 매핑을 판별할 수 없습니다.
새 체크포인트는 매핑을 내장하므로 동일 크기라도 다른 매핑을 거부합니다.

## 별도 평가

```powershell
py -3.14 -m localai evaluate --checkpoint logs/experiments/<실험ID>/best.pt --split val
py -3.14 -m localai evaluate --checkpoint logs/experiments/<실험ID>/best.pt --split test
```

학습 당시 저장된 분할과 평가 설정으로 CPU에서 평가합니다.
모델 선택은 검증 손실만 사용하고, 테스트는 선택을 완료한 모델의 최종 보고에 사용하세요.
테스트 결과를 반복 확인하며 모델을 고르면 독립 테스트의 의미가 약해집니다.
이 명령은 별도 평가 결과를 콘솔에 출력하며 기존 체크포인트를 수정하지 않습니다.

문장 단위 분리는 동일 문서의 주제 공유나 의미상 중복까지 제거하지 못합니다.
유사도 그룹 비교는 O(n²)이므로 큰 데이터셋에는 문서·출처 단위 관리가 필요합니다.
원문이 추가되면 분할도 바뀌므로 새 데이터 버전의 별도 실험으로 취급하세요.
기존 모델을 새 데이터로 이어 학습하는 기능은 현재 고정 분할 정책에서 지원하지 않습니다.
