# 데이터셋 만들기 가이드 (실습)

PLR 데이터셋을 **처음부터 직접 만드는 법**을 따라치기로 정리한 문서입니다.
디렉터리 구성 → `manifest.yaml` 작성 → `labels.jsonl` 작성 → 검증까지 다룹니다.

- 이 문서 = **만드는 법(how-to)**. 필드·어휘의 **완전한 명세**는 [DATASET_SPEC.md](DATASET_SPEC.md).
- 데이터셋을 만든 다음, 서버에 제출해 채점받는 흐름은 [TUTORIAL.md](TUTORIAL.md).
- GPU 없이 **바로 실습만** 할 거면 이 문서를 건너뛰고 `python3 lab.py demo --keep`
  (합성 데이터셋 `datasets/demo/` 생성) 후 [TUTORIAL.md](TUTORIAL.md)로 가도 됩니다.

---

## 1. 데이터셋이란 — 디렉터리 3종

```
my_dataset/
    crops/            # 객체당 크롭 이미지 1장 (<obj_id>.jpg)
    labels.jsonl      # 사람 정답 라벨
    manifest.yaml     # 데이터셋 메타 + 채점할 속성 선언
```

`lab validate-dataset` 통과에는 이 셋만 있으면 됩니다. `predictions.jsonl`·
`attributes.jsonl`은 나중에 `lab run`이 만듭니다(직접 만들 필요 없음).

**obj_id 규칙**: 크롭 파일명 stem이 곧 `obj_id`입니다.
- `crops/p001.jpg` → `obj_id = "p001"`
- `labels.jsonl`의 모든 `obj_id`는 크롭 stem과 **정확히** 일치해야 합니다(대소문자·확장자 없음).

> 복사해서 시작할 뼈대: `examples/dataset_template/` (그대로 validate PASS).

---

## 2. crops 준비

- 객체 하나당 JPEG 한 장, `crops/<obj_id>.jpg`.
- 파일명(obj_id)은 자유(영숫자 권장). 라벨과 1:1로 맞추기만 하면 됩니다.
- 출처: 운영 비디오는 `lab build-golden`(운영자 단계), 또는 임의 수집 크롭을 직접 투입.

```
crops/
    p001.jpg
    p002.jpg
    p003.jpg
```

---

## 3. `manifest.yaml` 작성 — 데이터 구조

### 3-1. 단일 속성 (가장 단순)

한 데이터셋에서 속성 하나만 채점할 때:

```yaml
attribute: gender                       # 채점할 속성 (프리셋: gender | vehicle_type | military)
n: 3                                     # 라벨된 객체 수(기대값)
created: "2026-07-04"                    # 생성일 (ISO)
source_note: "합성 예시 — 사람 크롭 3장"   # 출처 메모
```

**필수 필드**: `attribute`(또는 아래 `attributes:`), `n`, `created`, `source_note`.

### 3-2. 다속성 (권장 — 한 크롭셋에 여러 속성 정답)

같은 크롭에 여러 속성을 함께 라벨하면, 모델은 **크롭당 1회만** 실행하고
서버가 속성별로 재추출해 전부 채점합니다(속성 늘려도 GPU 비용 그대로):

```yaml
n: 3
created: "2026-07-04"
source_note: "..."
attributes:
  gender: {}                            # 내장 프리셋은 빈 dict면 프리셋 상속
  helmet:                               # 커스텀 속성 = labels + pred_path 선언
    labels: [helmet, no_helmet]         # 허용 라벨 값(= 어휘)
    pred_path: attributes.equipment[0].type   # plr_json에서 예측값 위치
    # (옵션) bias_pair: [no_helmet, helmet]    # 헤드라인 오분류 [정답, 오인]
    # (옵션) object_type_hint: person          # person | vehicle (기본 person)
```

- **`pred_path`** = 모델 출력(plr_json)에서 이 속성의 예측을 꺼내는 경로.
  점(`.`)으로 중첩을, `[n]`으로 배열 인덱스를 나타냅니다
  (예: `attributes.gender_scores.selected`, `attributes.equipment[0].type`).
- **프리셋**(gender/vehicle_type/military)은 `pred_path`·`labels`가 내장이라 빈 dict `{}`로 충분.
- 전체 옵션 키·프리셋 정의: [DATASET_SPEC.md §3](DATASET_SPEC.md).

---

## 4. `labels.jsonl` 작성 — 라벨 만드는 법

한 줄 = JSON 객체 하나(UTF-8, 후행 콤마 없음, 빈 줄 무시).

### 4-1. 형식

**단일 속성**은 `label`:

```json
{"obj_id": "p001", "label": "female"}
{"obj_id": "p002", "label": "male"}
{"obj_id": "p003", "label": "unknown", "notes": "심한 가림"}
```

**다속성**은 `labels` dict (두 형식 한 파일에 혼재 가능):

```json
{"obj_id": "p001", "labels": {"gender": "female", "helmet": "no_helmet"}}
{"obj_id": "p002", "labels": {"gender": "unknown", "helmet": "helmet"}}
{"obj_id": "p003", "labels": {"helmet": "helmet"}}
```

**사람/차량 혼합**은 행마다 `object_type`(크롭별 person/vehicle 프롬프트 라우팅):

```json
{"obj_id": "p1", "object_type": "person",  "labels": {"gender": "female"}}
{"obj_id": "v1", "object_type": "vehicle", "labels": {"vehicle_type": "sedan"}}
```

### 4-2. 라벨 규칙

- 속성 키가 **없는** 행 = 그 속성 평가에서 **자연 제외**(미라벨).
- **`unknown`** = 사람도 판별 불가(가림/저화질)일 때만. 채점에서 제외되고 별도
  집계(`n_label_unknown`) — 미라벨과 다릅니다.
- 라벨 값은 해당 속성의 **어휘** 안이어야 합니다(어휘 밖은 검증 **에러**):
  - `gender`: `male` / `female` / `unknown`
  - `military`: `military` / `civilian` / `unknown`
  - `vehicle_type`: `sedan`/`suv`/`truck`/… ([DATASET_SPEC.md §5](DATASET_SPEC.md))
  - 커스텀 속성: manifest의 `labels:` 선언이 곧 어휘

### 4-3. 방법 1 — 직접 작성

에디터로 위 형식대로 `labels.jsonl`을 씁니다. 소규모/명확할 때 가장 빠릅니다.

### 4-4. 방법 2 — `lab label` (모델 예측을 시드로 대량 정정)

`lab run`으로 `predictions.jsonl`을 먼저 만든 뒤, 모델 예측을 정답 시드로 깔고
**틀린 것만 정정**합니다:

```bash
# 예: 모델이 male로 본 것 중 M3,M7은 실제 female, M9는 판별 불가(unknown)
python3 lab.py label --dataset datasets/mine --female-in-male M3,M7 --unknown M9
# → datasets/mine/labels.jsonl 생성/갱신
```

> `--dataset` 뒤의 정정 플래그는 `eval/make_labels.py`로 그대로 전달됩니다.
> 지원 플래그는 `eval/make_labels.py` 소스를 참고하세요.

---

## 5. 검증 — `validate-dataset`

만든 데이터셋이 형식에 맞는지 **제출 전에** 확인합니다:

```bash
python3 lab.py validate-dataset --dataset datasets/mine
```

검사(요약): manifest 존재·필수필드 → labels.jsonl 유효 JSON·obj_id·라벨 →
라벨 값이 어휘 안 → 다속성 키가 manifest에 선언됨 → object_type person/vehicle →
crops/ 존재 → 라벨된 obj_id마다 크롭 존재. 라벨 없는 크롭은 warning.
(전체 목록·종료코드: [DATASET_SPEC.md §9](DATASET_SPEC.md).)

```
Summary: 3 crops, 3 labels, 0 error(s), 0 warning(s)
Result: PASS
```

> 라벨 **어휘(enum) 검증은 클라이언트(여기)** 담당입니다. 평가 서버는 이 검증을
> 신뢰하고, 등록(push) 때는 구조(파일 존재/파싱)만 확인합니다 — 즉
> **제출 전 `validate-dataset` 통과가 곧 라벨 품질 보증**입니다.

---

## 6. 최소 완성 예시 (처음부터)

```bash
mkdir -p datasets/mine/crops
# crops/p001.jpg, p002.jpg, p003.jpg 를 넣습니다 (임의의 사람 크롭 JPEG)

cat > datasets/mine/manifest.yaml <<'YAML'
attribute: gender
n: 3
created: "2026-07-04"
source_note: "합성 예시 — 사람 크롭 3장"
YAML

cat > datasets/mine/labels.jsonl <<'JSONL'
{"obj_id": "p001", "label": "female"}
{"obj_id": "p002", "label": "male"}
{"obj_id": "p003", "label": "unknown", "notes": "심한 가림"}
JSONL

python3 lab.py validate-dataset --dataset datasets/mine
# Result: PASS
```

---

## 7. 다음 단계

데이터셋이 `PASS`하면 서버에 제출해 채점받습니다 — [TUTORIAL.md](TUTORIAL.md)의
**Part C(CLI)** 또는 **Part D(웹 업로드)** 로:

```bash
python3 lab.py run -X plr_v1.5_cot --dataset datasets/mine --model mock
python3 lab.py dataset-push --dataset datasets/mine --name mine
python3 lab.py submit --dataset mine --run-dir datasets/mine -X plr_v1.5_cot --pull
```

> `datasets/`·크롭은 사적 CCTV 데이터 — **절대 커밋 금지**(gitignore 처리됨).
