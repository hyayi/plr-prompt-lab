# PLR Prompt Lab — 데이터셋 디렉터리 명세

`lab validate-dataset` / `lab run` / `lab submit`이 요구하는 데이터셋
디렉터리 구조와 파일 스키마의 정의 문서다.

기존 `eval/golden/<attribute>/` 디렉터리들 자체가 유효한 데이터셋이다 —
이 명세가 그 레이아웃에서 유도됐다.

---

## 1. 디렉터리 레이아웃

```
<dataset>/
    crops/
        <obj_id>.jpg          # 객체당 크롭 이미지 1장 (JPEG)
    labels.jsonl              # 사람 정답 라벨 (eval에 필수)
    predictions.jsonl         # 모델 출력 (`lab run`이 씀; obj_id 집합의 씨앗)
    attributes.jsonl          # 크롭당 PLR JSON 전체 (`lab run`이 씀)
    manifest.yaml             # 데이터셋 메타 (필수)
```

`lab validate-dataset` 통과에는 `crops/` + `labels.jsonl` + `manifest.yaml`
세 개만 있으면 된다. 복사해서 쓰는 뼈대는 `examples/dataset_template/`
(그대로 validate PASS).

**구조는 고정, 라벨 집합은 작성자의 것.** 필수 manifest 필드 외에,
데이터셋은 자기 속성을 직접 선언할 수 있다 (generic 데이터셋):

```yaml
attribute: helmet                        # 아무 이름 — 프리셋: gender | vehicle_type | military
labels: [helmet, no_helmet]              # 허용 라벨 값 (validate가 강제)
pred_path: attributes.equipment[0].type  # PLR JSON에서 예측값 위치 (dots + [idx])
margin_path: ...                         # 옵션 — 모델 확신도 경로
bias_pair: [no_helmet, helmet]           # 옵션 — 헤드라인 bias [정답, 오인]
object_type_hint: person                 # 옵션 — person | vehicle (기본 person)
```

PLR 3속성(gender / vehicle_type / military)은 내장 프리셋 — 선언 없이
동작하며 이 체계의 참고 예시를 겸한다 (`evalkit/dataset.py`의
`PRESET_SPECS`).

### 생성 절차

1. crops — `crops/<obj_id>.jpg` (운영 비디오는 `lab build-golden`,
   또는 임의 수집 크롭을 직접 투입)
2. manifest — `examples/dataset_template/manifest.yaml` 복사 후 수정
3. labels — `labels.jsonl` 직접 작성 또는 `lab label --dataset D ...`
   (사람도 판별 불가한 크롭 → `unknown`: 채점 제외, 별도 집계)
4. `lab validate-dataset --dataset D` → 5. `lab run` → 6. `lab submit --pull`
   (서버 채점 → pulled/gallery.html · pulled/report.html 반환 — 눈 검수는 gallery)

`predictions.jsonl`과 `attributes.jsonl`은 `lab run`이 쓰고, `lab submit` 시
서버가 읽어 채점한다. (`queries.jsonl` 검색 데이터셋 종류는 2026-07 제거 — lab은
PLR 전용.)

---

## 2. obj_id 규칙

- 크롭 파일명의 **stem**이 `obj_id`다:
  `crops/abc123.jpg` → `obj_id = "abc123"`.
- `labels.jsonl` 모든 행의 `obj_id`는 크롭 stem과 정확히 일치해야 한다
  (대소문자 구분, 확장자 없음).
- `labels.jsonl`과 `predictions.jsonl`은 같은 `obj_id` 네임스페이스를 쓴다.
- 라벨된 obj_id와 크롭 파일은 1:1 — 라벨 없는 크롭 파일은 **warning**,
  크롭 없는 라벨은 **error**.

---

## 3. `manifest.yaml` 스키마

```yaml
attribute: gender          # 필수 — 이 데이터셋이 다루는 PLR 속성
n: 150                     # 필수 — 라벨된 객체 수 (기대값)
created: "2026-07-01"      # 필수 — 생성일 (ISO)
source_note: "video vd_001_0032, frames 0–3600"  # 필수 — 출처 메모
```

**필수 필드**: `attribute`(또는 아래의 `attributes`), `n`, `created`,
`source_note`. — manifest 는 **데이터셋만** 기술한다.

> **모델/프롬프트 버전은 manifest 에 넣지 않는다.** 데이터셋은 모델과 무관해야
> 하고(같은 데이터셋을 여러 모델·버전으로 채점), 실제로 어떤 모델·프롬프트로
> 실행했는지는 **run 시점**에 기록된다: `lab run -X <version> --model <model>` →
> `run_provenance.json`(model·version·surface_hash), `lab submit` 의
> `version_label`. 서버 리더보드/리포트의 model·version 열은 이 run 레코드에서
> 온다(manifest 가 아님). 과거 `model:`/`prompt:` 옵션 필드는 아무도 읽지 않는
> 잔재였고 2026-07 제거됐다.

`attribute` 값이 `validate-dataset`의 라벨 어휘 검사 기준이 된다 (§5).

### 다속성 manifest (`attributes:` 맵 — 권장)

같은 크롭셋에 여러 속성의 정답을 함께 라벨할 때는 단일 `attribute:` 대신
속성별 스펙 맵을 선언한다. 모델 호출은 크롭당 1회뿐이고(`attributes.jsonl`에
전체 plr_json 저장), `lab submit` 시 서버가 라벨이 실제로 달린 속성 전부를
한 번에 채점한다 (선언만 되고 라벨이 없는 속성은 skip+안내) — 속성별 예측은
`attributes.jsonl`에서 `pred_path`로 재추출되므로 GPU 재실행이 없다.

```yaml
n: 150
created: "2026-07-03"
source_note: "..."
attributes:
  gender: {}                      # 내장 프리셋 — 빈 dict면 프리셋 상속
  helmet:                         # 커스텀 속성 — labels + pred_path 선언
    labels: [helmet, no_helmet]
    pred_path: attributes.equipment[0].type
    # margin_path / bias_pair / object_type_hint — §3의 옵션 키와 동일
```

---

## 4. `labels.jsonl` 스키마

한 줄에 JSON 객체 하나 (UTF-8, 후행 콤마 없음). 빈 줄은 무시.

```json
{"obj_id": "1003", "label": "female"}
{"obj_id": "1013", "label": "male", "notes": "애매 — 경계 사례"}
```

**행별 필수 필드**:

| 필드 | 타입 | 설명 |
|----------|--------|--------------------------------------------------|
| `obj_id` | string | 객체 식별자 — 크롭 stem과 일치해야 함 |
| `label`  | string | 사람 정답 값 (§5 어휘 참고) |

**행별 옵션 필드**: `notes`(자유 텍스트), 기타 추가 필드.

### 다속성 행 (`"labels"` dict)

다속성 데이터셋(§3의 `attributes:` 맵)의 행은 `label` 대신 속성별 dict를 쓴다.
두 형식은 한 파일에 혼재해도 된다 (legacy 단일 `label` 행은 어느 속성 평가에도
그 값이 쓰인다):

```json
{"obj_id": "1003", "labels": {"gender": "female", "helmet": "no_helmet"}}
{"obj_id": "1013", "labels": {"gender": "unknown", "helmet": "helmet"}}
{"obj_id": "1021", "labels": {"helmet": "helmet"}}
```

- 어떤 속성 키가 **없는** 행은 그 속성 평가에서 **자연 제외**된다(미라벨 —
  `unknown`과 다름: unknown은 "사람도 판별 불가"로 별도 집계, 미라벨은 조인
  자체에서 빠짐). 1021처럼 속성별로 라벨 가능한 것만 채우면 된다.
- manifest `attributes:`에 선언되지 않은 속성 키는 `validate-dataset`이
  **error**로 잡는다 (오타가 조용히 평가에서 빠지는 사고 방지).

### 사람·차량 혼합 데이터셋 (행별 `"object_type"`)

사람과 차량 크롭이 한 데이터셋에 섞여 있으면, 행마다 `object_type`을 적어
**크롭별로 어느 프롬프트(person/vehicle)를 쓸지** 지정한다 — 운영에서
트래커 클래스가 객체마다 힌트를 주는 것의 lab 대응물이다:

```json
{"obj_id": "p1", "object_type": "person",  "labels": {"gender": "female"}}
{"obj_id": "v1", "object_type": "vehicle", "labels": {"vehicle_type": "sedan"}}
```

- 허용값은 `person` | `vehicle` (그 외는 validation error).
- 필드가 없는 행은 manifest/프리셋의 `object_type_hint` 폴백(데이터셋 단위) —
  단일 종 데이터셋은 아무것도 바꿀 필요 없다.
- 속성 라벨은 해당 종의 크롭에만 달면 된다: 위 예에서 v1에 gender 키가
  없으므로 gender 평가에서 자연 제외된다.

**`label: unknown` 정책 (강제커밋, plr_v1.5_cot)**: 크롭에서 사람도 그
속성을 판별할 수 없을 때(가림/극단적 저품질)**만** `unknown`으로 라벨한다.
그런 크롭은 서버 채점의 **accuracy/recall/bias/confusion에서 제외**된다 —
강제커밋 프롬프트에서 모델은 어쨌든 답해야 하고, 그 답을 채점할 정답이
없기 때문이다. `n_label_unknown`으로 별도 보고되고, 모델 자신의 unknown
응답률은 `pred_unknown`으로 서버 metrics.json에 추적된다.
지정은 `lab label --unknown <타일 id들>`.

---

## 5. 라벨 어휘

`label`의 허용값은 manifest가 선언한 속성에 따른다. 어휘 밖 라벨은
**validation error**(경고 아님) — 조용히 채점을 오염시키기 때문이다.

### `attribute: gender`

| 값 | 의미 |
|-----------|--------------------------------------------------|
| `male`    | 남성으로 보이는 사람 |
| `female`  | 여성으로 보이는 사람 |
| `unknown` | 크롭만으로 판별 불가 (가림, 화질) |

### `attribute: vehicle_type`

허용값은 `plr_schema.VEHICLE_TYPE_ENUM`의 `type_topk` 라벨:

```
sedan, suv, hatchback, light_car, van, minivan,
pickup_truck, truck, bus, taxi,
ambulance, police_car, fire_truck, emergency_vehicle,
motorcycle, scooter, bicycle, kickboard,
construction_vehicle, vehicle_unknown
```

### `attribute: military`

| 값 | 의미 |
|------------|------------------------------------------|
| `military` | 군용 인원/차량 |
| `civilian` | 비군용 |
| `unknown`  | 크롭만으로 판별 불가 |

커스텀 속성은 manifest의 `labels:` 선언이 어휘가 된다 (다속성이면
`attributes.<이름>.labels`).

---

## 6. `predictions.jsonl` 스키마 (`lab run`이 씀)

```json
{"obj_id": "1003", "attribute": "gender", "pred": "male", "reason": "broad shoulders", "margin": 0.8, "quality": 0.71}
```

- `attribute` — 이 행이 어느 속성의 추출물인지 스탬프. 서버 채점 시 다른
  속성을 평가할 때 `attributes.jsonl` 재추출로 전환하는 근거.
- `margin` — 평가 속성에 대한 모델의 결정 확신도 (프롬프트의 `margins`
  블록에서; margin을 emit하지 않는 속성은 `null`). 강제커밋 체제에서
  제거된 `unknown` 도피처의 대체물.
- `quality` — 크롭 품질 점수 [0,1] (quality_gate — **측정 전용**, 모델
  호출을 게이트하지 않는다).

서버 채점은 두 신호로 accuracy를 분할해(`margin_stats`/`quality_stats`)
오답이 저신호 구간에 몰리는지 검증한다. 두 필드는 옵션 — 없는 구파일도
평가된다.

이 파일이 `re_score`가 처리할 obj_id 집합의 씨앗이다. `lab run`이
덮어쓰고 `lab submit` 시 서버가 읽는다.

---

## 7. `attributes.jsonl` 스키마 (`lab run`이 씀)

```json
{"obj_id": "1003", "plr_json": { ...PLR JSON 전체... }}
```

크롭당 PLR 출력 전체 — 평가 속성 하나를 넘어 슬롯별 분석(unknown율,
margin 분포)과 **다속성 평가의 재추출 원천**이다. `plr_json` 값은
`plr_schema.PERSON_SCHEMA` 또는 `VEHICLE_SCHEMA`를 만족해야 한다.

---

## 8. 최소 완성 예시

```
my_gender_dataset/
    manifest.yaml
    labels.jsonl
    crops/
        obj_001.jpg
        obj_002.jpg
        obj_003.jpg
```

**`manifest.yaml`**:
```yaml
attribute: gender
n: 3
created: "2026-07-01"
source_note: "합성 예시 — CCTV 사람 크롭 3장"
```

**`labels.jsonl`**:
```json
{"obj_id": "obj_001", "label": "female"}
{"obj_id": "obj_002", "label": "male"}
{"obj_id": "obj_003", "label": "unknown", "notes": "심한 가림"}
```

**`crops/`**: `obj_001.jpg`, `obj_002.jpg`, `obj_003.jpg` 세 JPEG.

검증:
```bash
python3 lab.py validate-dataset --dataset my_gender_dataset/
```

---

## 9. `validate-dataset` 검사 목록과 종료 코드 계약

`lab validate-dataset --dataset <path>`는 아래 검사를 순서대로 수행하고
각각 `PASS` / `WARN` / `FAIL` 줄을 낸다.

| # | 검사 | 실패 시 수준 |
|---|-------|-----------------|
| 1 | `manifest.yaml` 존재 + YAML 파싱 가능 | **error** |
| 2 | manifest 필수 필드(`attribute` 또는 `attributes`, `n`, `created`, `source_note`) | **error** |
| 3 | `labels.jsonl` 존재 | **error** |
| 4 | 모든 행이 유효한 JSON + `obj_id` + (`label` 또는 `labels` dict) | **error** |
| 5 | 모든 라벨 값이 해당 속성 어휘 안 (다속성은 속성별 어휘) | **error** |
| 6 | `labels` dict의 속성 키가 manifest `attributes:`에 선언됨 | **error** |
| 7 | 행별 `object_type`이 있으면 `person`/`vehicle` | **error** |
| 8 | `crops/` 디렉터리 존재 | **error** |
| 9 | 라벨된 모든 `obj_id`에 대응 `<obj_id>.jpg` 존재 | **error** |
| 10 | 라벨 없는 크롭 파일 | **warning** |

**종료 코드 계약**:
- **0**: 오류 없음 (경고는 허용).
- **0 아님**: 오류 1개 이상.

출력은 항상 한 줄 요약으로 끝난다:
```
Summary: N crops, N labels, N error(s), N warning(s)
Result: PASS   # 또는 FAIL
```

프로그래밍 API `validate_dataset(path) -> bool`은 통과 시 `True`,
오류 시 `False`를 반환하며 같은 줄들을 stdout에 찍는다.
