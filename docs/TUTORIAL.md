# 튜토리얼 — lab ↔ 평가 서버 (따라치기, 전체판)

이 문서는 **평가 서버(`plr-eval-server`)** 와 **lab 클라이언트(`plr-prompt-lab`)** 를
연동해 채점·리포트·갤러리·리더보드를 받아보는 전 과정을 복붙으로 따라 할 수 있게
정리한 것입니다. 데이터셋을 **직접 만드는 법**(manifest 구조·라벨 작성·검증)과,
CLI 경로 / 웹 UI 수동 업로드 **두 가지**를 모두 다룹니다.

전부 **GPU 없이**(mock 모델) 동작합니다. 실측(진짜 Gemma) 전환은 §E-3.

- 레포 위치(이 문서 기준): lab = `~/plr-prompt-lab`, 서버 = `~/plr-eval-server`
- 터미널 2개: **A = 서버**, **B = lab 클라이언트**
- 데이터셋 형식의 **완전한 스키마**는 [DATASET_SPEC.md](DATASET_SPEC.md) — 이 튜토리얼은
  실습 중심이고, 세부 규칙은 그 문서를 참조합니다.

```
[터미널 B: lab]                          [터미널 A: 서버]
 데이터셋 만들기 → validate-dataset
 lab run --model mock  ─ attributes.jsonl + run_provenance.json
 lab dataset-push ───────────────────▶  데이터셋 등록
 lab submit --pull ──────────────────▶  채점 → metrics/report/gallery 렌더
        ◀───────────  pulled/ 로 회수  ─┘
 브라우저 ───────────────────────────▶  리더보드 /d/<dataset>
```

---

# Part 0. 사전 준비

```bash
python3 --version    # 3.10+

# 서버
cd ~/plr-eval-server && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# lab (별도 터미널/venv)
cd ~/plr-prompt-lab && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 두 레포는 **독립**입니다. 공유하는 건 `evalkit/dataset.py`·`evalkit/provenance.py`
> 두 파일의 byte-identical 복본뿐(`contract/CONTRACT.md`).

---

# Part A. 서버 띄우기 (터미널 A)

```bash
cd ~/plr-eval-server && source .venv/bin/activate

export EVAL_SERVER_DATA=~/eval_server_data    # 데이터셋·run 파일 저장 볼륨
export EVAL_SERVER_TOKEN=tutorial-token       # 변이 API(X-Auth-Token) 값 — 아무 문자열

uvicorn server.app:app --host 0.0.0.0 --port 8890 --workers 1
```

> `--workers 1` **필수**(쓰기 잠금이 단일 프로세스). 토큰을 안 걸면 인증 없이 열립니다(로컬 전용).

확인:

```bash
curl -s http://127.0.0.1:8890/health          # {"ok":true,...}
# 브라우저: http://127.0.0.1:8890/
```

---

# Part B. 데이터셋 만들기

## B-1. 데이터셋이란 — 디렉터리 3종

`lab validate-dataset` 통과에 필요한 최소 구성:

```
my_dataset/
    crops/            # 객체당 크롭 이미지 1장 (<obj_id>.jpg)
    labels.jsonl      # 사람 정답 라벨
    manifest.yaml     # 데이터셋 메타 + 채점할 속성 선언
```

- **`obj_id` = 크롭 파일명 stem**: `crops/p001.jpg` → `obj_id = "p001"`.
- `labels.jsonl`의 모든 `obj_id`는 크롭 stem과 정확히 일치해야 함.

> GPU 없이 바로 실습만 할 거면 이 B 파트를 건너뛰고
> `python3 lab.py demo --keep` 로 합성 데이터셋(`datasets/demo/`)을 만들어도 됩니다.

## B-2. `manifest.yaml` — 데이터 구조

**단일 속성** (가장 단순):

```yaml
attribute: gender                       # 이 데이터셋이 채점할 속성
n: 3                                     # 라벨된 객체 수(기대값)
created: "2026-07-04"                    # 생성일
source_note: "합성 예시 — 사람 크롭 3장"   # 출처 메모
```

**다속성** (한 크롭셋에 여러 속성 정답을 함께 — 권장):

```yaml
n: 3
created: "2026-07-04"
source_note: "..."
attributes:
  gender: {}                            # 내장 프리셋(gender/vehicle_type/military)은 빈 dict
  helmet:                               # 커스텀 속성 = labels + pred_path 선언
    labels: [helmet, no_helmet]         # 허용 라벨 값(어휘)
    pred_path: attributes.equipment[0].type   # plr_json에서 예측값 위치(dots + [idx])
    # (옵션) bias_pair: [no_helmet, helmet]   # 헤드라인 오분류 [정답, 오인]
    # (옵션) object_type_hint: person
```

- 필수 필드: `attribute`(또는 `attributes:`), `n`, `created`, `source_note`.
- `pred_path`는 모델 출력(plr_json)에서 그 속성 예측을 꺼내는 경로 — 그래서 속성을
  늘려도 **모델 재실행 없이** 서버가 재추출해 채점합니다.
- 전체 필드·프리셋 목록: [DATASET_SPEC.md §3](DATASET_SPEC.md).

## B-3. `labels.jsonl` — 라벨 만드는 법

한 줄 = JSON 객체 하나(UTF-8). **단일 속성**은 `label`:

```json
{"obj_id": "p001", "label": "female"}
{"obj_id": "p002", "label": "male"}
{"obj_id": "p003", "label": "unknown", "notes": "심한 가림"}
```

**다속성**은 `labels` dict (두 형식 혼재 가능):

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

라벨 규칙:
- 속성 키가 **없는** 행 = 그 속성 평가에서 **자연 제외**(미라벨).
- **`unknown`** = 사람도 판별 불가(가림/저화질)일 때만. 채점에서 제외되고 별도 집계
  (`n_label_unknown`). 미라벨과 다름.
- 라벨 값은 해당 속성 **어휘** 안이어야 함(gender: male/female/unknown 등 — [DATASET_SPEC.md §5](DATASET_SPEC.md)).

### 방법 1 — 직접 작성
텍스트 에디터로 위 형식대로 `labels.jsonl`을 씁니다(소규모/명확할 때 가장 빠름).

### 방법 2 — `lab label` (모델 예측을 시드로 대량 정정)
`lab run`으로 `predictions.jsonl`을 먼저 만든 뒤, 모델 예측을 정답 시드로 깔고
**틀린 것만 정정**하는 방식:

```bash
# 예: 모델이 male로 본 것 중 M3,M7은 실제 female, M9는 사람도 판별 불가(unknown)
python3 lab.py label --dataset datasets/mine --female-in-male M3,M7 --unknown M9
# → datasets/mine/labels.jsonl 생성/갱신
```

> `lab label`은 `eval/make_labels.py`를 감싼 것으로, `--dataset`을 주면
> `predictions.jsonl`을 읽어 `labels.jsonl`로 씁니다. `--dataset` 뒤의 정정 플래그
> (`--female-in-male`, `--unknown` 등)는 make_labels.py로 그대로 전달됩니다. 지원
> 플래그는 `eval/make_labels.py` 소스를 참고하세요. 소규모/명확한 경우엔 방법 1(직접 작성)이 간단합니다.

## B-4. 검증 — `validate-dataset`

만든 데이터셋이 형식에 맞는지 **제출 전에** 확인합니다:

```bash
python3 lab.py validate-dataset --dataset datasets/mine
```

검사 항목(요약): manifest 존재·필수필드 → labels.jsonl 유효 JSON·obj_id·라벨 →
라벨 값이 어휘 안 → 다속성 키가 manifest에 선언됨 → object_type person/vehicle →
crops/ 존재 → 라벨된 obj_id마다 크롭 존재. 라벨 없는 크롭은 warning.
(전체 목록·종료코드: [DATASET_SPEC.md §9](DATASET_SPEC.md).)

```
Summary: 3 crops, 3 labels, 0 error(s), 0 warning(s)
Result: PASS
```

> 라벨 **어휘(enum) 검증은 클라이언트(여기)** 담당입니다. 서버는 이걸 신뢰하고,
> push 때는 구조(파일 존재/파싱)만 확인합니다. → **제출 전 `validate-dataset` 통과가 곧 품질 보증**.

---

# Part C. CLI로 실행·제출 (터미널 B) — 권장 경로

```bash
cd ~/plr-prompt-lab && source .venv/bin/activate
export EVAL_SERVER_URL=http://127.0.0.1:8890
export EVAL_SERVER_TOKEN=tutorial-token       # 서버와 같은 값!
```

## C-1. 모델 실행 (GPU-free)

```bash
python3 lab.py run -X plr_v1.5_cot --dataset datasets/mine --model mock
```

생성물(모두 `datasets/mine/`):
- `attributes.jsonl` — 크롭당 plr_json 전체(서버가 채점에 씀)
- `predictions.jsonl` — 속성 추출 뷰
- `run_provenance.json` — 실행 지문(surface_hash·model·version…, 서버 무결성 대조용)

> `-X` = 프롬프트 버전 라벨(리더보드 구분키). `--model mock`이 GPU-free. 실측은 `--model gemma`(§E-3).

## C-2. 등록 → 제출 → 회수

```bash
python3 lab.py dataset-push --dataset datasets/mine --name mine
python3 lab.py submit --dataset mine --run-dir datasets/mine -X plr_v1.5_cot --pull
```

출력 예:
```
[submit] run r20260704-124852-80a96b
[submit] aggregate: macro_f1=1.0 macro_acc=1.0 micro_acc=1.0
[submit]   gender: acc=1.0 macro_f1=1.0 n=5
[submit] pulled metrics.json, report.html, gallery.html -> datasets/mine/pulled
```

- `--dataset mine` = 서버 측 이름(dataset-push의 `--name`), `--run-dir` = attributes.jsonl 있는 로컬 dir
- `--pull` = 서버 렌더 결과를 `<run-dir>/pulled/`로 회수(`--out <경로>`로 변경 가능)
- `⚠ hash unverified` 배지 = `run_provenance.json` 없이 제출됨(채점은 정상). C-1을 거치면 안 뜸.

---

# Part D. 웹 UI로 수동 업로드 (CLI 없이)

브라우저 <http://127.0.0.1:8890/upload> 에 두 폼이 있습니다. CLI가 자동으로 만들어
주던 파일들을 **직접 만들어** 올리는 방식입니다.

## D-1. 각 파일이 무엇이고 어떻게 만드나

**① 데이터셋 등록 폼 — `tar.gz`** (crops/ + labels.jsonl + manifest.yaml 한 덩어리)

```bash
# 데이터셋 디렉터리를 통째로 압축 (top-level 또는 단일 하위 dir에 manifest.yaml)
tar czf mine.tgz -C datasets mine
```

**② run 제출 폼 — 3개 파일**

| 폼 필드 | 정체 | 만드는 법 |
|---|---|---|
| `attributes.jsonl` | 모델이 낸 plr_json 전체(채점 원천) | `lab run`이 `datasets/mine/`에 생성 |
| `surface.tar.gz` | 프롬프트 표면 번들(파서·코어·스키마·prompts·vocab·configs) — 서버가 해시 대조용으로만 보관, **실행 안 함** | 아래 한 줄로 생성 ↓ |
| `run_provenance.json` (선택) | 실행 지문(surface_hash 등). 있으면 `hash verified` 배지 | `lab run`이 `datasets/mine/`에 생성 |

`surface.tar.gz` 만들기 (lab 헬퍼가 정확한 파일 집합을 묶음 — 손으로 tar 뜨지 마세요):

```bash
python3 -c "from pathlib import Path; from runners.client import build_surface_bundle; \
Path('surface.tgz').write_bytes(build_surface_bundle(Path('.')))"
```

## D-2. 폼에 올리기

- 데이터셋 등록 폼: **토큰**=`tutorial-token`, **이름**=`mine`, **tar.gz**=`mine.tgz` → 등록
- run 제출 폼: **토큰**·**데이터셋 이름**(`mine`)·**version_label**(`plr_v1.5_cot`) 입력 →
  `attributes.jsonl`, `surface.tgz`, `run_provenance.json`(선택) 선택 → 제출

## D-3. (참고) 웹 폼 = 이 curl과 동일

폼 제출은 아래 multipart와 정확히 같습니다(자동화/디버깅용):

```bash
# 데이터셋 등록
curl -X POST http://127.0.0.1:8890/api/datasets \
  -H "X-Auth-Token: tutorial-token" \
  -F "name=mine" -F "created_by=me" \
  -F "archive=@mine.tgz;type=application/gzip"
# → {"name":"mine","n_crops":5}

# run 제출
curl -X POST http://127.0.0.1:8890/api/runs \
  -H "X-Auth-Token: tutorial-token" \
  -F "dataset=mine" -F "version_label=plr_v1.5_cot" -F "submitted_by=me" \
  -F "attributes=@datasets/mine/attributes.jsonl;type=application/json" \
  -F "surface=@surface.tgz;type=application/gzip" \
  -F "provenance=@datasets/mine/run_provenance.json;type=application/json"
# → {"run_id":"...","hash_verified":true,"aggregate":{"macro_f1":1.0,...},...}
```

> 웹 폼에 "CLI 권장" 안내가 붙은 이유: `surface.tar.gz`를 손으로 정확히 묶기 번거롭기
> 때문입니다. `lab submit`은 이 번들을 메모리에서 자동 생성해 한 번에 올립니다.

---

# Part E. 결과·개선·실측·문제해결

## E-1. 결과 보기

```bash
ls datasets/mine/pulled/          # gallery.html  metrics.json  report.html
xdg-open datasets/mine/pulled/gallery.html   # 오답 우선·속성 태그·AND/OR 필터
xdg-open datasets/mine/pulled/report.html    # 버전 비교표
```

- 서버 리더보드: <http://127.0.0.1:8890/d/mine>
- run 상세: <http://127.0.0.1:8890/r/&lt;run_id&gt;>
- `metrics.json` 필드: accuracy/recall/precision/f1/macro_f1/bias/confusion/
  pred_unknown/n_label_unknown/margin_stats/quality_stats (속성별)

## E-2. 개선 루프 — 버전 A/B

```bash
# 프롬프트(prompts/<버전>/person.yaml) 수정했다고 치고, 새 버전으로 run+submit
python3 lab.py run -X plr_v1.6_test --dataset datasets/mine --model mock
python3 lab.py submit --dataset mine --run-dir datasets/mine -X plr_v1.6_test
# → 리더보드 http://127.0.0.1:8890/d/mine 에 두 버전이 나란히 (Δ 비교)
```

전체 개선 워크플로·반납(`lab port`)은 [HANDOFF.md](HANDOFF.md).

## E-3. 실측(GPU) 전환

`--model mock` → `--model gemma`만 바꾸면 진짜 Gemma로 채점. 전제: 전용 GPU +
Gemma GGUF + 사람 라벨 골든셋([INSTALL.md](INSTALL.md)). 서버 흐름(push/submit)은 **동일**.

## E-4. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `invalid or missing X-Auth-Token` (401) | 서버·lab의 `EVAL_SERVER_TOKEN`을 **같은 값**으로 |
| `--server 또는 EVAL_SERVER_URL이 필요합니다` | lab에 `export EVAL_SERVER_URL=...` (또는 `--server` 직접) |
| `⚠ hash unverified` | `run_provenance.json` 없이 제출. `lab run`(C-1) 먼저 |
| `manifest.yaml not found in archive` | tar에 manifest.yaml이 top-level/단일하위에 없음. `-C datasets mine` 형태로 |
| `validate-dataset FAILED` / 422 | 구조/라벨 문제 — `lab validate-dataset`으로 원인 확인 후 재시도 |
| `dataset ... already exists` (409) | 같은 이름 재등록 불가(크롭 불변). 새 이름(`mine_v2`)으로 |
| 리더보드 빈 화면 | 아직 submit 안 함, 또는 `--dataset` 이름이 등록 이름과 불일치 |

## E-5. 정리

```bash
# 서버: 터미널 A Ctrl-C
rm -rf ~/plr-prompt-lab/datasets/mine        # 로컬 산출물(gitignore)
rm -f  ~/plr-prompt-lab/surface.tgz mine.tgz
rm -rf ~/eval_server_data                     # 서버 볼륨(원하면)
```

> `datasets/`·크롭은 사적 데이터 — **절대 커밋 금지**(gitignore 처리됨).
