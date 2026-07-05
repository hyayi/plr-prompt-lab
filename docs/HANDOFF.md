# PLR Prompt Lab — 외부 프롬프트 엔지니어 인수인계 가이드

**ZioVision 추론 서비스를 직접 건드리지 않고 PLR 프롬프트를 개선하는
프롬프트 엔지니어**를 위한 문서다. (lab은 PLR 전용 — 텍스트 검색
파이프라인은 2026-07 제거.) 손으로 따라 하는 실습은
[GUIDE.html](GUIDE.html)이 짝 문서다 — 이 문서는 역할·규칙·경계 중심.

---

## 누구를 위한 문서인가

`plr-prompt-lab` 패키지를 전달받은 프롬프트 엔지니어. 할 일은 PLR(객체
속성 추출)을 움직이는 프롬프트를 반복 개선하고, 라벨된 골든셋에서 효과를
측정하고, 이긴 버전 + diff를 ZioVision 팀에 반납하는 것이다. 운영 DB,
Redis, 라이브 GPU 서비스에는 접근 권한이 **없고, 필요하지도 않다**.

---

## 무엇을 편집하는가

### 유일한 원천: `prompts/<버전>/` 디렉터리 (기능별 yaml 5개)

2026-07부터 프롬프트 텍스트는 **전부 yaml에 산다** — `plr_prompts.py`는
프롬프트 텍스트 0줄, 로드와 조합만 한다:

```
prompts/plr_v1.5_cot/        # 버전당 디렉터리 1개, 기능당 파일 1개
  person.yaml                # system + user_cot(운영 기본) + user_plain + commit_enums
  vehicle.yaml               # system + user
  query_parser.yaml          # 검색 쿼리 파서 (core/ir 검색용 — 건드리지 말 것)
  vqa.yaml                   # 검색 VQA system
  retry.yaml                 # 스키마 실패 재시도 템플릿
```

- **새 버전 만들기** = 디렉터리 복사 후 수정 (`cp -r prompts/plr_v1.5_cot
  prompts/plr_v1.6_test`). 이름 ≤16자. `lab run -X plr_v1.6_test`가 그
  디렉터리를 로드한다 — 실험의 prompt 축이 이렇게 실제 변형들을 비교한다.
- **현재/기본 버전**(`-X` 없이 도는 것) = `plr_prompts.PROMPT_VERSION_YAML_COT`
  상수가 가리키는 디렉터리. 이 포인터 bump가 승격의 핵심이다.
- CoT 토글: `IR_PLR_REASON=on|off` (on = 근거-먼저 user_cot, 토큰 ~+35%).
  experiment yaml의 `reasons:` 축으로도 지정 가능.
- **어휘(enum) 확장은 프롬프트가 아니라 `schema/vocab.yaml`** — 주입·파싱·
  검증이 자동 동행한다. 프롬프트의 `{colors}` 같은 주입 자리는 지우지 말 것.
- 출력 슬롯을 추가/변경하면 파서(`plr_parse.py`)와 스키마(`plr_schema.py`)가
  동행해야 한다 — `skills/co-change/SKILL.md`의 매트릭스를 먼저 볼 것
  (스키마 선언 누락은 `tests/test_schema_declares_parser_keys.py`가 잡는다).

### 지켜야 할 프롬프트 계약 (전체는 `skills/author-prompt/SKILL.md`)

- **강제커밋(v1.5+)**: `unknown`을 답 선택지로 제시 금지. 저신뢰는
  `margins` 블록으로 — `test_live_prompt_never_offers_unknown`이 감시.
- enum은 손글씨 금지, placeholder 주입 유지 (`commit_enums: true`).
- 출력 형식은 파서와 한 몸 — 형식을 바꾸면 co-change 매트릭스 따라 동행.

### 건드리지 말 것

| 파일/영역 | 이유 |
|---|---|
| `runners/re_score.py` | 재채점 러너 — 프롬프트를 고쳐라, 러너 말고 |
| `plr_core.py` | PLR 추론 코어 |
| `plr_schema.py`, `plr_parse.py` | 스키마·파서 — 출력 슬롯 변경 시에만, co-change 절차로 |
| `gemma_model.py`, `gemma_backend.py` | GPU 모델 로더 — 프롬프트 작업 범위 밖 |
| `eval/` 채점 스크립트 | 버그 수정만 — 점수 부풀리기 금지 |
| `core/ir/` | 운영 서비스 — 직접 수정 금지, diff 반납으로 |

---

## 반복 루프

```
데이터셋 준비 → lab run → lab submit --pull → 서버 채점·gallery/report 분석
                                  ↑                    │
                                  └── 새 버전 작성 ◄────┘
                                        │ (이겼을 때)
                          버전별 submit → 서버 리더보드 Δ → lab port → ZioVision에 반납
```

### 1. 데이터셋 준비 / 지정

데이터셋 = 라벨된 크롭 디렉터리. 정확한 레이아웃과 스키마는
[DATASET_SPEC.md](DATASET_SPEC.md).

**기존 데이터셋을 받았다면** 풀고 검증만:

```bash
python3 lab.py validate-dataset --dataset /path/to/my_dataset/
```

**새로 만들려면** (실 비디오 + DB 필요 — 운영자 단계): `prepare-dataset`
스킬 또는 `lab build-golden` + `lab label`. 외부 엔지니어는 보통 라벨
완료된 데이터셋을 전달받는다.

### 2. 재채점 (실데이터는 GPU 필요)

```bash
python3 lab.py run -X plr_v1.5_cot --dataset /path/to/my_dataset/
```

데이터셋의 모든 크롭에 지정 버전 프롬프트로 Gemma를 **크롭당 1회** 호출하고
`predictions.jsonl`(추출 뷰) / `attributes.jsonl`(plr_json 전체) /
`raw_responses.jsonl`(모델 원문+토큰수)을 쓴다. `-A`는 옵션 — 모델 호출은
속성과 무관하며, 사람/차량 혼합 데이터셋은 labels.jsonl 행의
`object_type`이 크롭별 프롬프트를 정한다.

**실측 전제조건** ([INSTALL.md](INSTALL.md)):
- 전용 GPU (다른 서비스가 VRAM을 물고 있으면 중지 협의 — 운영 `ir` 컨테이너
  중지/재기동은 관리자 결정).
- Gemma-4-E4B GGUF 다운로드 + env 설정.
- 사람이 라벨한 `labels.jsonl`.

**GPU 없는 체험**: `python3 lab.py demo` — mock 전체 사이클.

### 3. 제출·채점 — 서버가 라벨된 전 속성을 채점

```bash
python3 lab.py dataset-push --dataset /path/to/my_dataset/   # 최초 1회 등록
python3 lab.py submit --dataset my_dataset --run-dir /path/to/my_dataset/ -X plr_v1.5_cot --pull
```

평가 서버(별도 레포 `~/plr-eval-server`)가 attributes.jsonl 에서 라벨된 속성을
전부 재추출·채점하고, `metrics.json` + `report.html`(버전 비교표) +
`gallery.html`(오답 우선, 속성별 태그, AND/OR 오답 필터)을 렌더해
`--pull` 로 `<run-dir>/pulled/` 에 회수한다. 리더보드는 서버 `/d/<dataset>`.

### 4. 숫자 읽기

```
=== gender eval: plr_v1.5_cot (n=150) ===
accuracy: 0.927 (139/150)   Δ vs plr_v1.4_cot: +0.014 (0.913 → 0.927)
bias female->male: 0.067 (5/75)   Δ: -0.027
margin split (>= 0.7): high acc=0.96 (n=120)  low acc=0.77 (n=30)
recall / precision / f1 / confusion ...
```

| 지표 | 의미 |
|---|---|
| `accuracy` | 전체 정답률 |
| `bias female->male` | 여성이 남성으로 오분류되는 비율 (이 쌍에선 낮을수록 좋음) |
| `recall`/`precision`/`f1` | 클래스별 성능 (+macro_f1) |
| `pred_unknown` | 강제커밋 준수도 — 모델이 그래도 unknown이라 답한 비율 |
| `margin/quality split` | 오답이 저신뢰/저품질 크롭에 몰리는가 (캘리브레이션) |
| `Δ vs <이전>` | 서버 리더보드의 직전 다른 버전 대비 변화 |

### 5. 반복

`prompts/<개선 중인 버전>/person.yaml`을 고치고 2번으로. 서버 리더보드 Δ가
개선 여부를 말해준다. A/B는 버전별로 `run` → `submit` 후 서버 리더보드
(`/d/<dataset>`)에서 버전 간 지표를 비교한다.

**규칙**:
- 측정 없이 프롬프트를 바꾸지 말 것 — 읽기 좋아졌는데 점수가 내려간 건
  개선이 아니다.
- 라벨 없는 데이터셋으로 채점하지 말 것 — 숫자가 무의미하다.
- `core/ir`에 직접 커밋하지 말 것 — 아래 반납 절차로.

### 6. ZioVision에 반납 (lab port)

새 버전이 기준선을 이겼을 때:

```bash
python3 lab.py port [--core-ir /path/to/ziomilitary/core/ir]
```

프롬프트 표면 전체(`prompts/**`, `schema/vocab.yaml`, `plr_prompts.py`,
`plr_parse.py`, `plr_core.py`, `plr_schema.py`, `preprocess.py`)의 lab ↔
core/ir unified diff를 출력한다. 기본은 **읽기 전용** — core/ir에 쓰지
않는다. **개발 중에 diff가 있는 것은 정상**이다 (실험 중 표면은 의도적으로
달라져 있음 — `skills/co-change` "개발 단계" 참고).

ZioVision 팀에 보내는 것:

1. `lab port`가 출력한 diff (복사 또는 파일로).
2. 이긴 `prompts/<버전>/` 디렉터리.
3. 전/후 Δ를 보여주는 서버 리더보드 링크 또는 `lab submit --pull`로 회수한
   `metrics.json` + `report.html` (채점·이력은 별도 평가 서버 레포 담당).

적용은 ZioVision이 `skills/co-change`의 **승격 체크리스트**(byte-sync →
양쪽 테스트 → `PROMPT_VERSION_YAML_COT` bump)로 수행하고, 배포는 컨테이너
재시작(= 재인덱싱 트리거)이므로 별도 결정 사항이다. **직접 적용 금지.**

---

## skills로 프롬프트 작성·개선

`skills/`는 Claude Code(에이전트)가 따라가는 워크플로 지침입니다. Claude에게
**"<스킬명> 스킬대로 해줘"** 라고 요청하면 해당 `skills/<name>/SKILL.md`를 읽어 수행합니다.
프롬프트 작업엔 셋을 씁니다:

### ① 새 버전 작성 — `author-prompt`
`prompts/<version>.yaml`을 만들 때. 지켜야 할 계약을 강제합니다:
- **강제커밋(v1.5+)**: `unknown`을 답 선택지로 주지 않음 — 낮은 확신은 `margins` 블록으로
- **버전명 ≤ 16자·재사용 금지**(`plr_v<major>.<minor>_<tag>`) — 재인덱싱 트리거·리더보드 키
- **enum은 주입**(`{colors}` 등 플레이스홀더 유지, `commit_enums: true`) — 손으로 쓰면 vocab과 드리프트
- **출력 스키마 = 파서 계약**(`plr_parse.parse_plr_response` 파싱 가능) · **marker 문단 유지**
- 작성 즉시 `lab run -X <version> --dataset D`로 실행 가능

### ② 결과 기반 개선 — `improve-prompt`
`lab submit --pull` **후**에, 회수한 결과로 **근거 있는** 개선안을 만듭니다:
- 입력: `pulled/metrics.json`(지표)·`pulled/report.html`(버전비교)·`pulled/gallery.html`(오답 크롭)
  + 로컬 `raw_responses.jsonl`(모델 원문)·`crops/<obj_id>.jpg`(**직접 봄**)
- **6역할 토론 루프**(최대 3라운드) — 판정자 수용까지. 모든 주장은 **지표 또는 실제 본 크롭(obj_id)**으로 뒷받침
- ⚠ 측정 없는 "문구 다듬기" 금지

### ③ 표면 수정 전 필독 — `co-change`
프롬프트·어휘(`schema/vocab.yaml`)·파서·스키마·전처리 중 **무엇이든 고치기 전에** 열어,
변경 유형별 **동행 수정 목록**을 적용(조용한 드리프트 방지). 승격(core/ir 반영)은 별도 체크리스트.

### 전형적 사이클
```
개선 버전 선택 → (author-prompt: 새 버전 yaml) → lab run → lab submit --pull
   → (improve-prompt: 오답 분석·개선안) → 새 버전 → 리더보드 Δ → 승격(lab port)
```

---

## 빠른 참조

```bash
python3 lab.py demo                                       # GPU-free 온보딩
python3 lab.py validate-dataset --dataset D               # 받은 데이터셋 검증
python3 lab.py run -X plr_v1.5_cot --dataset D            # 재채점 (GPU+모델)
python3 lab.py submit --dataset N --run-dir D -X V --pull # 서버 제출→채점, metrics/report/gallery 회수
python3 lab.py port                                       # 반납용 diff
python3 -m pytest tests/ -q                               # 전체 테스트 (72 passed, 4 xfailed)
```

---

## 참고 문서

- [GUIDE.html](GUIDE.html) — 복붙 실습 가이드 (전체 개선 루프 + 명령 레시피)
- [DATASET_SPEC.md](DATASET_SPEC.md) — 데이터셋 형식·파일 스키마 전체
- [INSTALL.md](INSTALL.md) — Python 환경, GPU 빌드, 모델 다운로드
- `SEED.md` — 이 lab이 추출된 core/ir 커밋
- `skills/` — author-prompt(작성 계약) · improve-prompt(개선 루프) ·
  prepare-dataset(데이터셋 구성) · co-change(동행 수정 매트릭스)
- 지표 이력·리더보드 — 별도 평가 서버 레포(`~/plr-eval-server`), `lab submit --pull`로 회수

---

## 금지 사항 (요약)

- **측정 없이 프롬프트 변경 금지.** 편집 전/후 `lab run` + `lab submit` —
  감으로 하는 건 워크플로가 아니다.
- **라벨 없는 데이터셋 반출 금지.** `labels.jsonl`이 사람 검증까지 끝나야
  eval이 의미를 가진다.
- **`core/ir` 직접 수정 금지.** diff를 `lab port`로 반납하고 ZioVision이
  자체 재검증 후 적용한다.
- **라이브 `ir` 컨테이너가 떠 있는 동안 `lab run` 금지** — GPU 경합/OOM.
  중지는 관리자와 협의.
- **크롭/라벨 git 커밋 금지.** `datasets/`와 `eval/golden/*/crops/`는
  gitignore — 사적 CCTV 데이터는 로컬에만 둔다.
