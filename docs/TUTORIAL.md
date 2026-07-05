# 튜토리얼 — lab ↔ 평가 서버 (따라치기, 전체판)

이 문서는 **lab 클라이언트(`plr-prompt-lab`)** 에서 데이터셋을 **평가 서버
(`plr-eval-server`)** 에 제출해 채점·리포트·갤러리·리더보드를 받아보는 **워크플로**를
복붙으로 따라 할 수 있게 정리한 것입니다. CLI 경로 / 웹 UI 수동 업로드 **두 가지**를 다룹니다.

전부 **GPU 없이**(mock 모델) 동작합니다. 실측(진짜 Gemma) 전환은 §E-3.

관련 문서(역할 분리):
- **서버 띄우기** → 서버 레포 `~/plr-eval-server`의 `README.md` (§실행)
- **데이터셋 만들기**(manifest·라벨·검증) → [DATASET_GUIDE.md](DATASET_GUIDE.md) (실습) / [DATASET_SPEC.md](DATASET_SPEC.md) (명세)
- **이 문서** → 그 데이터셋을 서버에 제출하는 워크플로

```
[터미널 B: lab]                          [터미널 A: 서버(별도 레포)]
 데이터셋 준비 → validate-dataset
 lab run --model mock  ─ attributes.jsonl + run_provenance.json
 lab dataset-push ───────────────────▶  데이터셋 등록
 lab submit --pull ──────────────────▶  채점 → metrics/report/gallery 렌더
        ◀───────────  pulled/ 로 회수  ─┘
 브라우저 ───────────────────────────▶  리더보드 /d/<dataset>
```

---

# Part 0. 사전 준비 (lab 클라이언트)

```bash
python3 --version    # 3.10+
cd ~/plr-prompt-lab && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 두 레포는 **독립**입니다. 공유하는 건 `evalkit/dataset.py`·`evalkit/provenance.py`
> 두 파일의 byte-identical 복본뿐(`contract/CONTRACT.md`).

---

# Part A. 전제 — 평가 서버가 떠 있어야 함 (터미널 A)

**서버 기동은 서버 레포 소관**입니다. 별도 터미널에서 `~/plr-eval-server`를 띄우세요
— 설치·실행·Docker·env는 서버 레포 [`README.md`](../../plr-eval-server/README.md)의 **§실행** 참고.

요약(서버 레포에서):

```bash
cd ~/plr-eval-server && source .venv/bin/activate
export EVAL_SERVER_DATA=~/eval_server_data EVAL_SERVER_TOKEN=tutorial-token
uvicorn server.app:app --host 0.0.0.0 --port 8890 --workers 1
```

lab 쪽(터미널 B)에서 서버가 떴는지 확인:

```bash
curl -s http://127.0.0.1:8890/health          # {"ok":true,...}
```

> 이 아래(Part B~E)는 전부 **터미널 B(lab)** 에서 실행합니다.

---

# Part B. 데이터셋 준비 (터미널 B)

채점하려면 라벨된 데이터셋이 필요합니다. 둘 중 하나로 준비하세요.

## ① 지금 바로 실습 — GPU-free 합성 데이터셋 (권장 시작점)

```bash
python3 lab.py demo --keep
# datasets/demo/ 에 crops/ + labels.jsonl + manifest.yaml 자동 생성 (GPU·서버 불필요)
```

이 튜토리얼의 나머지(Part C~E)는 **`datasets/demo`** 를 예시로 씁니다. 실제
데이터셋을 쓸 땐 아래 명령의 `demo`를 그 디렉터리/이름으로 바꾸면 됩니다.

## ② 실제 데이터셋 만들기

crops 준비 · `manifest.yaml` 작성(단일/다속성) · `labels.jsonl` 작성(직접/`lab label`) ·
검증까지는 **별도 문서**로 분리했습니다:

- **[DATASET_GUIDE.md](DATASET_GUIDE.md)** — 만드는 법(실습, 처음부터 따라치기)
- [DATASET_SPEC.md](DATASET_SPEC.md) — 형식의 완전한 명세(필드·어휘 레퍼런스)

만든 뒤 **제출 전에 반드시 검증**(라벨 어휘 검증은 클라이언트 담당 — 서버는 신뢰):

```bash
python3 lab.py validate-dataset --dataset datasets/demo    # Result: PASS
```

---

# Part C. CLI로 실행·제출 (터미널 B) — 권장 경로

```bash
cd ~/plr-prompt-lab && source .venv/bin/activate
export EVAL_SERVER_URL=http://127.0.0.1:8890
export EVAL_SERVER_TOKEN=tutorial-token       # 서버와 같은 값!
```

## C-1. 모델 실행 (GPU-free)

```bash
python3 lab.py run -X plr_v1.5_cot --dataset datasets/demo --model mock
```

생성물(모두 `datasets/demo/`):
- `attributes.jsonl` — 크롭당 plr_json 전체(서버가 채점에 씀)
- `predictions.jsonl` — 속성 추출 뷰
- `run_provenance.json` — 실행 지문(surface_hash·model·version…, 서버 무결성 대조용)

> `-X` = 프롬프트 버전 라벨(리더보드 구분키). `--model mock`이 GPU-free. 실측은 `--model gemma`(§E-3).

## C-2. 등록 → 제출 → 회수

```bash
python3 lab.py dataset-push --dataset datasets/demo --name demo
python3 lab.py submit --dataset demo --run-dir datasets/demo -X plr_v1.5_cot --pull
```

출력 예:
```
[submit] run r20260704-124852-80a96b
[submit] aggregate: macro_f1=1.0 macro_acc=1.0 micro_acc=1.0
[submit]   gender: acc=1.0 macro_f1=1.0 n=5
[submit] pulled metrics.json, report.html, gallery.html -> datasets/demo/pulled
```

- `--dataset demo` = 서버 측 이름(dataset-push의 `--name`), `--run-dir` = attributes.jsonl 있는 로컬 dir
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
tar czf demo.tgz -C datasets demo
```

**② run 제출 폼 — 3개 파일**

| 폼 필드 | 정체 | 만드는 법 |
|---|---|---|
| `attributes.jsonl` | 모델이 낸 plr_json 전체(채점 원천) | `lab run`이 `datasets/demo/`에 생성 |
| `surface.tar.gz` | 프롬프트 표면 번들(파서·코어·스키마·prompts·vocab·configs) — 서버가 해시 대조용으로만 보관, **실행 안 함** | 아래 한 줄로 생성 ↓ |
| `run_provenance.json` (선택) | 실행 지문(surface_hash 등). 있으면 `hash verified` 배지 | `lab run`이 `datasets/demo/`에 생성 |

`surface.tar.gz` 만들기 (lab 헬퍼가 정확한 파일 집합을 묶음 — 손으로 tar 뜨지 마세요):

```bash
python3 -c "from pathlib import Path; from runners.client import build_surface_bundle; \
Path('surface.tgz').write_bytes(build_surface_bundle(Path('.')))"
```

## D-2. 폼에 올리기

- 데이터셋 등록 폼: **토큰**=`tutorial-token`, **이름**=`demo`, **tar.gz**=`demo.tgz` → 등록
- run 제출 폼: **토큰**·**데이터셋 이름**(`demo`)·**version_label**(`plr_v1.5_cot`) 입력 →
  `attributes.jsonl`, `surface.tgz`, `run_provenance.json`(선택) 선택 → 제출

## D-3. (참고) 웹 폼 = 이 curl과 동일

폼 제출은 아래 multipart와 정확히 같습니다(자동화/디버깅용):

```bash
# 데이터셋 등록
curl -X POST http://127.0.0.1:8890/api/datasets \
  -H "X-Auth-Token: tutorial-token" \
  -F "name=demo" -F "created_by=me" \
  -F "archive=@demo.tgz;type=application/gzip"
# → {"name":"demo","n_crops":5}

# run 제출
curl -X POST http://127.0.0.1:8890/api/runs \
  -H "X-Auth-Token: tutorial-token" \
  -F "dataset=demo" -F "version_label=plr_v1.5_cot" -F "submitted_by=me" \
  -F "attributes=@datasets/demo/attributes.jsonl;type=application/json" \
  -F "surface=@surface.tgz;type=application/gzip" \
  -F "provenance=@datasets/demo/run_provenance.json;type=application/json"
# → {"run_id":"...","hash_verified":true,"aggregate":{"macro_f1":1.0,...},...}
```

> 웹 폼에 "CLI 권장" 안내가 붙은 이유: `surface.tar.gz`를 손으로 정확히 묶기 번거롭기
> 때문입니다. `lab submit`은 이 번들을 메모리에서 자동 생성해 한 번에 올립니다.

---

# Part E. 결과·개선·실측·문제해결

## E-1. 결과 보기

```bash
ls datasets/demo/pulled/          # gallery.html  metrics.json  report.html
xdg-open datasets/demo/pulled/gallery.html   # 오답 우선·속성 태그·AND/OR 필터
xdg-open datasets/demo/pulled/report.html    # 버전 비교표
```

- 서버 리더보드: <http://127.0.0.1:8890/d/demo>
- run 상세: <http://127.0.0.1:8890/r/&lt;run_id&gt;>
- `metrics.json` 필드: accuracy/recall/precision/f1/macro_f1/bias/confusion/
  pred_unknown/n_label_unknown/margin_stats/quality_stats (속성별)

## E-2. 개선 루프 — skills로 프롬프트 정리·개선

`skills/`는 **Claude Code(에이전트)가 따라가는 워크플로 지침**입니다. Claude에게
"**<스킬명> 스킬대로 해줘**"라고 요청하면 해당 `skills/<name>/SKILL.md`를 읽어 수행합니다.
개선 한 바퀴는 이렇게 돕니다:

### 1) 오답 분석 → 개선안: `improve-prompt`
방금 `submit --pull`로 받은 결과로 **근거 있는** 개선안을 만듭니다. Claude에게:

> "improve-prompt 스킬대로 `datasets/demo`의 submit 결과를 분석해서 개선안 만들어줘"

Claude가 `pulled/metrics.json`(지표)·`pulled/gallery.html`(오답 크롭)·로컬
`raw_responses.jsonl`(모델 원문)·`crops/<obj_id>.jpg`(직접 봄)를 읽고 **6역할 토론
루프**(최대 3라운드)로 제안을 냅니다. ⚠ 측정 없는 "문구 다듬기"는 하지 않습니다.

### 2) 새 버전 작성: `author-prompt`
개선안대로 새 프롬프트 버전을 만듭니다. Claude에게:

> "author-prompt 스킬대로 그 제안을 반영한 `prompts/plr_v1.6_test/` 버전 만들어줘"

강제커밋(unknown 선택지 금지)·버전명 ≤16자·enum 주입·파서 계약 등 **깨지면 안 되는
계약**을 스킬이 강제합니다(상세: [HANDOFF.md](HANDOFF.md) "skills로 프롬프트 작성·개선").

### 3) (프롬프트 외 표면을 건드리면) `co-change`
어휘(`schema/vocab.yaml`)·파서·스키마·전처리까지 손대면, **고치기 전에** co-change로
동행 수정 목록을 적용(조용한 드리프트 방지).

### 4) 재실행 → 리더보드 Δ
```bash
python3 lab.py run -X plr_v1.6_test --dataset datasets/demo --model gemma  # (mock이면 --model mock)
python3 lab.py submit --dataset demo --run-dir datasets/demo -X plr_v1.6_test
# → 리더보드 http://127.0.0.1:8890/d/demo 에 plr_v1.5_cot vs plr_v1.6_test 나란히 (Δ 비교)
```

이겼으면 `lab port`로 core/ir에 반납합니다(승격 절차: [HANDOFF.md](HANDOFF.md)).

## E-3. 실측(GPU) 전환

`--model mock` → `--model gemma`만 바꾸면 진짜 Gemma로 채점. 전제: 전용 GPU +
Gemma GGUF + 사람 라벨 골든셋([INSTALL.md](INSTALL.md)). 서버 흐름(push/submit)은 **동일**.

## E-4. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `invalid or missing X-Auth-Token` (401) | 서버·lab의 `EVAL_SERVER_TOKEN`을 **같은 값**으로 |
| `--server 또는 EVAL_SERVER_URL이 필요합니다` | lab에 `export EVAL_SERVER_URL=...` (또는 `--server` 직접) |
| `⚠ hash unverified` | `run_provenance.json` 없이 제출. `lab run`(C-1) 먼저 |
| `manifest.yaml not found in archive` | tar에 manifest.yaml이 top-level/단일하위에 없음. `-C datasets demo` 형태로 |
| `validate-dataset FAILED` / 422 | 구조/라벨 문제 — `lab validate-dataset`으로 원인 확인 후 재시도 |
| `dataset ... already exists` (409) | 같은 이름 재등록 불가(크롭 불변). 새 이름(`demo_v2`)으로 |
| 리더보드 빈 화면 | 아직 submit 안 함, 또는 `--dataset` 이름이 등록 이름과 불일치 |

## E-5. 정리

```bash
# 서버: 터미널 A Ctrl-C
rm -rf ~/plr-prompt-lab/datasets/demo        # 로컬 산출물(gitignore)
rm -f  ~/plr-prompt-lab/surface.tgz demo.tgz
rm -rf ~/eval_server_data                     # 서버 볼륨(원하면)
```

> `datasets/`·크롭은 사적 데이터 — **절대 커밋 금지**(gitignore 처리됨).
