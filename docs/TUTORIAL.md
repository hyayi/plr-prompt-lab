# 튜토리얼 — lab ↔ 평가 서버 연동 (따라치기)

이 문서는 **평가 서버(`plr-eval-server`)를 띄우고**, **lab 클라이언트(`plr-prompt-lab`)로
데이터셋을 등록·제출**해서 **채점·리포트·갤러리·리더보드**를 받아보는 전 과정을
복붙으로 따라 할 수 있게 정리한 것입니다.

전부 **GPU 없이**(mock 모델) 동작합니다. 실측(진짜 Gemma) 전환은 §7 참고.

- 두 레포 위치(이 문서 기준):
  - lab(클라이언트): `~/plr-prompt-lab`
  - 서버: `~/plr-eval-server`
- 터미널 2개를 씁니다: **터미널 A = 서버**, **터미널 B = lab 클라이언트**.

```
[터미널 B: lab]                         [터미널 A: 서버]
 lab run --model mock  ─ attributes.jsonl
 lab dataset-push ───────────────────▶  데이터셋 등록
 lab submit --pull ──────────────────▶  채점 → metrics/report/gallery 렌더
        ◀───────────  pulled/ 로 회수  ─┘
 브라우저 ───────────────────────────▶  리더보드 /d/<dataset>
```

---

## 0. 사전 준비

```bash
# 파이썬 3.10+ 확인
python3 --version
```

두 레포에 각각 의존성을 설치합니다(가상환경 권장).

```bash
# 서버
cd ~/plr-eval-server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# lab (별도 터미널/별도 venv)
cd ~/plr-prompt-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 서버와 lab은 **독립 레포**입니다. 공유하는 건 `evalkit/dataset.py`·`evalkit/provenance.py`
> 두 파일의 byte-identical 복본뿐입니다(자세한 건 `contract/CONTRACT.md`).

---

## 1. 터미널 A — 평가 서버 띄우기

```bash
cd ~/plr-eval-server
source .venv/bin/activate

# 데이터 저장 위치와 인증 토큰을 정합니다(원하는 값으로)
export EVAL_SERVER_DATA=~/eval_server_data     # 데이터셋·run 파일이 쌓일 볼륨
export EVAL_SERVER_TOKEN=tutorial-token        # 변이 API(X-Auth-Token) 값

uvicorn server.app:app --host 0.0.0.0 --port 8890 --workers 1
```

> `--workers 1` **필수**: 쓰기 잠금이 단일 프로세스 asyncio.Lock이라 다중 워커 금지.

다른 터미널에서 살아있는지 확인:

```bash
curl -s http://127.0.0.1:8890/health
# {"ok":true,"data_root":"...","rebuild":{"runs":[],"quarantined":[]}}
```

브라우저로 홈/리더보드도 열립니다: <http://127.0.0.1:8890/>

---

## 2. 터미널 B — lab 클라이언트 준비 + 데이터셋 만들기

서버 주소·토큰을 환경변수로 걸어두면 매 명령에 `--server`/`--token`을 안 붙여도 됩니다.

```bash
cd ~/plr-prompt-lab
source .venv/bin/activate

export EVAL_SERVER_URL=http://127.0.0.1:8890
export EVAL_SERVER_TOKEN=tutorial-token        # 서버와 같은 값!
```

GPU 없이 바로 써볼 **합성 데이터셋**을 만듭니다(크롭+라벨+manifest 생성):

```bash
python3 lab.py demo --keep
# datasets/demo/ 에 crops/ · labels.jsonl · manifest.yaml 생성
```

---

## 3. 연동 — run → dataset-push → submit --pull

### 3-1. 모델 실행 (GPU-free, mock)

```bash
python3 lab.py run -X plr_v1.5_cot --dataset datasets/demo --model mock
# → datasets/demo/attributes.jsonl (plr_json 전체)
#   datasets/demo/predictions.jsonl (속성 추출 뷰)
#   datasets/demo/run_provenance.json (표면 해시 — 서버 대조용)
```

> `--model mock` 이 GPU-free 핵심입니다. 실측은 `--model gemma`(§7).
> `-X` 는 프롬프트 버전 라벨(리더보드에서 버전을 구분하는 키).

### 3-2. (선택) 데이터셋 형식 검증

```bash
python3 lab.py validate-dataset --dataset datasets/demo
```

> 라벨 어휘(enum) 검증은 **클라이언트 담당**입니다. 서버는 이 검증을 신뢰하고
> push 시엔 구조(파일 존재/파싱)만 확인합니다.

### 3-3. 데이터셋을 서버에 등록

```bash
python3 lab.py dataset-push --dataset datasets/demo --name demo
# [dataset-push] registered 'demo' (crops=5)
```

> `--name` 은 서버 측 데이터셋 이름(생략 시 디렉터리명). 크롭은 등록 후 **불변**,
> 라벨만 이후 정정 가능. 구성이 바뀌면 새 이름(`demo_v2`)으로 등록하세요.

### 3-4. run 산출물을 제출하고 결과를 회수 (`--pull`)

```bash
python3 lab.py submit --dataset demo --run-dir datasets/demo -X plr_v1.5_cot --pull
```

출력 예:

```
[submit] run r20260704-122603-5afc57
[submit] aggregate: macro_f1=1.0 macro_acc=1.0 micro_acc=1.0
[submit]   gender: acc=1.0 macro_f1=1.0 n=5
[submit] pulled metrics.json, report.html, gallery.html -> datasets/demo/pulled
```

- `--dataset demo` = 3-3에서 등록한 **서버 측 이름**
- `--run-dir datasets/demo` = `attributes.jsonl`(+`run_provenance.json`)이 있는 로컬 디렉터리
- `--pull` = 서버가 채점·렌더한 `metrics.json`/`report.html`/`gallery.html`을
  `<run-dir>/pulled/` 로 회수 (저장 위치 바꾸려면 `--out <경로>`)

> **`⚠ hash unverified`** 배지가 뜨면 `run_provenance.json` 없이 제출된 것입니다
> (3-1의 `lab run`을 거치면 생성됩니다). 채점은 정상이며 표면 해시 대조만 생략됩니다.

---

## 4. 결과 보기

### 로컬로 회수된 파일

```bash
ls datasets/demo/pulled/
# gallery.html  metrics.json  report.html

# 브라우저로 열기 (예: 리눅스)
xdg-open datasets/demo/pulled/gallery.html   # 오답 우선, 속성 태그, AND/OR 필터
xdg-open datasets/demo/pulled/report.html    # 버전 비교표
```

- `metrics.json` — accuracy/recall/precision/f1/bias/confusion/pred_unknown/
  margin_stats/quality_stats (속성별)
- `gallery.html` — 크롭 이미지가 base64로 내장되어 오프라인에서도 열림
- `report.html` — 데이터셋의 버전 추이

### 서버 리더보드 (브라우저)

- 데이터셋 리더보드: <http://127.0.0.1:8890/d/demo>
- 개별 run 상세: <http://127.0.0.1:8890/r/&lt;run_id&gt;>
- 홈: <http://127.0.0.1:8890/>

---

## 5. 개선 루프 — 두 번째 버전으로 A/B

프롬프트를 고쳤다고 치고(여기선 버전 라벨만 바꿔 시연), 새 버전을 run→submit 하면
같은 데이터셋 리더보드에 **버전이 나란히** 쌓여 Δ를 비교할 수 있습니다.

```bash
# 새 버전 실행 + 제출
python3 lab.py run -X plr_v1.6_test --dataset datasets/demo --model mock
python3 lab.py submit --dataset demo --run-dir datasets/demo -X plr_v1.6_test
```

이제 <http://127.0.0.1:8890/d/demo> 를 새로고침하면 `plr_v1.5_cot` vs `plr_v1.6_test`
두 버전이 리더보드에 함께 보입니다.

> 실제 개선 루프: `prompts/<버전>/person.yaml` 편집 → `lab run` → `lab submit` →
> 리더보드 Δ 확인 → 이겼으면 `lab port`로 반납. 자세한 건 [HANDOFF.md](HANDOFF.md).

---

## 6. 전체 스크립트 (복붙 한 번에)

터미널 A(서버)가 떠 있는 상태에서, 터미널 B:

```bash
cd ~/plr-prompt-lab && source .venv/bin/activate
export EVAL_SERVER_URL=http://127.0.0.1:8890 EVAL_SERVER_TOKEN=tutorial-token

python3 lab.py demo --keep
python3 lab.py run -X plr_v1.5_cot --dataset datasets/demo --model mock
python3 lab.py dataset-push --dataset datasets/demo --name demo
python3 lab.py submit --dataset demo --run-dir datasets/demo -X plr_v1.5_cot --pull

xdg-open datasets/demo/pulled/report.html
# 리더보드: http://127.0.0.1:8890/d/demo
```

---

## 7. 실측(GPU) 전환

`--model mock` 을 `--model gemma` 로 바꾸면 진짜 Gemma로 크롭을 재채점합니다.
전제조건(자세한 건 [INSTALL.md](INSTALL.md)):

1. 전용 GPU (운영 `ir` 컨테이너가 VRAM을 물고 있으면 중지 협의 — 관리자 결정).
2. Gemma GGUF 다운로드 + env 설정.
3. 사람이 라벨한 `labels.jsonl`(합성 demo가 아닌 실제 골든셋).

```bash
python3 lab.py run -X plr_v1.5_cot --dataset datasets/my_golden --model gemma
python3 lab.py dataset-push --dataset datasets/my_golden --name my_golden
python3 lab.py submit --dataset my_golden --run-dir datasets/my_golden -X plr_v1.5_cot --pull
```

> 서버 흐름(dataset-push/submit)은 mock과 **완전히 동일**합니다 — 모델만 바뀝니다.

---

## 8. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `invalid or missing X-Auth-Token` (401) | lab의 `EVAL_SERVER_TOKEN` ≠ 서버의 `EVAL_SERVER_TOKEN`. 두 값을 같게. |
| `[submit] --server 또는 EVAL_SERVER_URL이 필요합니다` | `export EVAL_SERVER_URL=...` 안 됨. 또는 `--server http://...` 직접 지정. |
| `⚠ hash unverified` 배지 | `run_provenance.json` 없이 제출됨. `lab run`(§3-1)을 먼저 돌리면 생성됨. 채점 자체는 정상. |
| `dataset ... already exists` (409) | 같은 이름 재등록 불가(크롭 불변 규칙). 새 이름(`demo_v2`)으로 push. |
| `manifest.yaml not found in archive` | 데이터셋 디렉터리에 `manifest.yaml`이 없음. `validate-dataset`로 형식 확인. |
| 리더보드가 비어 있음 | 아직 submit 안 함, 또는 `--dataset` 이름 오타(등록 이름과 일치해야). |
| 포트 8890 충돌 | `uvicorn ... --port 8899` 로 바꾸고 lab의 `EVAL_SERVER_URL`도 동일 포트로. |

---

## 9. 정리(cleanup)

```bash
# 서버 중지: 터미널 A 에서 Ctrl-C

# lab 로컬 산출물(gitignore 대상)
rm -rf ~/plr-prompt-lab/datasets/demo

# 서버 데이터 볼륨(원하면)
rm -rf ~/eval_server_data
```

> `datasets/`·크롭은 사적 데이터라 **절대 커밋 금지**(gitignore 처리됨).
