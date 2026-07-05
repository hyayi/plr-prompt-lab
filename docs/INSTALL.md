# INSTALL — 새 머신에 plr-prompt-lab 설치

이 lab은 이동 가능한 독립 패키지이고, 설치 경로가 두 갈래입니다:

- **GPU-free 경로** — import·테스트·`lab demo`식 mock/합성 사이클.
  Python + `requirements.txt`의 의존성만 있으면 됩니다.
- **실측 경로** — `lab run`이 GPU에서 Gemma로 크롭을 재채점합니다.

---

## 1. Python 환경

Python **3.10 이상**이 필요합니다 (`X | None` 타입 문법 사용).

```bash
cd plr-prompt-lab
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

환경변수 설정 (GPU-free 경로에서는 전부 옵션):

```bash
cp .env.example .env
# .env를 편집한 뒤 필요한 것만 export, 예:
#   export CORE_IR_PATH=/path/to/ziomilitary/core/ir
#   export RESULT_PATH=./results
```

### GPU-free 경로 검증

```bash
python3 -m pytest tests/ -q      # 기대: 99 passed, 4 xfailed
python3 lab.py demo              # mock 전체 사이클 (3초, exit 0)
```

**GPU·DB·Redis·모델 다운로드 전부 없이** 통과해야 정상입니다. 새 머신이나
패키지를 새로 받은 사람의 권장 첫 단계입니다.

---

## 2. llama-cpp-python — GPU 빌드 (실측 전용)

`requirements.txt`가 설치하는 `llama-cpp-python` 휠은 **CPU 전용**입니다.
`lab run`에는 CUDA 빌드가 필요합니다:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

동작하는 CUDA 툴킷 + 대응 NVIDIA 드라이버가 전제입니다. GPU 선택은
`CUDA_VISIBLE_DEVICES`로 (`.env.example` 참고).

---

## 3. 모델 다운로드 (실측 전용)

`lab run`은 **Gemma-4-E4B GGUF** 모델을 VRAM에 올립니다. Hugging Face에서
받아서 env 변수로 지정하세요:

```bash
pip install huggingface_hub
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    --include "*Q4_0*.gguf" "*mmproj*" \
    --local-dir ./models/gemma-4-E4B-it-GGUF

export IR_GEMMA_REPO=unsloth/gemma-4-E4B-it-GGUF
# 옵션: 메인 파일 고정 (없으면 Q4_0 파일 자동 탐지)
export IR_GEMMA_MAIN_FILE=gemma-4-E4B-it-Q4_0.gguf
```

`gemma_backend`는 `IR_GEMMA_MAIN_FILE` / `IR_GEMMA_MMPROJ_FILE` /
`IR_GEMMA_N_CTX` / `IR_GEMMA_N_GPU_LAYERS`도 읽습니다 (해당 모듈 헤더 참고).

---

## 4. 실측에는 GPU가 필요하다

`lab run`은 전용 GPU가 필요합니다 (다른 서비스가 GPU를 물고 있으면 중지
협의 또는 오프피크 사용 — 운영 `ir` 컨테이너 중지는 관리자와 결정).
사람이 라벨한 골든셋(`labels.jsonl`)도 전제입니다 (DATASET_SPEC.md 참고).

```bash
python3 lab.py run  -X plr_v1.5_cot --dataset datasets/my_test
python3 lab.py dataset-push --dataset datasets/my_test  # 데이터셋을 평가 서버에 등록 (최초 1회)
python3 lab.py submit --dataset my_test --run-dir datasets/my_test -X plr_v1.5_cot --pull
# 서버 채점 → metrics.json + report.html + gallery.html → datasets/my_test/pulled/ 로 회수
```

---

## 5. GPU-free 온보딩 (가장 빠른 길)

GPU도 실데이터도 없이 전체 eval 사이클을 체험하려면:

```bash
python3 lab.py demo --keep       # 합성 데이터셋 + mock 모델 전체 사이클
```

`datasets/demo/`에 산출물이 남고, 신규 체크아웃의 배선이 올바른지
GPU·라벨 준비 전에 확인하는 가장 빠른 방법입니다. 이어서
[`docs/TUTORIAL.md`](TUTORIAL.md)부터 따라 하세요.

---

## 6. Docker (권장 배포)

lab·서버 각각 Dockerfile이 있습니다. 소스는 빌드 시점에 고정되고, **런타임엔
토큰·주소·볼륨만** 정하면 됩니다. 아래는 요약 — **완결형 배포 가이드는
[DOCKER.md](DOCKER.md)** (사전 준비·전체 흐름·트러블슈팅·보안 포함).

### 6-1. lab (GPU 실측)

`Dockerfile`이 CUDA 베이스 위에 `llama-cpp-python`을 **GPU 빌드**하고, 모델
(Gemma-4-E4B GGUF)은 **첫 `lab run` 때 자동 다운로드**됩니다(이미지에 안 굽고 HF
캐시 볼륨에 보관). 크롭·`.git`은 `.dockerignore`로 이미지에서 제외됩니다.

```bash
cd ~/plr-prompt-lab
docker build -t plr-prompt-lab .          # CUDA llama-cpp 빌드 — 수 분 소요
```

런타임에 정하는 것은 **3가지 env + 볼륨 + --gpus** 뿐:

```bash
docker run --rm --gpus all \
  -e EVAL_SERVER_URL=http://<서버>:8890 \    # 평가 서버 주소
  -e EVAL_SERVER_TOKEN=<토큰> \               # 서버 변이 토큰
  -e HF_TOKEN=<hf_...> \                       # HuggingFace 토큰(Gemma 다운로드 인증)
  -v "$PWD/datasets:/app/datasets" \          # 데이터셋(크롭+라벨)
  -v plr_hf_cache:/hf-cache \                  # 모델 캐시(첫 run 이후 재사용)
  plr-prompt-lab run -X plr_v1.5_cot --dataset datasets/mine --model gemma

docker run --rm --gpus all -e EVAL_SERVER_URL=... -e EVAL_SERVER_TOKEN=... \
  -v "$PWD/datasets:/app/datasets" -v plr_hf_cache:/hf-cache \
  plr-prompt-lab submit --dataset mine --run-dir datasets/mine -X plr_v1.5_cot --pull
```

`docker-compose.example.yml`로도 동일하게(1회성 `run --rm`):

```bash
# 파일에서 EVAL_SERVER_URL·EVAL_SERVER_TOKEN·HF_TOKEN 편집한 뒤
docker compose -f docker-compose.example.yml run --rm lab \
  run -X plr_v1.5_cot --dataset datasets/mine --model gemma
```

> 모델 env: `IR_GEMMA_REPO`(기본 `unsloth/gemma-4-E4B-it-GGUF`), 옵션
> `IR_GEMMA_MAIN_FILE`/`IR_GEMMA_N_GPU_LAYERS`/`CUDA_VISIBLE_DEVICES`.
> GPU-free 체험(demo/테스트)은 `--gpus` 없이 같은 이미지로 됩니다(mock은 GPU 미로드).

### 6-2. 평가 서버 (GPU 불필요)

서버는 채점·렌더만 하므로 GPU가 필요 없습니다. `server/Dockerfile` + compose:

```bash
cd ~/plr-eval-server
docker build -t plr-eval-server -f server/Dockerfile .
docker run -d -p 8890:8890 \
  -e EVAL_SERVER_TOKEN=<토큰> -e EVAL_SERVER_DATA=/data \
  -v plr_eval_data:/data plr-eval-server
# 또는: docker compose -f server/docker-compose.example.yml up -d --build
```

> **보안**: lab 이미지엔 크롭·`.git`(비밀 백업 브랜치)이 `.dockerignore`로 빠져
> 있습니다. 두 이미지 모두 **사내 LAN 전용** — 공인망 노출 금지(크롭=사적 CCTV).
