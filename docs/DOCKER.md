# Docker 설치·배포 가이드

lab(클라이언트, GPU)과 평가 서버 두 이미지를 Docker로 배포하는 완결형 가이드입니다.
소스는 빌드 시점에 고정되고, **런타임엔 토큰·주소·볼륨만** 정합니다.

> ⚠ **사내 LAN 전용** — 두 이미지 모두 공인망 노출 금지(크롭 = 사적 CCTV 데이터).
> lab 이미지엔 크롭·`.git`(비밀 백업 브랜치)이 `.dockerignore`로 빠져 있습니다.

- 실습형 로컬 튜토리얼(비-Docker): [TUTORIAL.md](TUTORIAL.md)
- 데이터셋 만들기: [DATASET_GUIDE.md](DATASET_GUIDE.md)

---

## 0. 사전 준비

| | 서버 | lab (실측) |
|---|---|---|
| Docker | 필요 | 필요 |
| GPU | 불필요 | **필요** (NVIDIA) |
| NVIDIA Container Toolkit | 불필요 | **필요** (`--gpus` 지원) |

GPU 컨테이너용 NVIDIA Container Toolkit 설치·등록(lab만):

```bash
# (미설치 시) 설치는 배포판별 — https://docs.nvidia.com/datacenter/cloud-native/
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # GPU 보이면 OK
```

---

## 1. 평가 서버 (GPU 불필요)

채점·리더보드·report/gallery 렌더만 하므로 GPU가 필요 없습니다.

```bash
cd ~/plr-eval-server

# 빌드 (컨텍스트 = 레포 루트, Dockerfile 은 server/Dockerfile)
docker build -t plr-eval-server -f server/Dockerfile .

# 실행 — 정하는 것: 토큰 + 데이터 볼륨 + 포트
docker run -d --name plr-eval-server -p 8890:8890 \
  -e EVAL_SERVER_TOKEN=<토큰> \        # 변이 API(등록/제출/삭제) 인증값
  -e EVAL_SERVER_DATA=/data \          # 데이터 볼륨 경로(고정)
  -v plr_eval_data:/data \             # 데이터셋·run 파일 영속 볼륨
  --restart unless-stopped \
  plr-eval-server

curl -s http://localhost:8890/health    # {"ok":true,...} 확인
```

compose로도:

```bash
# server/docker-compose.example.yml 의 EVAL_SERVER_TOKEN 편집 후
docker compose -f server/docker-compose.example.yml up -d --build
```

> `--workers 1` 고정(이미지 CMD에 포함) — 쓰기 직렬화가 단일 프로세스 asyncio.Lock.
> 코드/템플릿을 바꾸면 **재빌드+재기동** 필요(`docker compose up -d --build`).

---

## 2. lab (GPU 실측)

CUDA 베이스 위에 `llama-cpp-python`을 GPU 빌드하고, 모델(Gemma-4-E4B GGUF)은
**첫 `lab run` 때 자동 다운로드**됩니다(이미지 미포함 → HF 캐시 볼륨).

```bash
cd ~/plr-prompt-lab
docker build -t plr-prompt-lab .        # CUDA llama-cpp 빌드 — 수 분 소요
```

런타임에 정하는 것은 **env 3개 + 볼륨 2개 + --gpus** 뿐:

| 항목 | 값 | 용도 |
|---|---|---|
| `-e EVAL_SERVER_URL` | `http://<서버>:8890` | 평가 서버 주소 |
| `-e EVAL_SERVER_TOKEN` | 서버와 **동일 토큰** | 제출/등록 인증 |
| `-e HF_TOKEN` | `hf_...` | HuggingFace 모델 다운로드 인증 |
| `-v $PWD/datasets:/app/datasets` | | 데이터셋(크롭+라벨) |
| `-v plr_hf_cache:/hf-cache` | | 모델 캐시(첫 run 이후 재사용) |
| `--gpus all` | | GPU 노출 |

`lab`은 CLI라 서브명령을 그대로 붙입니다(`ENTRYPOINT=lab.py`):

```bash
# 편의를 위해 공통 옵션을 변수로
LAB="docker run --rm --gpus all \
  -e EVAL_SERVER_URL=http://<서버>:8890 \
  -e EVAL_SERVER_TOKEN=<토큰> \
  -e HF_TOKEN=<hf_...> \
  -v $PWD/datasets:/app/datasets \
  -v plr_hf_cache:/hf-cache \
  plr-prompt-lab"

$LAB validate-dataset --dataset datasets/mine       # (선택) 형식 검증
$LAB run    -X plr_v1.5_cot --dataset datasets/mine --model gemma   # 첫 run: 모델 자동 다운로드
$LAB dataset-push --dataset datasets/mine --name mine
$LAB submit --dataset mine --run-dir datasets/mine -X plr_v1.5_cot --pull
```

compose로도(1회성 `run --rm`):

```bash
# docker-compose.example.yml 의 EVAL_SERVER_URL·EVAL_SERVER_TOKEN·HF_TOKEN 편집 후
docker compose -f docker-compose.example.yml run --rm lab \
  run -X plr_v1.5_cot --dataset datasets/mine --model gemma
```

> 모델 env: `IR_GEMMA_REPO`(기본 `unsloth/gemma-4-E4B-it-GGUF`), 옵션
> `IR_GEMMA_MAIN_FILE`/`IR_GEMMA_N_GPU_LAYERS`/`CUDA_VISIBLE_DEVICES`.

---

## 3. 전체 흐름 (서버 + lab)

```bash
# ① 서버 (한 번 띄워두면 계속): §1
docker run -d --name plr-eval-server -p 8890:8890 \
  -e EVAL_SERVER_TOKEN=sekrit -e EVAL_SERVER_DATA=/data \
  -v plr_eval_data:/data plr-eval-server

# ② lab: 데이터셋 준비(datasets/mine) 후 run→push→submit (§2)
docker run --rm --gpus all \
  -e EVAL_SERVER_URL=http://<서버>:8890 -e EVAL_SERVER_TOKEN=sekrit -e HF_TOKEN=hf_... \
  -v $PWD/datasets:/app/datasets -v plr_hf_cache:/hf-cache \
  plr-prompt-lab run -X plr_v1.5_cot --dataset datasets/mine --model gemma
# ... dataset-push, submit --pull 동일 ...

# ③ 결과: 브라우저 http://<서버>:8890/d/mine (리더보드) · datasets/mine/pulled/*.html
```

---

## 4. GPU 없이 (mock 체험)

같은 lab 이미지로 `--gpus`·`HF_TOKEN` 없이 GPU-free 체험이 됩니다(mock은 GPU 미로드):

```bash
docker run --rm -v $PWD/datasets:/app/datasets plr-prompt-lab demo --keep
docker run --rm -e EVAL_SERVER_URL=http://<서버>:8890 -e EVAL_SERVER_TOKEN=sekrit \
  -v $PWD/datasets:/app/datasets plr-prompt-lab \
  run -X plr_v1.5_cot --dataset datasets/demo --model mock
```

---

## 5. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `could not select device driver ... gpu` | NVIDIA Container Toolkit 미설정 — §0의 `nvidia-ctk runtime configure` |
| `nvidia-smi` 컨테이너에서 안 보임 | 드라이버/toolkit 불일치 — 호스트 `nvidia-smi` 먼저 확인 |
| HF 다운로드 `401`/gated | `HF_TOKEN` 미지정/무효 — 유효 토큰 전달, 모델 접근 권한 확인 |
| 첫 run 매우 느림 | 모델 최초 다운로드(수 GB) — `plr_hf_cache` 볼륨에 캐시되어 다음부터 빠름 |
| `invalid or missing X-Auth-Token` | lab `EVAL_SERVER_TOKEN` ≠ 서버 토큰 — 동일하게 |
| `--server 또는 EVAL_SERVER_URL 필요` | lab에 `-e EVAL_SERVER_URL=...` 누락 |
| `⚠ hash unverified` | `run_provenance.json` 없이 제출 — `lab run`을 먼저(자동 생성) |
| 포트 8890 충돌 | 서버 `-p 8899:8890` 로 바꾸고 lab `EVAL_SERVER_URL`도 그 포트로 |
| 데이터셋이 컨테이너에 없음 | `-v $PWD/datasets:/app/datasets` 마운트 확인(상대경로는 datasets/… 로) |

---

## 6. 보안 노트

- lab 이미지: `.dockerignore`가 **`.git`(비밀 백업 브랜치)·`datasets/`(CCTV 크롭)·
  `models/`·`.venv`**를 제외 → 비밀·사적 데이터가 이미지에 굽히지 않음(검증됨).
- 데이터셋은 **볼륨 마운트로만** 컨테이너에 들어갑니다(이미지엔 없음).
- 두 이미지·서버 포트는 **사내 LAN 전용**. 레지스트리 푸시/공인망 노출 금지.
