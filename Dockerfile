# plr-prompt-lab — GPU 실측 이미지 (lab run: Gemma-4-E4B GGUF on CUDA)
#
# 빌드:   docker build -t plr-prompt-lab .
# 런타임에 정하는 것(3가지 + 볼륨):
#   -e EVAL_SERVER_URL=http://<서버>:8890   평가 서버 주소
#   -e EVAL_SERVER_TOKEN=<토큰>             서버 변이 토큰
#   -e HF_TOKEN=<hf_...>                    HuggingFace 토큰(Gemma 다운로드용)
#   -v $PWD/datasets:/app/datasets          데이터셋(크롭+라벨) — 이미지에 안 굽고 마운트
#   -v hf_cache:/hf-cache                   모델 캐시(첫 run 때 자동 다운로드, 이후 재사용)
#   --gpus all                              GPU 노출
#
# 모델은 이미지에 넣지 않는다: 첫 `lab run` 때 huggingface_hub 가 IR_GEMMA_REPO 를
# HF 캐시 볼륨으로 자동 다운로드한다(HF_TOKEN 으로 인증). 이미지는 lean 유지.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# python + llama-cpp-python CUDA 빌드에 필요한 툴체인
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential cmake ninja-build libgomp1 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) 의존성 먼저(레이어 캐시). requirements 의 llama-cpp-python 은 CPU 휠이라,
#    이어서 CUDA 빌드로 강제 재설치한다(INSTALL.md 의 CMAKE_ARGS 레시피).
COPY requirements.txt .
RUN pip install -r requirements.txt huggingface_hub \
    && CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python

# 2) lab 소스 (.dockerignore 로 .git/datasets/models 제외 — 비밀·크롭·가중치 미포함)
COPY . /app

# 모델 자동 다운로드 기본값 + HF 캐시 위치. HF_TOKEN 은 런타임 -e 로 주입.
ENV IR_GEMMA_REPO=unsloth/gemma-4-E4B-it-GGUF \
    HF_HOME=/hf-cache
VOLUME ["/app/datasets", "/hf-cache"]

# lab 은 서비스가 아니라 CLI — entrypoint 를 lab.py 로.
#   예) docker run --gpus all -e ... -v ... plr-prompt-lab \
#         run -X plr_v1.5_cot --dataset datasets/mine --model gemma
#       docker run ... plr-prompt-lab submit --dataset mine --run-dir datasets/mine -X plr_v1.5_cot --pull
ENTRYPOINT ["python3", "lab.py"]
CMD ["--help"]
