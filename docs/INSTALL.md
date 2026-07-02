# INSTALL — plr-prompt-lab on a fresh machine

This lab is a portable, standalone package. It has two paths:

- **GPU-free path** — imports, tests, and the `lab demo`-style mock/synthetic
  cycle. Needs only Python + the deps in `requirements.txt`.
- **Real-run path** — `lab run` re-scores crops with Gemma on a GPU.

---

## 1. Python environment

Python 3.10+ is required (the code uses `X | None` type syntax).

```bash
cd plr-prompt-lab
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Configure your environment (all vars are optional for the GPU-free path):

```bash
cp .env.example .env
# edit .env, then export the ones you need, e.g.:
#   export CORE_IR_PATH=/path/to/ziomilitary/core/ir
#   export RESULT_PATH=./results
```

### Verify the GPU-free path

```bash
python3 -c "import lab, dataset, re_score, config; print('portable import OK')"
python3 -m pytest tests/ -q      # expect: 21 passed
```

This works with **no GPU, no database, no Redis, no model download**. It is the
recommended onboarding step for a new machine or a new recipient of this package.

---

## 2. llama-cpp-python — GPU build (real runs only)

The `llama-cpp-python` wheel installed by `requirements.txt` is CPU-only. For
`lab run` you need a CUDA build:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

Requires a working CUDA toolkit and a matching NVIDIA driver. Pick the GPU with
`CUDA_VISIBLE_DEVICES` (see `.env.example`).

---

## 3. Model download (real runs only)

`lab run` loads a **Gemma-4-E4B GGUF** model into VRAM. Download it from Hugging
Face and point the env vars at it:

```bash
pip install huggingface_hub
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    --include "*Q4_0*.gguf" "*mmproj*" \
    --local-dir ./models/gemma-4-E4B-it-GGUF

export IR_GEMMA_REPO=unsloth/gemma-4-E4B-it-GGUF
# optional: pin the exact main file (else the Q4_0 file is auto-detected)
export IR_GEMMA_MAIN_FILE=gemma-4-E4B-it-Q4_0.gguf
```

`gemma_backend` also reads `IR_GEMMA_MAIN_FILE` / `IR_GEMMA_MMPROJ_FILE` /
`IR_GEMMA_N_CTX` / `IR_GEMMA_N_GPU_LAYERS` (see that module's header).

---

## 4. Real runs need a GPU

`lab run` requires a dedicated GPU (or an off-peak window if another service
holds the GPU). See the **Real-run preconditions** section in `README.md` — it
also needs a human-labeled golden set (`labels.jsonl`).

```bash
python3 lab.py run  --attribute gender --version plr_v1.4_cot
python3 lab.py eval --attribute gender --version plr_v1.4_cot
```

---

## 5. GPU-free onboarding (the fast path)

To try the full eval cycle without a GPU or real data, use the mock/synthetic
path exercised by the tests (see `tests/test_cycle_e2e.py`): it builds a
synthetic dataset dir, runs `re_score` with a mock model, and scores it — no
model download, no GPU, no DB. That is the quickest way to confirm a fresh
checkout is wired correctly before provisioning GPU + labels for a real run.
