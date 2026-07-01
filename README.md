# plr-prompt-lab

Standalone prompt/pipeline eval-cycle for PLR (Person-Level Recognition) **attribute
scoring (A)** and **text-search (B)**, lean-extracted from `core/ir` at HEAD
`c2fc1cf20a6fbd3ad4272aec8439d438a4febf34` (see [SEED.md](SEED.md)).

The lab can iterate on prompts, scoring logic, and the eval harness **without
a database, Redis, or a GPU** for the mock/synthetic path. The real Gemma
inference step (re-scoring crops) requires a dedicated GPU and human-labeled
crops — see [Real-run preconditions](#real-run-preconditions) below.

---

## What the lab is

The lab holds a **lean snapshot** of the PLR dev surface from `core/ir`:

- `plr_core.py`, `plr_prompts.py`, `plr_schema.py` — pure PLR inference core
- `gemma_model.py` — `Model` Protocol + `LabGemmaModel` (direct, no scheduler)
- `gemma_backend.py` — GPU GGUF loader (guarded; not imported unless `lab run`)
- `search_core.py`, `scoring.py`, `query_parser.py` — pure search/scoring
- `prompts/` — PLR prompt YAMLs (`plr_v0.4`, `plr_v1.3_cot`, `plr_v1.4_cot`)
- `eval/` — golden sets, runner scripts, ledger

**Not copied** (service/DB/redis/embedding layer): `storage.py`, `redis_handler.py`,
`indexing.py`, `main.py`, `scheduler.py`, `text_embed.py`, `backfill.py`, etc.

### Import purity contract

```bash
python3 -c "import plr_core, search_core, gemma_model, query_parser, scoring, \
    quality_gate, plr_prompts, plr_schema; print('lab imports OK')"
```

None of `storage`, `psycopg2`, `redis` may appear in `sys.modules` after those
imports.

---

## The cycle

```
build-golden  ──►  label  ──►  run  ──►  eval  ──►  port
    │                               │          │
    │  (real data, operator step)   │          └──► ledger.jsonl delta
    │                               └──► re-runs Gemma on crops (GPU)
    └── eval/golden/<attr>/crops/<obj_id>.jpg
```

### A — PLR attribute eval (gender, vehicle_type, military)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. build-golden  (operator step — real video + DB required)        │
│     lab build-golden --video <vd_id> --attribute gender             │
│     → eval/golden/gender/crops/<obj_id>.jpg                         │
│     → eval/golden/gender/index_map.json                             │
│     → eval/golden/gender/predictions.jsonl  (bootstrap)             │
│                                                                     │
│  2. label  (human step)                                             │
│     lab label --female-in-male M3,M7 --male-in-female F2           │
│     → eval/golden/gender/labels.jsonl                               │
│                                                                     │
│  3. run  (GPU step)                                                 │
│     lab run --attribute gender --version plr_v1.4_cot               │
│     → eval/golden/gender/predictions.jsonl  (overwritten)           │
│     → eval/golden/gender/attributes.jsonl                           │
│                                                                     │
│  4. eval                                                            │
│     lab eval --attribute gender --version plr_v1.4_cot              │
│     → prints accuracy/confusion/bias + Δ vs prior version           │
│     → appends record to eval/ledger.jsonl                           │
│                                                                     │
│  5. port  (read-only diff / --apply to push changes to core/ir)     │
│     lab port [--apply] [--core-ir /path/to/core/ir]                 │
└─────────────────────────────────────────────────────────────────────┘
```

### B — Text-search eval (recall@k)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. (build-golden / label as above — same crops)                    │
│                                                                     │
│  2. run  (also writes attributes.jsonl)                             │
│     lab run --attribute gender --version plr_v1.4_cot               │
│     → eval/golden/search/queries.jsonl  (hand-authored)             │
│     → eval/golden/search/attributes.jsonl  (from run)               │
│     → eval/golden/search/search_results.jsonl  (from run)           │
│                                                                     │
│  3. eval --mode search                                              │
│     lab eval --attribute search --mode search --version plr_v1.4_cot│
│     → prints recall@k / precision@k + Δ vs prior version            │
│     → appends record to eval/ledger.jsonl                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Commands

```bash
# Build golden set (real data, operator step — see preconditions)
python3 lab.py build-golden --video <vd_...> --attribute gender

# Label (human)
python3 lab.py label --female-in-male M3,M7 --male-in-female F2

# Re-score with Gemma (GPU step)
python3 lab.py run --attribute gender --version plr_v1.4_cot

# Evaluate PLR attribute
python3 lab.py eval --attribute gender --version plr_v1.4_cot

# Evaluate text-search (recall@k)
python3 lab.py eval --attribute search --mode search --version plr_v1.4_cot

# Diff (or apply) lab prompt surface to core/ir
python3 lab.py port [--apply] [--core-ir /path/to/core/ir]

# Run all tests (no GPU, no DB)
python3 -m pytest tests/ -q
```

---

## Cold-start preconditions

A clean checkout cannot run `eval` until all three conditions are met:

1. **Crops seeded** — `eval/golden/<attr>/crops/<obj_id>.jpg` must exist for
   every `obj_id` in `predictions.jsonl`. Produced by `lab build-golden` (which
   pulls crops from the indexed video's objects dir in `RESULT_PATH`). The
   `eval/golden/*/crops/` directories are gitignored — they are never committed.

2. **`labels.jsonl` produced** — `eval/golden/<attr>/labels.jsonl` must exist.
   Produced by `lab label` after a human reviews the contact sheets and names
   the misclassified tiles. Until a human labels the set, this file does not
   exist and `run_eval.py` will raise `FileNotFoundError`.

3. **`ledger.jsonl` created on first eval** — `eval/ledger.jsonl` is created
   automatically on the first `lab eval` run (it is appended, not read, on the
   first run). It does not need to pre-exist; the file is created on first write.

---

## Real-run preconditions

`lab run` loads `LabGemmaModel` which calls `gemma_backend.load_backend()`,
which downloads and loads a 4B GGUF model into VRAM.

**Two explicit blockers apply before a real measurement is possible:**

1. **Dedicated GPU or off-peak window required.**
   The live `ir` service running in `engine/` already holds the GPU and most
   of the VRAM (it keeps `GemmaBackend` loaded for indexing/search). Running
   `lab run` concurrently will either OOM or contend for the GPU. The operator
   must either stop the live `ir` container or schedule the lab run during an
   off-peak maintenance window:

   ```bash
   # Stop ir before running lab
   cd /home/ziovision/ziomilitary/engine && docker-compose stop imageretrieval
   python3 lab.py run --attribute gender --version plr_v1.4_cot
   # Restart ir when done
   docker-compose start imageretrieval
   ```

2. **Human-labeled golden set required.**
   `eval/golden/gender/labels.jsonl` (and equivalent for other attributes) must
   exist and contain human-verified ground truth before `lab eval` produces a
   meaningful accuracy number. The gender golden set currently has **no human
   labels** — all predictions are model-bootstrapped. A labeler must review the
   contact sheets (`gender_MALE.png`, `gender_FEMALE.png`) and run `lab label`
   with the corrected tile IDs before the first real measurement.

These are **operator steps** — the lab CLI is wired and the code path is
complete; only execution awaits GPU availability and human labels.

---

## Metrics and ledger

### A — PLR attribute metrics (`eval/ledger.jsonl` record fields)

| Field | Description |
|---|---|
| `attribute` | `"gender"` / `"vehicle_type"` / `"military"` |
| `version` | PLR prompt version string (e.g. `"plr_v1.4_cot"`) |
| `date` | ISO-8601 timestamp |
| `n` | Number of labeled objects scored |
| `accuracy` | Overall accuracy (correct / n) |
| `recall` | Per-class recall dict |
| `bias` | Per-attribute bias metric (e.g. `female→male` misclassification rate) |
| `confusion` | Full confusion matrix (rows=true, cols=pred) |
| `seed_hash` | `core/ir HEAD` at seed time (from `SEED.md`) |
| `gemma_repo` | `IR_GEMMA_REPO` env at run time |

`run_eval.py` diffs the current run against the **most recent prior version**
in the ledger and prints `Δ accuracy / Δ bias`.

### B — Text-search metrics (`eval/ledger.jsonl` record fields)

| Field | Description |
|---|---|
| `attribute` | `"search"` |
| `version` | PLR prompt version string |
| `k` | Rank cutoff |
| `recall_at_k` | Mean recall@k across all queries |
| `precision_at_k` | Mean precision@k across all queries |
| `n_queries` | Number of evaluated queries |
| `seed_hash` | `core/ir HEAD` at seed time |
| `gemma_repo` | `IR_GEMMA_REPO` env at run time |

`run_search_eval.py` diffs against the most recent prior version in the ledger
and prints `Δ recall@k / Δ precision@k`.

Both A and B records land in the **same** `eval/ledger.jsonl` file, keyed by
`(attribute, version)`.

---

## Re-seeding from core/ir

```bash
./seed.sh /path/to/ziomilitary/core/ir
```

This re-copies the lab source files from `core/ir` HEAD and updates `SEED.md`.
Run this after significant changes to the upstream PLR/search surface to keep
the lab in sync.
