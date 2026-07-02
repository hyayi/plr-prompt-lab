# plr-prompt-lab  (v2)

Standalone prompt/pipeline eval-cycle for PLR (Person-Level Recognition) **attribute
scoring (A)** and **text-search (B)**, lean-extracted from `core/ir` at HEAD
`c2fc1cf20a6fbd3ad4272aec8439d438a4febf34` (see [SEED.md](SEED.md)).

The lab can iterate on prompts, scoring logic, and the eval harness **without
a database, Redis, or a GPU** for the mock/synthetic path. The real Gemma
inference step (re-scoring crops) requires a dedicated GPU and human-labeled
crops — see [Real-run preconditions](#real-run-preconditions) below.

New in v2: selectable `--dataset` parameter, `validate-dataset` subcommand,
`lab demo` GPU-free onboarding, [DATASET_SPEC.md](DATASET_SPEC.md) format
spec, and [HANDOFF.md](HANDOFF.md) external prompt-engineer guide.

---

## What the lab is

The lab holds a **lean snapshot** of the PLR dev surface from `core/ir`:

- `plr_core.py`, `plr_prompts.py`, `plr_schema.py` — pure PLR inference core
- `gemma_model.py` — `Model` Protocol + `LabGemmaModel` (direct, no scheduler)
- `gemma_backend.py` — GPU GGUF loader (guarded; not imported unless `lab run`)
- `search_core.py`, `scoring.py`, `query_parser.py` — pure search/scoring
- `prompts/` — PLR prompt YAMLs (`plr_v0.4`, `plr_v1.3_cot`, `plr_v1.4_cot`)
- `eval/` — golden sets, runner scripts, ledger
- `demo.py` — self-contained MockModel + synthetic dataset for `lab demo`

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

## Parameter model

The lab's four selectable dimensions for an experiment:

| Dimension | How to select | Status |
|---|---|---|
| **Dataset** | `--dataset /path/to/dir` (default: `eval/golden/<attribute>/`) | Built |
| **Prompt** | Edit `prompts/*.yaml` and pass `--version <name>` to `lab run` + `lab eval` | Built |
| **Model** | Hard-wired to Gemma-4-E4B GGUF (real runs) or MockModel (demo/tests) | Roadmap |
| **Pipeline** | Hard-wired to `plr_core.run_plr` | Roadmap |

**Roadmap (Phase 2 — not yet built)**: a model registry (swap GGUF without
editing code), a pipeline registry (swap scoring/search logic), an experiment
matrix runner (all combinations), and an HTML report generator. These are
planned but not implemented; do not depend on them yet.

---

## The cycle

```
build-golden  ──►  label  ──►  run  ──►  eval  ──►  port
    │                               │          │
    │  (real data, operator step)   │          └──► ledger.jsonl delta
    │                               └──► re-runs Gemma on crops (GPU)
    └── <dataset>/crops/<obj_id>.jpg
```

### A — PLR attribute eval (gender, vehicle_type, military)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. build-golden  (operator step — real video + DB required)        │
│     lab build-golden --video <vd_id> --attribute gender             │
│     → <dataset>/crops/<obj_id>.jpg                                  │
│     → <dataset>/index_map.json                                      │
│     → <dataset>/predictions.jsonl  (bootstrap)                      │
│                                                                     │
│  2. label  (human step)                                             │
│     lab label --dataset <dataset> --female-in-male M3,M7           │
│     → <dataset>/labels.jsonl                                        │
│                                                                     │
│  3. run  (GPU step)                                                 │
│     lab run --attribute gender --version plr_v1.4_cot               │
│              --dataset <dataset>                                    │
│     → <dataset>/predictions.jsonl  (overwritten)                    │
│     → <dataset>/attributes.jsonl                                    │
│                                                                     │
│  4. eval                                                            │
│     lab eval --attribute gender --version plr_v1.4_cot              │
│              --dataset <dataset>                                    │
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
│              --dataset <dataset>                                    │
│     → <dataset>/queries.jsonl  (hand-authored)                      │
│     → <dataset>/attributes.jsonl  (from run)                        │
│     → <dataset>/search_results.jsonl  (from run)                    │
│                                                                     │
│  3. eval --mode search                                              │
│     lab eval --attribute search --mode search --version plr_v1.4_cot│
│              --dataset <dataset>                                    │
│     → prints recall@k / precision@k + Δ vs prior version            │
│     → appends record to eval/ledger.jsonl                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Commands

```bash
# GPU-free onboarding — see the full loop immediately, no data needed
python3 lab.py demo

# Build golden set (real data, operator step — see preconditions)
python3 lab.py build-golden --video <vd_...> --attribute gender [--dataset <dir>]

# Label (human step)
python3 lab.py label [--dataset <dir>] --female-in-male M3,M7 --male-in-female F2

# Re-score with Gemma (GPU step)
python3 lab.py run --attribute gender --version plr_v1.4_cot [--dataset <dir>]

# Evaluate PLR attribute
python3 lab.py eval --attribute gender --version plr_v1.4_cot [--dataset <dir>]

# Evaluate text-search (recall@k)
python3 lab.py eval --attribute search --mode search --version plr_v1.4_cot [--dataset <dir>]

# Validate a dataset directory
python3 lab.py validate-dataset --dataset <dir>

# Diff (or apply) lab prompt surface to core/ir
python3 lab.py port [--apply] [--core-ir /path/to/core/ir]

# Run all tests (no GPU, no DB)
python3 -m pytest tests/ -q
```

---

## `--dataset` parameter

Every command that reads or writes a golden set accepts `--dataset <dir>`.
When omitted, the command falls back to `eval/golden/<attribute>/` — the
same layout used before v2, so all existing workflows are unchanged.

A dataset directory must conform to [DATASET_SPEC.md](DATASET_SPEC.md).
Validate any new dataset before using it with `lab run`:

```bash
python3 lab.py validate-dataset --dataset /path/to/my_dataset/
```

The `prepare-dataset` skill (see `skills/`) automates the build-golden +
label steps into a single guided workflow.

---

## `lab demo` — GPU-free onboarding

```bash
python3 lab.py demo
```

Runs a complete mock eval cycle with **no GPU, no database, no model download**:

1. Builds a 5-crop synthetic dataset in `demo_dataset/` (tiny JPEGs + labels).
2. Calls `re_score()` twice with a built-in `MockModel` — v1 predicts female
   (accuracy 1.0), v2 predicts male (accuracy 0.0).
3. Runs `run_eval()` for each version and prints accuracy + Δ.
4. Prints a walkthrough of what happened and pointers to next steps.
5. Cleans up `demo_dataset/` on exit (pass `--keep` to retain it).

Use `lab demo` to verify a fresh install is wired correctly, or to demonstrate
the loop to a new team member before provisioning GPU + labels.

---

## `HANDOFF.md` — external prompt-engineer guide

[HANDOFF.md](HANDOFF.md) is the guide for a prompt engineer who is improving
PLR/text-search prompts without touching the production service. It covers:

- What to edit (`prompts/*.yaml`, and when also `plr_prompts.py`).
- What not to touch (inference core, storage — neither is present in the lab).
- The full iteration loop: prepare dataset → `lab run` → `lab eval` → read Δ →
  iterate → `lab port` → hand diff + winning YAML back to ZioVision.
- Real-run preconditions (GPU + model) vs the GPU-free `lab demo`.
- Hand-back mechanics: `lab port` produces a read-only diff; the external
  engineer sends the diff and winning prompt YAML to ZioVision, who apply it
  gated on re-eval inside `core/ir`.

---

## Cold-start preconditions

A clean checkout cannot run `eval` until all three conditions are met:

1. **Crops seeded** — `<dataset>/crops/<obj_id>.jpg` must exist for every
   `obj_id` in `predictions.jsonl`. Produced by `lab build-golden`. The
   `eval/golden/*/crops/` directories are gitignored — never committed.

2. **`labels.jsonl` produced** — `<dataset>/labels.jsonl` must exist.
   Produced by `lab label` after a human reviews the contact sheets. Until
   a human labels the set, `run_eval.py` raises `FileNotFoundError`.

3. **`ledger.jsonl` created on first eval** — `eval/ledger.jsonl` is created
   automatically on the first `lab eval` run. It does not need to pre-exist.

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
   `<dataset>/labels.jsonl` must exist and contain human-verified ground truth
   before `lab eval` produces a meaningful accuracy number. Run `lab label`
   with the corrected tile IDs after a human reviews the contact sheets.

See [INSTALL.md](INSTALL.md) for the full setup (Python env, CUDA build, model
download).

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
