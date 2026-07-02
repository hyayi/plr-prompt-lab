# PLR Prompt Lab — External Prompt-Engineer Handoff Guide

This guide is for a **prompt engineer improving PLR prompts** without
touching the ZioVision inference service directly. (The lab is PLR-only:
its text-search pipeline was removed 2026-07.)

---

## Who this is for

You are a prompt engineer who has received the `plr-prompt-lab` package. Your
job is to iterate on the prompts that drive PLR (Person-Level Recognition)
attribute scoring, measure the effect of your
changes on a labeled golden dataset, and hand back a diff + winning YAML to the
ZioVision team. You do **not** have access to the production database, Redis, or
the live GPU service — and you do not need them.

---

## What you edit

### Primary (actual runtime source): `prompts/<current-version>.yaml`

Since 2026-07 the live templates are LOADED from
`prompts/<PROMPT_VERSION_YAML_COT>.yaml` (single source — the historical
constants were removed after byte-equality verification). Editing the
current version's yaml IS editing the runtime prompt. Two env vars still
switch variants:

```
IR_PLR_FORMAT = yaml | json     # yaml is current; json is a legacy A/B path
IR_PLR_REASON = on   | off      # on = chain-of-thought (longer, ~+35% tokens)
```

Both env vars are also available as experiment-matrix axes (`formats:` /
`reasons:` in experiment.yaml — see EXPERIMENT_SPEC.md), so you can sweep them
without touching your shell environment.

What to edit (to actually change behavior): the yaml blocks of the current
version file —

```
prompts/plr_v1.5_cot.yaml
  plr.system                 # role + output discipline
  plr.person_user            # CoT person (IR_PLR_REASON=on, production)
  plr.person_user_no_reason  # plain person
  plr.vehicle_user           # vehicle
  query_parser.*             # search prompt (core/ir side uses it; keep intact)
```
(The only remaining constants in plr_prompts.py are the legacy JSON path.)

### Keep `prompts/*.yaml` in parity (declarative mirror — not read at runtime)

```
prompts/
    plr_v0.4.yaml         # early baseline (reference only)
    plr_v1.3_cot.yaml     # v1.3 chain-of-thought
    plr_v1.4_cot.yaml     # pre-forced-commit (A/B baseline)
    plr_v1.5_cot.yaml     # current — forced-commit (no unknown)
```

These per-version yaml files are a human-readable MIRROR of the constants above,
used for versioned diffing and `lab port` — **not loaded on the runtime path**.
Keep them identical to the constants: core/ir enforces this with
`tests/test_prompt_source_parity.py`; the lab checks it via `lab port` diff.

**Two edit paths:**
- To iterate a **specific version** for experiments: edit that `prompts/<version>.yaml`.
  `lab run --version <version>` loads it via `FilePromptProvider`, so the change
  takes effect for that version (this is how the experiment prompt-axis compares
  real prompt variants in one checkout).
- To change the **default / current** prompt (what runs without `--version`, and
  what `lab port` ships to core/ir): edit the `plr_prompts.py` constants **and**
  keep `prompts/plr_v1.5_cot.yaml` in parity with them.

Also update `parse_plr_response()` in `plr_prompts.py` if you change the output
schema (add/rename a field).

### What NOT to touch

| File / area | Reason |
|---|---|
| `re_score.py` | Core re-scoring runner — edit prompts, not the runner. |
| `plr_core.py` | PLR inference core — edit prompts, not the inference logic. |
| `plr_schema.py` | PLR output schema — only change if the schema itself changes. |
| `gemma_model.py`, `gemma_backend.py` | GPU model loader — out of scope for prompt work. |
| `eval/` runner scripts | Eval harness — edit to fix bugs only, not to inflate scores. |
| `core/ir/` | Production service — never edit directly. Hand diffs back to ZioVision. |

---

## The iteration loop

```
prepare dataset  →  lab run  →  lab eval  →  read Δ  →  edit prompt  →  repeat
                                                             ↓ (when improved)
                                                          lab port  →  hand diff to ZioVision
```

### 1. Prepare / point at a dataset

A dataset is a directory containing labeled crops. See [DATASET_SPEC.md](DATASET_SPEC.md)
for the exact layout and file schemas.

**Receiving an existing dataset**: if ZioVision hands you a dataset tarball,
unpack it and validate:

```bash
python3 lab.py validate-dataset --dataset /path/to/my_dataset/
```

**Preparing a fresh dataset** (requires a real video + DB — operator step):
use the `prepare-dataset` skill documented in the lab's skills directory or run
`lab build-golden` + `lab label`. This is a ZioVision operator step; as an
external prompt engineer you will normally receive a pre-built, labeled dataset.

### 2. Run re-scoring (GPU required for real data)

```bash
python3 lab.py run --attribute gender --version plr_v1.5_cot --dataset /path/to/my_dataset/
```

This calls Gemma on every crop in the dataset using the prompt version you name.
It writes `predictions.jsonl` and `attributes.jsonl` into the dataset directory.

**Preconditions for a real run** (see [INSTALL.md](INSTALL.md)):
- A dedicated GPU with no other service holding VRAM (stop the live `ir`
  container first — see INSTALL.md section 4).
- The Gemma-4-E4B GGUF model downloaded and env vars set.
- A human-labeled `labels.jsonl` in the dataset (see DATASET_SPEC.md).

**GPU-free trial**: run `python3 lab.py demo` instead — it runs a complete mock
cycle with no GPU, no DB, and no real data so you can see the loop immediately.

### 3. Evaluate

```bash
# Attribute accuracy (gender / vehicle_type / military)
python3 lab.py eval --attribute gender --version plr_v1.5_cot --dataset /path/to/my_dataset/
```

### 4. Read accuracy, bias, recall, and ledger Δ

`lab eval` prints to stdout:

```
=== gender eval: plr_v1.5_cot (n=150) ===
accuracy: 0.927 (139/150)   Δ vs plr_v1.4_cot: +0.014 (0.913 → 0.927)
bias female->male: 0.067 (5/75)   Δ: -0.027
recall: female=0.933, male=0.920
confusion (rows=true, cols=pred):
          female      male   unknown
  female      70         5         0
    male       6      67         2
 unknown       0         1         0
```

Key numbers to track:

| Metric | What it means |
|---|---|
| `accuracy` | Overall fraction correct |
| `bias female->male` | Rate at which females are predicted male (lower = better for this pair) |
| `recall` | Per-class recall — catch rate for each label |
| `Δ vs <prior>` | Change vs the most recent other version in the ledger |

The ledger record is appended to `eval/ledger.jsonl` (or the dataset's own
ledger if you pass `--ledger`). Every run is recorded — you can always diff
any two versions.

### 5. Iterate

Edit `prompts/plr_v1.5_cot.yaml` (or whichever version you are improving),
go back to step 2. The ledger Δ tells you whether the change helped.

**Rules**:
- Do not change prompts without measuring. A prompt that reads better but
  scores worse is not an improvement.
- Do not ship unlabeled datasets. `lab eval` on an unlabeled dataset produces
  meaningless numbers.
- Do not commit to production (`core/ir`) yourself. See hand-back mechanics below.

### 6. Hand back to ZioVision (lab port)

When a new version improves accuracy/recall over the baseline:

```bash
python3 lab.py port [--core-ir /path/to/ziomilitary/core/ir]
```

This prints a unified diff of every file in the prompt surface
(`prompts/*.yaml`, `plr_prompts.py`, `plr_core.py`) between the lab and
`core/ir`. It does **not** write to `core/ir` (read-only by default).

Send the ZioVision team:

1. The diff printed by `lab port` (copy-paste or redirect to a file).
2. The winning `prompts/<version>.yaml` file.
3. The ledger record(s) showing the before/after Δ — excerpt from
   `eval/ledger.jsonl` or the eval stdout.

The ZioVision team applies the diff to `core/ir`, runs the parity test
(`tests/test_prompt_source_parity.py`), and gates the change on a re-eval
inside the service before deploying. **Do not apply the diff to `core/ir`
yourself.**

---

## GPU-free trial: `lab demo`

Before you have GPU access or a real dataset, run:

```bash
python3 lab.py demo
```

This builds a 5-crop synthetic dataset, runs two mock-model versions (v1 =
all-female predictions, v2 = all-male predictions), evaluates both against
ground truth, and prints the accuracy + Δ. Exit code 0, no GPU, no DB.

Use `lab demo` to:
- Confirm the lab is installed correctly on a new machine.
- Understand the cycle structure before working with real data.
- Demonstrate the loop to a new team member.

---

## Quick reference

```bash
# GPU-free onboarding
python3 lab.py demo

# Validate a received dataset
python3 lab.py validate-dataset --dataset /path/to/dataset/

# Re-score (GPU + model required)
python3 lab.py run  --attribute gender --version plr_v1.5_cot --dataset /path/to/dataset/

# Evaluate attribute accuracy
python3 lab.py eval --attribute gender --version plr_v1.5_cot --dataset /path/to/dataset/


# Show diff to hand back
python3 lab.py port

# Run all tests (no GPU)
python3 -m pytest tests/ -q
```

---

## References

- [DATASET_SPEC.md](DATASET_SPEC.md) — full dataset directory format and file schemas
- [INSTALL.md](INSTALL.md) — Python env setup, GPU build, model download
- [SEED.md](SEED.md) — which `core/ir` commit this lab was extracted from
- `prompts/plr_v1.4_cot.yaml` — current production prompt (start here when iterating)
- `eval/ledger.jsonl` — historical accuracy/recall records across all versions

---

## Don'ts (summary)

- **Don't change prompts without measuring.** Always run `lab run` + `lab eval`
  before and after any edit; guessing is not a workflow.
- **Don't ship unlabeled datasets.** `labels.jsonl` must exist and be
  human-verified before `lab eval` means anything.
- **Don't edit `core/ir` directly.** Hand the diff back via `lab port`;
  let ZioVision apply it after their own re-eval.
- **Don't run `lab run` while the live `ir` container is up** — you will OOM
  or contend for the GPU. Stop the container first (see INSTALL.md).
- **Don't commit crops or labels to git.** The `eval/golden/*/crops/` directories
  are gitignored; keep them local to the machine that ran `lab build-golden`.
