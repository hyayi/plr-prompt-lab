# EXPERIMENT_SPEC.md — Experiment Matrix Runner

The `lab experiment run <experiment.yaml>` command enumerates the
**cross-product** of `datasets × models × prompts × pipelines × attributes`
and for each cell runs:

1. **run** — the pipeline's runner via the registry
2. **eval** — scores predictions and appends a ledger record

## Schema

```yaml
# Required axes
datasets:   [./datasets/gender_v1]        # one or more dataset directory paths
models:     [mock]                         # registry model names (mock | gemma)
prompts:    [plr_v1.4_cot, plr_v1.5_exp]  # version tags passed as --version to run_eval
pipelines:  [plr]                          # plr | search
attributes: [gender]                       # PLR attributes (plr pipeline only)

# Optional
ledger:     ./eval/ledger.jsonl            # ledger path (default: eval/ledger.jsonl)
formats:    [yaml]                         # IR_PLR_FORMAT axis (yaml | json), plr cells only
reasons:    ["on", "off"]                  # IR_PLR_REASON axis (on | off), plr cells only
```

### Field reference

| Field        | Type           | Required | Description |
|-------------|----------------|----------|-------------|
| `datasets`  | list[str]      | yes      | Paths to dataset directories. Each must contain `crops/`, `labels.jsonl`, and `predictions.jsonl` for plr; `queries.jsonl` and `attributes.jsonl` for search. |
| `models`    | list[str]      | yes      | Registry model names. `mock` is GPU-free; `gemma` requires weights. Unknown names raise an error before any cell runs. |
| `prompts`   | list[str]      | yes      | Prompt version tags. Passed as `--version` to `run_eval`. If the tag is `yaml` or `json` it also sets `IR_PLR_FORMAT`. |
| `pipelines` | list[str]      | yes      | `plr` (attribute extraction) or `search` (text retrieval). Unknown names raise an error before any cell runs. |
| `attributes`| list[str]      | no       | PLR attribute names (e.g. `gender`, `vehicle_type`, `military`). Required for the `plr` pipeline. Ignored for `search`. Default: `[""]`. |
| `ledger`    | str            | no       | Path to the ledger JSONL file. Relative paths are resolved relative to the experiment YAML file. Default: `eval/ledger.jsonl` inside the lab root. |
| `formats`   | list[str]      | no       | `IR_PLR_FORMAT` env axis; allowed values `yaml`, `json` (quote-free). Applies to **plr cells only**. Set per cell and restored after each cell. Default: env untouched. |
| `reasons`   | list[str]      | no       | `IR_PLR_REASON` env axis; allowed values `on`, `off` (**quote them** — bare YAML `on`/`off` parse as booleans). Applies to **plr cells only**. Default: env untouched. |

### format/reason axis semantics

- These are the *effective* runtime axes the constants prompt path switches on
  (`plr_prompts.py`: format picks the yaml/json template family, reason picks
  the CoT vs no-reason person template).
- **Ledger disambiguation**: two cells that differ only in an env axis would
  otherwise stamp the same `version` + `prompt_hash` (the hash covers files,
  not env). The runner therefore appends the axis values to the ledger version
  tag: `plr_v1.4_cot+json+reason-off`.
- **Conflict guard**: a yaml-backed prompt version pins its own wire format in
  the yaml's `format:` key, while the *response parser* follows
  `IR_PLR_FORMAT`. A conflicting combination (e.g. `prompts: [plr_v1.4_cot]`
  which pins yaml × `formats: [json]`) would make every parse fail, so the
  cell fails loudly with a clear error instead of producing garbage metrics.
  The `reasons` axis has no such conflict (the provider reads it from env).

## Cell enumeration

The cross-product is: `pipelines × datasets × models × prompts`.

For the `plr` pipeline, the `attributes` (and optional `formats` / `reasons`)
axes are added:
`pipelines × datasets × models × prompts × attributes × formats × reasons`.

For the `search` pipeline, the `attributes`/`formats`/`reasons` axes are not
used (search reads `queries.jsonl` from the dataset directory, and its
dictionary parse sends no PLR prompt).

## Dispatch via registry

- **plr cell** → `registry.get_model(model)` + `re_score.re_score(attribute, model, golden_dir=dataset)` → `eval/run_eval.py main()`
- **search cell** → `re_score.run_search_over_golden(queries_path, attributes_path)` → `run_search_eval.main()`

Both paths append a ledger record carrying:
`dataset`, `model`, `pipeline`, `prompt_hash` (from `provenance.prompt_hash`).

## Validation

Unknown `models` or `pipelines` values raise a `ValueError` with a clear
message listing available names **before any cell runs**.

## Fail-loud-but-continue

A cell that raises any exception is caught, logged with a per-cell error line,
recorded as `status=failed`, and the runner continues to the next cell.

After all cells run, a matrix summary is printed:

```
[experiment] === MATRIX SUMMARY ===
[experiment] total=4  ok=3  failed=1
[experiment] failed cells:
[experiment]   {dataset='./datasets/missing', model='mock', ...}
[experiment]     FileNotFoundError: Dataset directory not found: ./datasets/missing
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | All cells passed (or at least one passed without `--strict`) |
| 1    | `--strict` flag set and at least one cell failed |
| 2    | ALL cells failed |

## CLI usage

```bash
# Run the matrix
python3 lab.py experiment run examples/experiment.example.yaml

# Fail immediately on first failure (for CI)
python3 lab.py experiment run examples/experiment.example.yaml --strict

# GPU-free smoke test with mock model
python3 lab.py experiment run tests/fixtures/mock_experiment.yaml
```

## Ledger record shape

Each cell appends a record to the ledger.  For `plr` cells:

```json
{
  "attribute": "gender",
  "version": "plr_v1.4_cot",
  "date": "2026-07-02T12:00:00",
  "n": 5,
  "accuracy": 1.0,
  "recall": {"female": 1.0},
  "bias": null,
  "confusion": {"female": {"female": 5}},
  "seed_hash": "",
  "gemma_repo": "",
  "dataset": "./datasets/gender_v1",
  "model": "mock",
  "pipeline": "plr",
  "prompt_hash": "abc123"
}
```

For `search` cells:

```json
{
  "attribute": "search",
  "version": "plr_v1.4_cot",
  "date": "2026-07-02T12:00:00",
  "k": 5,
  "recall_at_k": 1.0,
  "precision_at_k": 0.2,
  "n_queries": 2,
  "seed_hash": "",
  "gemma_repo": "",
  "dataset": "./datasets/search_v1",
  "model": "mock",
  "pipeline": "search",
  "prompt_hash": "abc123"
}
```

## Example file

See `examples/experiment.example.yaml` for a fully-annotated example.
