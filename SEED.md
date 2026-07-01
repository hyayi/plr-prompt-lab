# plr-prompt-lab — SEED provenance

This is a **standalone** repo (not a submodule of the ziomilitary deploy repo,
not inside `core/ir`). It holds a **lean, DB-free snapshot** of the PLR
prompt/scoring dev surface from `core/ir`, so prompt/scoring iteration can run
without storage / redis / scheduler / indexing.

## Source

- Source repo: `ziomilitary/core/ir`
- Source `core/ir HEAD`: `1690f257b2a4a463322c7fe441e3f15a7501459b`
  (the S0/S1 shared-core refactor `refactor(ir): extract shared pure cores`; the lab
  was seeded from this working tree — originally `c2fc1cf` + the run_plr/run_search/
  Model extraction, now committed as this hash so a re-seed reproduces the surface.)
- Seeded on: **2026-07-01**
- Re-seed with: `./seed.sh /path/to/core/ir` (see `seed.sh`)

## Copied dev surface

Snapshot-copied FILES (this is a separate repo, not a live checkout):

**Single files (pure: stdlib / PIL / numpy / yaml only):**
- `plr_core.py` — pure single-view PLR core; canonical home of the moved
  `_draw_target_marker`, `_top_color`, `_attach_military_flags` helpers
  (+ `_MILITARY_COLOR`), moved out of `indexing.py` in Part A.
- `search_core.py`
- `gemma_model.py` — carries the `Model` Protocol, the retained-but-unused
  `SchedulerGemmaModel`, and the lab's direct `LabGemmaModel` (calls
  `gemma_backend.load_backend().generate(...)` with no scheduler).
- `gemma_backend.py` — GPU model loader (llama.cpp GGUF). Not copied from the
  spec's core list per se, but required by the lab; its provider/registry
  registration is guarded by `try/except ImportError` so import stays clean.
- `plr_prompts.py`
- `plr_schema.py`
- `quality_gate.py`
- `query_parser.py`
- `query_normalizer.py` — lazy dep of `query_parser`.
- `scoring.py` — verified NOT importing storage.
- `config.py` — env-driven config; lazy dep of `providers` / `registry`.
- `registry.py` — provider registry; lazy dep of `gemma_backend` / `providers`.

**Directories:**
- `prompts/` — PLR prompt YAMLs (`plr_v0.4`, `plr_v1.3_cot`, `plr_v1.4_cot`).
- `providers/` — provider ABCs + file prompt provider + bootstrap.
- `parser/` — YAML query parser + `qp_v0.4.yaml`.
- `eval/` — offline eval harness (golden sets, run scripts). Per-run
  `labels.jsonl` / `crops/` are gitignored.

## Moved symbols (Part A, now canonical in `plr_core.py`)

- `_draw_target_marker(pil, ...)` — yellow corner-bracket target marker (PIL).
- `_top_color(color_topk)` — highest-score color label helper.
- `_attach_military_flags(plr_json)` — in-place `is_military` / `is_soldier` /
  `primary_color` enrichment.
- `_MILITARY_COLOR = "military_olive"` constant.

In `core/ir`, `indexing.py` now imports these from `plr_core` (single source of
truth) instead of defining them, so `import plr_core` never pulls
storage/psycopg2/redis.

## NOT copied (service / DB / redis / embedding)

`indexing.py`, `image_retrieval.py`, `storage.py`, `redis_handler.py`,
`main.py`, `backfill.py`, `text_embed.py`, `scheduler.py`, `sr_policy.py`.

## Import-closure / purity contract

```
python3 -c "import plr_core, search_core, gemma_model, query_parser, scoring, quality_gate, plr_prompts, plr_schema; print('lab imports OK')"
```

None of `storage`, `psycopg2`, `redis`, `text_embed`, `scheduler`, `indexing`,
`main`, `backfill`, `image_retrieval`, `sr_policy` may appear in `sys.modules`
after those imports.
