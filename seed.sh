#!/usr/bin/env bash
# seed.sh — re-seed the plr-prompt-lab dev surface from a core/ir checkout.
#
# Usage:
#   ./seed.sh [/path/to/core/ir]
#
# Default source is the sibling deploy repo checkout. The script snapshot-copies
# (rsync) ONLY the pure dev surface — no storage / redis / scheduler / indexing /
# text_embed / main / backfill / image_retrieval / sr_policy. See SEED.md for the
# canonical file list and the source HEAD this lab was originally seeded from.
#
# Re-seeding is reproducible: run against a newer core/ir path to pull drift.
set -euo pipefail

SRC="${1:-/home/ziovision/ziomilitary/core/ir}"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$SRC" ]]; then
  echo "seed.sh: source core/ir not found: $SRC" >&2
  exit 1
fi

echo "seed.sh: source = $SRC"
echo "seed.sh: dest   = $DST"

# --- Single-file dev surface (pure: stdlib / PIL / numpy / yaml only) ---
# NOTE: config.py, registry.py and gemma_model.py are NOT re-seeded — both diverged from
# core/ir on purpose (lab-lean config; registry carries the lab's MODELS/
# PIPELINES section; gemma_model carries LabGemmaModel/MockModel). Sync them manually if the shared slot layer changes.
FILES=(
  plr_core.py
  plr_parse.py
  providers/file_prompt_provider.py
  preprocess.py
  gemma_backend.py
  plr_prompts.py
  plr_schema.py
  quality_gate.py
)

# --- Directory dev surface ---
# providers/ is NOT re-seeded as a directory: lab's providers/__init__.py is
# deliberately slimmer (PLR-only). Only the byte-synced provider file comes over.
# eval/ and parser/ are lab-owned / removed respectively.
DIRS=(
  prompts
  schema
)

RSYNC_EXCLUDES=(
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '.omc'
  --exclude '.pytest_cache'
  --exclude 'golden/*/labels.jsonl'
  --exclude 'golden/*/crops'
)

for f in "${FILES[@]}"; do
  if [[ ! -f "$SRC/$f" ]]; then
    echo "seed.sh: WARNING missing source file: $f" >&2
    continue
  fi
  rsync -a "$SRC/$f" "$DST/$f"
  echo "  copied $f"
done

for d in "${DIRS[@]}"; do
  if [[ ! -d "$SRC/$d" ]]; then
    echo "seed.sh: WARNING missing source dir: $d" >&2
    continue
  fi
  rsync -a "${RSYNC_EXCLUDES[@]}" "$SRC/$d/" "$DST/$d/"
  echo "  copied $d/"
done

echo "seed.sh: done. Verify with:"
echo "  python3 -c \"import plr_core, search_core, gemma_model, query_parser, scoring, quality_gate, plr_prompts, plr_schema; print('lab imports OK')\""
