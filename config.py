# config.py — lab configuration (lean).
#
# The lab keeps ONLY what its own code consumes: the provider version pins
# read by registry.py / providers/file_prompt_provider.py. The core/ir
# service settings that used to live here (REST API, Redis streams, eager
# indexing guardrails, ...) were removed 2026-07 — the lab never touches any
# of those services (import-purity contract, see README).
#
# Runtime env vars the lab DOES use live outside this module:
#   IR_PLR_FORMAT / IR_PLR_REASON   — prompt constants selector (plr_prompts.py)
#   IR_GEMMA_REPO / IR_GEMMA_*      — GPU model download (gemma_backend.py)
#   CORE_IR_PATH / RESULT_PATH / DATASET_DIR — paths (lab.py, dataset.py)
# See .env.example for the full list.
import os

# Provider version pins (pluggable-modules layer — see registry.py).
# Each var selects the active implementation for its slot; empty = first
# registered. IR_PARSER_VER / IR_SCORING_VER slots have no lab-side
# implementations (the search pipeline was removed 2026-07) — they are kept
# so registry.py stays structurally identical to core/ir\'s.
IR_MODEL_VER   = os.getenv("IR_MODEL_VER",   "")
IR_PROMPT_VER  = os.getenv("IR_PROMPT_VER",  "plr_v1.5_cot")
IR_PARSER_VER  = os.getenv("IR_PARSER_VER",  "")
IR_SCORING_VER = os.getenv("IR_SCORING_VER", "")
