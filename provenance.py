"""provenance — stable content hashes of the active prompt surface.

The "prompt surface" is the same true-surface set that ``lab port`` diffs
against core/ir: the prompt YAML files under ``prompts/`` plus ``plr_prompts.py``.
``prompt_hash()`` returns a short, stable sha256 prefix of that content so every
ledger writer (``eval/run_eval.py``, ``run_search_eval.py``) and any reporter
stamps the SAME hash for a given prompt state — letting a ledger record be tied
back to the exact prompt bytes that produced it.

GPU-free and dependency-light (stdlib only): safe to import from any runner.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parent

# Files that make up the active prompt surface, relative to the lab root.
# Mirrors lab.py's _PORT_FILES prompt entries (the true-surface diff set):
# every prompts/*.yaml plus plr_prompts.py.  plr_core.py is deliberately
# excluded — it is parsing/scoring glue, not prompt content.
_PROMPTS_DIR = "prompts"
_PROMPT_PY = "plr_prompts.py"


def _surface_paths(lab_root: str | os.PathLike[str] | None = None) -> list[Path]:
    """Return the sorted list of existing prompt-surface files.

    Sorted by relative path so the hash is deterministic regardless of the
    filesystem's directory-listing order.
    """
    root = Path(lab_root) if lab_root else _LAB_ROOT
    paths: list[Path] = []
    prompts_dir = root / _PROMPTS_DIR
    if prompts_dir.is_dir():
        paths.extend(sorted(prompts_dir.glob("*.yaml")))
    py_path = root / _PROMPT_PY
    if py_path.exists():
        paths.append(py_path)
    # Sort by path relative to root for a stable, location-independent order.
    return sorted(paths, key=lambda p: str(p.relative_to(root)))


def prompt_hash(
    lab_root: str | os.PathLike[str] | None = None,
    length: int = 12,
) -> str:
    """Return a short stable sha256 hash of the active prompt surface content.

    Hashes ``prompts/*.yaml`` + ``plr_prompts.py`` (the same true-surface files
    ``lab port`` diffs).  Each file contributes its relative path plus its raw
    bytes, so both a content edit and a rename change the hash.

    Args:
      lab_root: repo root to hash (default: this module's directory).
      length:   number of leading hex chars to return (default: 12).

    Returns:
      The first ``length`` hex characters of the sha256 digest, or a hash of
      the empty surface if no prompt files exist (still deterministic).
    """
    root = Path(lab_root) if lab_root else _LAB_ROOT
    h = hashlib.sha256()
    for path in _surface_paths(root):
        rel = str(path.relative_to(root))
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:length]
