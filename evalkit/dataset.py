"""Dataset — a selectable golden-set directory for the PLR lab.

A dataset directory is a self-describing bundle of the files the eval cycle
already uses, so the existing ``eval/golden/<attribute>/`` layout is itself a
valid dataset. That keeps every current command and test working unchanged
while letting callers point ``--dataset`` at an arbitrary path.

Layout (all optional except crops/ + labels for a real run)::

    <dataset>/
        crops/<obj_id>.jpg      # object crops (gitignored in the repo layout)
        labels.jsonl            # {"obj_id": ..., "label": ...}  human ground truth
        predictions.jsonl       # {"obj_id": ..., "pred": ...}   model output
        attributes.jsonl        # {"obj_id": ..., "plr_json": {...}}
        queries.jsonl           # {"query": ..., "relevant": [obj_id, ...]}   (search)
        manifest.yaml           # {attribute, n, created, source_note}

``Dataset(path)`` only wraps paths; it does not read the jsonl bodies (the
runners already do that). Accessors return ``pathlib.Path`` objects so callers
can pass ``str(ds.crops_dir)`` etc. to the existing runners.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Dataset:
    """A selectable golden-set directory.

    Args:
      path: dataset directory. The existing ``eval/golden/<attribute>/`` dirs
            are valid datasets.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    # -- path accessors --------------------------------------------------

    @property
    def crops_dir(self) -> Path:
        return self.path / "crops"

    @property
    def labels_path(self) -> Path:
        return self.path / "labels.jsonl"

    @property
    def predictions_path(self) -> Path:
        return self.path / "predictions.jsonl"

    @property
    def attributes_path(self) -> Path:
        return self.path / "attributes.jsonl"

    @property
    def queries_path(self) -> Path:
        return self.path / "queries.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.yaml"

    # -- content accessors -----------------------------------------------

    @property
    def manifest(self) -> dict[str, Any]:
        """Parsed manifest.yaml, or {} if absent.

        Fields (all optional): attribute, n, created, source_note.
        """
        if not self.manifest_path.exists():
            return {}
        import yaml

        with open(self.manifest_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    def obj_ids(self) -> list[str]:
        """obj_ids for this dataset, read from predictions.jsonl (the file that
        seeds the golden obj_id set — mirrors re_score's contract). Falls back
        to labels.jsonl if predictions.jsonl is absent. Returns [] if neither
        exists. Order is preserved from the file for deterministic output."""
        for p in (self.predictions_path, self.labels_path):
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    return [
                        json.loads(line)["obj_id"]
                        for line in f
                        if line.strip()
                    ]
        return []

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Dataset({str(self.path)!r})"


def resolve_dataset_dir(
    lab_root: str | os.PathLike[str],
    attribute: str,
    dataset: str | None = None,
) -> Path:
    """Resolve the dataset directory for a command.

    If ``dataset`` is given, use it verbatim. Otherwise fall back to the
    backward-compatible ``<lab_root>/eval/golden/<attribute>/`` layout (which is
    itself a valid dataset dir), so existing commands and tests keep working.
    """
    if dataset:
        return Path(dataset)
    return Path(lab_root) / "eval" / "golden" / attribute


# =====================================================================
# Attribute spec — manifest-driven labels (generic) with PLR presets
# =====================================================================
# The dataset template fixes the STRUCTURE (crops/ + labels.jsonl +
# manifest.yaml); the LABEL SET is the user's choice. A dataset declares its
# own attribute in manifest.yaml:
#
#   attribute: helmet
#   labels: [helmet, no_helmet]                  # allowed label values
#   pred_path: attributes.equipment[0].type      # where the pred lives in PLR JSON
#   margin_path: attributes.gender_scores.decision_margin   # optional
#   bias_pair: [no_helmet, helmet]               # optional headline bias
#   object_type_hint: person                     # optional (default person)
#
# The three original PLR attributes (gender / vehicle_type / military) are
# built-in PRESETS — they work without any manifest declaration and serve as
# the reference examples of the scheme.

PRESET_SPECS: dict[str, dict[str, Any]] = {
    "gender": {
        "labels": ("male", "female", "unknown"),
        "pred_path": "attributes.gender_scores.selected",
        "margin_path": "attributes.gender_scores.decision_margin",
        "bias_pair": ("female", "male"),
        "object_type_hint": "person",
    },
    "vehicle_type": {
        "labels": None,  # validate.py keeps its preset vocabulary
        "pred_path": "attributes.type_topk[0].label",
        "margin_path": None,
        "bias_pair": None,
        "object_type_hint": "vehicle",
    },
    "military": {
        "labels": ("military", "civilian", "unknown"),
        "pred_path": "attributes.military",
        "margin_path": None,
        "bias_pair": None,
        "object_type_hint": "person",
    },
}


def resolve_json_path(data: Any, dotted: str) -> Any:
    """Resolve a dotted path with optional [idx] segments against nested
    dicts/lists: "attributes.equipment[0].type" -> data["attributes"]
    ["equipment"][0]["type"]. Returns None on any missing step (defensive:
    model output may omit fields)."""
    cur = data
    for raw in dotted.split("."):
        seg = raw
        idxs: list[int] = []
        while seg.endswith("]") and "[" in seg:
            seg, _, tail = seg.rpartition("[")
            idxs.insert(0, int(tail[:-1]))
        if seg:
            if not isinstance(cur, dict) or seg not in cur:
                return None
            cur = cur[seg]
        for i in idxs:
            if not isinstance(cur, (list, tuple)) or i >= len(cur):
                return None
            cur = cur[i]
    return cur


def attribute_spec(dataset_dir: str | Path, attribute: str) -> dict[str, Any]:
    """Effective spec for one attribute: manifest declaration wins, preset
    fills the gaps, unknown attribute without a manifest declaration gets an
    empty spec (callers fail loudly where a field is required)."""
    base = dict(PRESET_SPECS.get(attribute) or {
        "labels": None, "pred_path": None, "margin_path": None,
        "bias_pair": None, "object_type_hint": "person",
    })
    try:
        manifest = Dataset(Path(dataset_dir)).manifest
    except Exception:  # noqa: BLE001 — malformed manifest is validate's job
        manifest = {}
    if manifest.get("attribute") == attribute or "attribute" not in manifest:
        for key in ("labels", "pred_path", "margin_path", "bias_pair", "object_type_hint"):
            if manifest.get(key) is not None:
                base[key] = manifest[key]
    return base
