"""Dataset — a selectable golden-set directory for the PLR lab.

A dataset directory is a self-describing bundle of the files the eval cycle
already uses, so the existing ``eval/golden/<attribute>/`` layout is itself a
valid dataset. That keeps every current command and test working unchanged
while letting callers point ``--dataset`` at an arbitrary path.

Layout (all optional except crops/ + labels for a real run)::

    <dataset>/
        crops/<obj_id>.jpg      # object crops (gitignored in the repo layout)
        labels.jsonl            # {"obj_id": ..., "true": ...}   human ground truth
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
