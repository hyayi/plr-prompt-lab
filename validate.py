"""Dataset validator for the PLR prompt lab.

Checks a dataset directory against the contract documented in DATASET_SPEC.md.
Each check prints a human-readable PASS/WARN/FAIL line. Exits non-zero on any
error (FAIL). Warnings do not block exit 0 unless --strict is requested.

Usage (programmatic):
    from validate import validate_dataset
    ok = validate_dataset("/path/to/dataset")   # returns True iff no errors

Usage (CLI entry-point — wired via lab.py validate-dataset):
    python3 lab.py validate-dataset <path>
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plr_schema import (
    GENDER_ENUM,
    MILITARY_ENUM,
    VEHICLE_TYPE_ENUM,
)

# ---------------------------------------------------------------------------
# Allowed label vocabularies per attribute.
# These are the values that validate_dataset accepts without error.
# ---------------------------------------------------------------------------

_ATTR_VOCAB: dict[str, frozenset[str]] = {
    "gender": frozenset((*GENDER_ENUM, "unknown")),
    "vehicle_type": frozenset(VEHICLE_TYPE_ENUM),
    "military": frozenset(MILITARY_ENUM),
}

# Required fields in manifest.yaml
_MANIFEST_REQUIRED = ("attribute", "n", "created", "source_note")

# Required fields per labels.jsonl line
_LABELS_REQUIRED = ("obj_id", "label")

# Required fields per queries.jsonl line
_QUERIES_REQUIRED = ("query", "relevant")


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------


@dataclass
class _Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        print(f"  FAIL  {msg}")
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        print(f"  WARN  {msg}")
        self.warnings.append(msg)

    def ok(self, msg: str) -> None:
        print(f"  PASS  {msg}")

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_manifest(ds_path: Path, report: _Report) -> dict[str, Any]:
    """Check manifest.yaml: present, parses, has required fields.

    Returns parsed manifest dict (or {} on failure) so later checks can
    read the attribute field.
    """
    mpath = ds_path / "manifest.yaml"
    if not mpath.exists():
        report.error("manifest.yaml not found")
        return {}

    try:
        import yaml  # type: ignore[import]
        with open(mpath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            report.error("manifest.yaml does not parse to a YAML mapping")
            return {}
    except Exception as exc:  # noqa: BLE001
        report.error(f"manifest.yaml parse error: {exc}")
        return {}

    missing = [k for k in _MANIFEST_REQUIRED if k not in data]
    if missing:
        report.error(f"manifest.yaml missing required fields: {missing}")
    else:
        report.ok(
            f"manifest.yaml valid (attribute={data.get('attribute')!r}, n={data.get('n')})"
        )
    return data


def _check_labels(
    ds_path: Path,
    attribute: str | None,
    report: _Report,
) -> set[str]:
    """Check labels.jsonl: present, each line valid JSON with required fields,
    labels in allowed vocabulary.

    Returns the set of obj_ids found in labels.jsonl (empty set on absence/error).
    """
    lpath = ds_path / "labels.jsonl"
    if not lpath.exists():
        report.error("labels.jsonl not found")
        return set()

    vocab: frozenset[str] | None = _ATTR_VOCAB.get(attribute or "") if attribute else None
    obj_ids: set[str] = set()
    line_errors = 0

    with open(lpath, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                report.error(f"labels.jsonl line {lineno}: JSON parse error: {exc}")
                line_errors += 1
                continue

            missing = [k for k in _LABELS_REQUIRED if k not in rec]
            if missing:
                report.error(
                    f"labels.jsonl line {lineno}: missing required fields {missing}"
                )
                line_errors += 1
                continue

            obj_ids.add(rec["obj_id"])
            label = rec["label"]

            if vocab is not None and label not in vocab:
                # Out-of-vocabulary label: error (malformed) rather than warn,
                # because it will silently break eval scoring.
                report.error(
                    f"labels.jsonl line {lineno}: obj_id={rec['obj_id']!r} "
                    f"label={label!r} not in vocab for attribute={attribute!r} "
                    f"(allowed: {sorted(vocab)})"
                )
                line_errors += 1

    if line_errors == 0:
        report.ok(f"labels.jsonl: {len(obj_ids)} records, all valid")

    return obj_ids


def _check_crops(
    ds_path: Path,
    labeled_ids: set[str],
    report: _Report,
) -> set[str]:
    """Check crops/ directory: present, every labeled obj_id has a .jpg,
    report crops-without-labels (warn) and labels-without-crops (error).

    Returns the set of obj_ids found as crop files.
    """
    crops_dir = ds_path / "crops"
    if not crops_dir.exists():
        report.error("crops/ directory not found")
        return set()

    crop_ids: set[str] = {p.stem for p in crops_dir.glob("*.jpg")}

    labels_without_crops = labeled_ids - crop_ids
    crops_without_labels = crop_ids - labeled_ids

    if labels_without_crops:
        report.error(
            f"labels-without-crops ({len(labels_without_crops)} obj_ids): "
            f"{sorted(labels_without_crops)}"
        )
    if crops_without_labels:
        report.warn(
            f"crops-without-labels ({len(crops_without_labels)} obj_ids): "
            f"{sorted(crops_without_labels)} — no label assigned (not an error)"
        )

    if not labels_without_crops:
        report.ok(
            f"crops/: {len(crop_ids)} crop(s), all labeled obj_ids have a crop"
        )

    return crop_ids


def _check_queries(
    ds_path: Path,
    all_obj_ids: set[str],
    report: _Report,
) -> None:
    """Check queries.jsonl (optional): each line valid, relevant obj_ids exist."""
    qpath = ds_path / "queries.jsonl"
    if not qpath.exists():
        # Optional file — not an error.
        return

    line_errors = 0
    dangling: list[str] = []
    n_queries = 0

    with open(qpath, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                report.error(f"queries.jsonl line {lineno}: JSON parse error: {exc}")
                line_errors += 1
                continue

            missing = [k for k in _QUERIES_REQUIRED if k not in rec]
            if missing:
                report.error(
                    f"queries.jsonl line {lineno}: missing required fields {missing}"
                )
                line_errors += 1
                continue

            n_queries += 1
            relevant: list[str] = rec.get("relevant") or []
            for oid in relevant:
                if all_obj_ids and oid not in all_obj_ids:
                    dangling.append(
                        f"line {lineno} query={rec['query']!r} references obj_id={oid!r} "
                        f"not in dataset"
                    )

    if dangling:
        for d in dangling:
            report.error(f"queries.jsonl: dangling relevant obj_id — {d}")
    if line_errors == 0 and not dangling:
        report.ok(f"queries.jsonl: {n_queries} queries, all valid")
    elif line_errors == 0 and dangling:
        # Already printed errors above; do not double-count.
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_dataset(path: str | Path, *, verbose: bool = True) -> bool:
    """Validate a dataset directory against the PLR dataset spec.

    Args:
        path:    Path to the dataset directory.
        verbose: If True (default), print PASS/WARN/FAIL lines to stdout.

    Returns:
        True if no errors were found (warnings allowed), False otherwise.
    """
    ds_path = Path(path)
    report = _Report()

    if verbose:
        print(f"Validating dataset: {ds_path}")

    # 1. manifest.yaml
    manifest = _check_manifest(ds_path, report)
    attribute: str | None = manifest.get("attribute") if manifest else None

    # 2. labels.jsonl
    labeled_ids = _check_labels(ds_path, attribute, report)

    # 3. crops/
    crop_ids = _check_crops(ds_path, labeled_ids, report)

    # 4. queries.jsonl (optional)
    all_dataset_ids = labeled_ids | crop_ids
    _check_queries(ds_path, all_dataset_ids, report)

    # Summary
    if verbose:
        print(
            f"\nSummary: {len(crop_ids)} crops, {len(labeled_ids)} labels, "
            f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
        )
        if report.has_errors:
            print("Result: FAIL (errors found — see above)")
        else:
            print("Result: PASS")

    return not report.has_errors
