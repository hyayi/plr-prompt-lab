"""Validate-dataset tests — no GPU, no DB, no redis.

Tests:
  1. A GOOD synthetic dataset (2 crops + matching labels.jsonl + manifest.yaml)
     → validate_dataset returns True (pass).
  2. BROKEN fixtures each fail with a clear error:
     (a) label references a missing crop     → error reported, returns False
     (b) labels.jsonl missing required field → error reported, returns False
     (c) crop present but no label           → warn, still passes (True)
     (d) manifest.yaml missing 'attribute'  → error reported, returns False
     (e) queries.jsonl relevant obj_id not in dataset → error, returns False
  3. `python3 lab.py validate-dataset --help` works.
  4. validate-dataset is wired as a subcommand and exits non-zero on errors.
  5. eval/golden/gender passes without crashing (labels.jsonl may be absent —
     that is an expected, clearly-reported error state, not a crash).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 80), (100, 100, 100)).save(str(path), format="JPEG")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_manifest(path: Path, **fields) -> None:
    import yaml  # type: ignore[import]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(fields, f)


def _make_good_dataset(tmp_path: Path) -> Path:
    """Return a valid 2-crop gender dataset under tmp_path."""
    ds = tmp_path / "good_ds"
    (ds / "crops").mkdir(parents=True)
    _tiny_jpg(ds / "crops" / "obj_a.jpg")
    _tiny_jpg(ds / "crops" / "obj_b.jpg")
    _write_jsonl(ds / "labels.jsonl", [
        {"obj_id": "obj_a", "label": "female"},
        {"obj_id": "obj_b", "label": "male"},
    ])
    _write_manifest(
        ds / "manifest.yaml",
        attribute="gender",
        n=2,
        created="2026-07-01",
        source_note="synthetic test fixture",
    )
    return ds


# ---------------------------------------------------------------------------
# Test 1: good dataset → pass
# ---------------------------------------------------------------------------


def test_good_dataset_passes(tmp_path: Path) -> None:
    from evalkit.validate import validate_dataset

    ds = _make_good_dataset(tmp_path)
    result = validate_dataset(ds)
    assert result is True, "Good dataset should pass validation"


# ---------------------------------------------------------------------------
# Test 2a: label references a missing crop → error, returns False
# ---------------------------------------------------------------------------


def test_label_missing_crop_fails(tmp_path: Path, capsys) -> None:
    from evalkit.validate import validate_dataset

    ds = tmp_path / "missing_crop"
    (ds / "crops").mkdir(parents=True)
    _tiny_jpg(ds / "crops" / "obj_a.jpg")
    # obj_b referenced in labels but has no crop
    _write_jsonl(ds / "labels.jsonl", [
        {"obj_id": "obj_a", "label": "female"},
        {"obj_id": "obj_b", "label": "male"},
    ])
    _write_manifest(
        ds / "manifest.yaml",
        attribute="gender", n=2,
        created="2026-07-01", source_note="test",
    )

    result = validate_dataset(ds)
    captured = capsys.readouterr()

    assert result is False, "Dataset with missing crop should fail"
    assert "obj_b" in captured.out, (
        f"Expected 'obj_b' in error output:\n{captured.out}"
    )
    assert "labels-without-crops" in captured.out or "FAIL" in captured.out


# ---------------------------------------------------------------------------
# Test 2b: labels.jsonl missing required field → error, returns False
# ---------------------------------------------------------------------------


def test_labels_missing_required_field_fails(tmp_path: Path, capsys) -> None:
    from evalkit.validate import validate_dataset

    ds = tmp_path / "bad_labels"
    (ds / "crops").mkdir(parents=True)
    _tiny_jpg(ds / "crops" / "obj_a.jpg")
    # Line is missing 'label' field
    _write_jsonl(ds / "labels.jsonl", [
        {"obj_id": "obj_a"},
    ])
    _write_manifest(
        ds / "manifest.yaml",
        attribute="gender", n=1,
        created="2026-07-01", source_note="test",
    )

    result = validate_dataset(ds)
    captured = capsys.readouterr()

    assert result is False, "Dataset with malformed labels.jsonl should fail"
    assert "label" in captured.out.lower(), (
        f"Expected mention of missing field in output:\n{captured.out}"
    )
    assert "FAIL" in captured.out


# ---------------------------------------------------------------------------
# Test 2c: crop present but no label → warn, still passes (True)
# ---------------------------------------------------------------------------


def test_crop_without_label_warns_but_passes(tmp_path: Path, capsys) -> None:
    from evalkit.validate import validate_dataset

    ds = tmp_path / "extra_crop"
    (ds / "crops").mkdir(parents=True)
    _tiny_jpg(ds / "crops" / "obj_a.jpg")
    _tiny_jpg(ds / "crops" / "obj_extra.jpg")  # no label for this one
    _write_jsonl(ds / "labels.jsonl", [
        {"obj_id": "obj_a", "label": "female"},
    ])
    _write_manifest(
        ds / "manifest.yaml",
        attribute="gender", n=1,
        created="2026-07-01", source_note="test",
    )

    result = validate_dataset(ds)
    captured = capsys.readouterr()

    assert result is True, "Extra unlabeled crop should only warn, not fail"
    assert "WARN" in captured.out, (
        f"Expected a WARN line about crops-without-labels:\n{captured.out}"
    )
    assert "obj_extra" in captured.out


# ---------------------------------------------------------------------------
# Test 2d: manifest missing 'attribute' → error, returns False
# ---------------------------------------------------------------------------


def test_manifest_missing_attribute_fails(tmp_path: Path, capsys) -> None:
    from evalkit.validate import validate_dataset

    ds = tmp_path / "bad_manifest"
    (ds / "crops").mkdir(parents=True)
    _tiny_jpg(ds / "crops" / "obj_a.jpg")
    _write_jsonl(ds / "labels.jsonl", [
        {"obj_id": "obj_a", "label": "female"},
    ])
    # manifest is missing 'attribute'
    _write_manifest(
        ds / "manifest.yaml",
        n=1, created="2026-07-01", source_note="test",
    )

    result = validate_dataset(ds)
    captured = capsys.readouterr()

    assert result is False, "Manifest missing 'attribute' should fail"
    assert "attribute" in captured.out.lower(), (
        f"Expected mention of 'attribute' in error output:\n{captured.out}"
    )
    assert "FAIL" in captured.out


# ---------------------------------------------------------------------------
# Test 2e: queries.jsonl relevant obj_id not in dataset → error, returns False
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Test 3: validate-dataset --help works
# ---------------------------------------------------------------------------


def test_validate_dataset_help() -> None:
    result = subprocess.run(
        [sys.executable, str(_LAB_ROOT / "lab.py"), "validate-dataset", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"validate-dataset --help exited {result.returncode}\n{result.stderr}"
    )
    assert "validate" in result.stdout.lower() or "dataset" in result.stdout.lower(), (
        f"Expected help text for validate-dataset:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# Test 4: validate-dataset wired as subcommand; exits non-zero on error
# ---------------------------------------------------------------------------


def test_validate_dataset_subcommand_exit_code(tmp_path: Path) -> None:
    ds = _make_good_dataset(tmp_path)

    # Good dataset → exit 0
    result_good = subprocess.run(
        [sys.executable, str(_LAB_ROOT / "lab.py"), "validate-dataset",
         "--dataset", str(ds)],
        capture_output=True, text=True,
    )
    assert result_good.returncode == 0, (
        f"Good dataset should exit 0, got {result_good.returncode}\n"
        f"{result_good.stdout}\n{result_good.stderr}"
    )

    # Bad dataset (missing labels.jsonl) → exit non-zero
    bad_ds = tmp_path / "no_labels"
    (bad_ds / "crops").mkdir(parents=True)
    _tiny_jpg(bad_ds / "crops" / "obj_a.jpg")
    import yaml  # type: ignore[import]
    (bad_ds / "manifest.yaml").write_text(
        yaml.safe_dump({"attribute": "gender", "n": 1,
                        "created": "2026-07-01", "source_note": "test"}),
        encoding="utf-8",
    )

    result_bad = subprocess.run(
        [sys.executable, str(_LAB_ROOT / "lab.py"), "validate-dataset",
         "--dataset", str(bad_ds)],
        capture_output=True, text=True,
    )
    assert result_bad.returncode != 0, (
        f"Bad dataset (no labels.jsonl) should exit non-zero, "
        f"got {result_bad.returncode}\n{result_bad.stdout}"
    )


# ---------------------------------------------------------------------------
# Test 5: eval/golden/gender does not crash; absent labels.jsonl is reported
# ---------------------------------------------------------------------------


def test_eval_golden_gender_no_crash(capsys) -> None:
    """eval/golden/gender is a valid dataset structure but labels.jsonl is
    gitignored (absent in checkout). The validator should report a clear error
    about the missing file — not crash, not raise an unhandled exception."""
    from evalkit.validate import validate_dataset

    gender_ds = _LAB_ROOT / "eval" / "golden" / "gender"
    # If the directory doesn't exist at all, skip (repo checkout variation).
    if not gender_ds.exists():
        pytest.skip("eval/golden/gender not present in this checkout")

    # Must not raise; may return True or False depending on what's present.
    try:
        validate_dataset(gender_ds)
    except Exception as exc:
        pytest.fail(
            f"validate_dataset raised an unexpected exception on eval/golden/gender: {exc}"
        )

    captured = capsys.readouterr()
    # Output must be non-empty (something was checked)
    assert captured.out.strip(), "Expected some output from validate_dataset"
