"""P2-2: experiment matrix runner tests.

Tests:
  1. Two-cell matrix (2 prompts × 1 dataset × plr × mock × gender) → 2 ledger records.
  2. Fail-loud-but-continue: one cell has a missing dataset → that cell fails,
     the other cell completes and writes its ledger record.
  3. Unknown model name in yaml → clear ValueError before any cell runs.
  4. Unknown pipeline name in yaml → clear ValueError before any cell runs.
  5. Exit code 0 when all cells pass; exit code 1 with --strict when one fails;
     exit code 2 when ALL cells fail.

No GPU, no DB, no Redis required.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))


# =====================================================================
# Shared helpers (mirrors demo.py / test_cycle_e2e.py patterns)
# =====================================================================


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tiny_jpg(path: Path, rgb: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Minimal 100×150 JPEG (large enough for quality_gate normal_plr mode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (100, 150), rgb)
    img.save(str(path), format="JPEG")


def _make_plr_dataset(base: Path, obj_ids: list[str]) -> Path:
    """Create a minimal valid PLR dataset directory."""
    ds = base
    crops = ds / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        ds / "predictions.jsonl",
        [{"obj_id": oid, "pred": "unknown", "reason": ""} for oid in obj_ids],
    )
    _write_jsonl(
        ds / "labels.jsonl",
        [{"obj_id": oid, "label": "female"} for oid in obj_ids],
    )
    for oid in obj_ids:
        _tiny_jpg(crops / f"{oid}.jpg")

    return ds


def _write_experiment_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


# =====================================================================
# Test 1: 2-cell matrix → 2 ledger records
# =====================================================================


def test_two_cell_matrix_writes_two_ledger_records(tmp_path: Path) -> None:
    """2 prompts × 1 dataset × plr × mock × gender = 2 cells → 2 ledger records.

    Each record must carry: dataset, model=mock, pipeline=plr, prompt_hash,
    and accuracy (from run_eval).
    """
    ds_dir = _make_plr_dataset(tmp_path / "ds_gender", ["p1", "p2", "p3"])
    ledger_path = tmp_path / "ledger.jsonl"

    yaml_path = tmp_path / "experiment.yaml"
    _write_experiment_yaml(
        yaml_path,
        f"""\
        datasets:
          - {str(ds_dir)}
        models:
          - mock
        prompts:
          - mock_v1
          - mock_v2
        pipelines:
          - plr
        attributes:
          - gender
        ledger: {str(ledger_path)}
        """,
    )

    from runners.experiment import run_experiment

    exit_code = run_experiment(str(yaml_path))

    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    assert ledger_path.exists(), "ledger.jsonl was not created"

    records = _read_jsonl(ledger_path)
    assert len(records) == 2, (
        f"Expected 2 ledger records (one per prompt), got {len(records)}: {records}"
    )

    versions = {r["version"] for r in records}
    assert versions == {"mock_v1", "mock_v2"}, (
        f"Expected versions {{mock_v1, mock_v2}}, got {versions}"
    )

    for r in records:
        assert r["model"] == "mock", f"model should be 'mock', got {r.get('model')}"
        assert r["pipeline"] == "plr", f"pipeline should be 'plr', got {r.get('pipeline')}"
        assert "dataset" in r, "ledger record missing 'dataset' key"
        assert "accuracy" in r, "ledger record missing 'accuracy' key"
        assert "prompt_hash" in r, "ledger record missing 'prompt_hash' key"
        assert r["attribute"] == "gender", (
            f"attribute should be 'gender', got {r.get('attribute')}"
        )


# =====================================================================
# Test 2: fail-loud-but-continue — missing dataset cell fails,
#          the other cell still completes
# =====================================================================


def test_fail_loud_but_continue_missing_dataset(tmp_path: Path) -> None:
    """One cell points at a missing dataset → it fails; the other cell completes.

    Setup: 2 datasets in yaml — one real, one nonexistent.
    Expected: 1 ledger record (from the real dataset), exit code 0
              (at least one cell succeeded without --strict).
    """
    ds_real = _make_plr_dataset(tmp_path / "ds_real", ["x1", "x2"])
    ds_missing = tmp_path / "ds_missing_does_not_exist"
    ledger_path = tmp_path / "ledger.jsonl"

    yaml_path = tmp_path / "experiment.yaml"
    _write_experiment_yaml(
        yaml_path,
        f"""\
        datasets:
          - {str(ds_real)}
          - {str(ds_missing)}
        models:
          - mock
        prompts:
          - mock_v1
        pipelines:
          - plr
        attributes:
          - gender
        ledger: {str(ledger_path)}
        """,
    )

    from runners.experiment import run_experiment

    exit_code = run_experiment(str(yaml_path))

    # At least one cell succeeded → exit code 0 (not --strict)
    assert exit_code == 0, (
        f"Expected exit code 0 (one cell ok), got {exit_code}"
    )

    # Ledger must contain exactly 1 record (the real-dataset cell)
    assert ledger_path.exists(), "ledger.jsonl was not created by the successful cell"
    records = _read_jsonl(ledger_path)
    assert len(records) == 1, (
        f"Expected 1 ledger record from the successful cell, got {len(records)}: {records}"
    )
    assert str(ds_real) in records[0].get("dataset", ""), (
        f"Ledger record dataset should reference ds_real, got {records[0].get('dataset')}"
    )


def test_fail_loud_continue_strict_flag(tmp_path: Path) -> None:
    """With --strict: one cell fails → exit code 1 (not 0)."""
    ds_real = _make_plr_dataset(tmp_path / "ds_real", ["y1"])
    ds_missing = tmp_path / "ds_gone"
    ledger_path = tmp_path / "ledger.jsonl"

    yaml_path = tmp_path / "experiment.yaml"
    _write_experiment_yaml(
        yaml_path,
        f"""\
        datasets:
          - {str(ds_real)}
          - {str(ds_missing)}
        models:
          - mock
        prompts:
          - mock_v1
        pipelines:
          - plr
        attributes:
          - gender
        ledger: {str(ledger_path)}
        """,
    )

    from runners.experiment import run_experiment

    exit_code = run_experiment(str(yaml_path), strict=True)
    assert exit_code == 1, (
        f"Expected exit code 1 with --strict + one failed cell, got {exit_code}"
    )

    # Successful cell still wrote its record
    records = _read_jsonl(ledger_path)
    assert len(records) == 1, (
        f"Expected 1 ledger record from the successful cell, got {len(records)}"
    )


def test_all_cells_fail_returns_exit_code_2(tmp_path: Path) -> None:
    """When ALL cells fail, exit code is 2."""
    ds_missing1 = tmp_path / "ds_gone1"
    ds_missing2 = tmp_path / "ds_gone2"
    ledger_path = tmp_path / "ledger.jsonl"

    yaml_path = tmp_path / "experiment.yaml"
    _write_experiment_yaml(
        yaml_path,
        f"""\
        datasets:
          - {str(ds_missing1)}
          - {str(ds_missing2)}
        models:
          - mock
        prompts:
          - mock_v1
        pipelines:
          - plr
        attributes:
          - gender
        ledger: {str(ledger_path)}
        """,
    )

    from runners.experiment import run_experiment

    exit_code = run_experiment(str(yaml_path))
    assert exit_code == 2, (
        f"Expected exit code 2 when all cells fail, got {exit_code}"
    )


# =====================================================================
# Test 3: Unknown model name → clear error before any cell runs
# =====================================================================


def test_unknown_model_raises_before_running(tmp_path: Path) -> None:
    """An unknown model name in yaml raises ValueError before any cell runs."""
    ds_dir = _make_plr_dataset(tmp_path / "ds", ["a1"])
    ledger_path = tmp_path / "ledger.jsonl"

    yaml_path = tmp_path / "experiment.yaml"
    _write_experiment_yaml(
        yaml_path,
        f"""\
        datasets:
          - {str(ds_dir)}
        models:
          - definitely_not_a_real_model
        prompts:
          - mock_v1
        pipelines:
          - plr
        attributes:
          - gender
        ledger: {str(ledger_path)}
        """,
    )

    from runners.experiment import run_experiment

    with pytest.raises(ValueError, match="unknown model"):
        run_experiment(str(yaml_path))

    # No ledger record should have been written (error before cells ran)
    assert not ledger_path.exists(), (
        "Ledger should not exist — error must fire before any cell runs"
    )


# =====================================================================
# Test 4: Unknown pipeline name → clear error before any cell runs
# =====================================================================


def test_unknown_pipeline_raises_before_running(tmp_path: Path) -> None:
    """An unknown pipeline name in yaml raises ValueError before any cell runs."""
    ds_dir = _make_plr_dataset(tmp_path / "ds", ["b1"])
    ledger_path = tmp_path / "ledger.jsonl"

    yaml_path = tmp_path / "experiment.yaml"
    _write_experiment_yaml(
        yaml_path,
        f"""\
        datasets:
          - {str(ds_dir)}
        models:
          - mock
        prompts:
          - mock_v1
        pipelines:
          - not_a_pipeline
        attributes:
          - gender
        ledger: {str(ledger_path)}
        """,
    )

    from runners.experiment import run_experiment

    with pytest.raises(ValueError, match="unknown pipeline"):
        run_experiment(str(yaml_path))

    assert not ledger_path.exists(), (
        "Ledger should not exist — error must fire before any cell runs"
    )


# =====================================================================
# Test 5: lab CLI wiring — `lab experiment run` dispatches correctly
# =====================================================================


def test_lab_experiment_run_cli_wiring(tmp_path: Path) -> None:
    """lab.py experiment run <yaml> routes to _cmd_experiment_run → run_experiment.

    Smoke test: valid 1-cell experiment yaml, run via lab._cmd_experiment_run,
    verify ledger record written.
    """
    import argparse
    import lab as lab_module

    ds_dir = _make_plr_dataset(tmp_path / "ds_cli", ["cli_1", "cli_2"])
    ledger_path = tmp_path / "cli_ledger.jsonl"

    yaml_path = tmp_path / "cli_experiment.yaml"
    _write_experiment_yaml(
        yaml_path,
        f"""\
        datasets:
          - {str(ds_dir)}
        models:
          - mock
        prompts:
          - mock_cli_v1
        pipelines:
          - plr
        attributes:
          - gender
        ledger: {str(ledger_path)}
        """,
    )

    args = argparse.Namespace(
        experiment_yaml=str(yaml_path),
        strict=False,
    )
    exit_code = lab_module._cmd_experiment_run(args)

    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    assert ledger_path.exists()
    records = _read_jsonl(ledger_path)
    assert len(records) == 1
    assert records[0]["model"] == "mock"
    assert records[0]["pipeline"] == "plr"
    assert records[0]["version"] == "mock_cli_v1"


# =====================================================================
# Test 6: enumerate_cells cross-product shape
# =====================================================================


def test_enumerate_cells_cross_product(tmp_path: Path) -> None:
    """enumerate_cells returns the correct number of cells for each pipeline."""
    from runners.experiment import enumerate_cells

    # plr: 2 datasets × 2 models × 3 prompts × 1 pipeline × 2 attributes = 24
    cfg_plr: dict[str, Any] = {
        "datasets": ["ds1", "ds2"],
        "models": ["mock", "gemma"],
        "prompts": ["v1", "v2", "v3"],
        "pipelines": ["plr"],
        "attributes": ["gender", "military"],
    }
    cells_plr = enumerate_cells(cfg_plr)
    assert len(cells_plr) == 2 * 2 * 3 * 2, (
        f"Expected 24 plr cells, got {len(cells_plr)}"
    )
    for c in cells_plr:
        assert c.pipeline == "plr"
        assert c.attribute in {"gender", "military"}
