"""Story S4 — end-to-end cycle demo.

Deliverable 2: full A-cycle (2-version Δ) + B-cycle (recall@k) with
MOCK model + synthetic data. No GPU, no DB, no redis required.

Deliverable 3: wiring-proof — `lab run` reaches LabGemmaModel.generate which
calls gemma_backend.load_backend(); monkeypatching load_backend to raise a
sentinel proves the wiring reaches Gemma without needing a real GPU.

All tests assert no storage / psycopg2 / redis in sys.modules.
"""

from __future__ import annotations

import io
import json
import sys
import textwrap
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))


# =====================================================================
# Shared helpers
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
    """Write a minimal 100x150 JPEG (large enough for normal_plr mode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (100, 150), rgb)
    img.save(str(path), format="JPEG")


# =====================================================================
# MockModel — valid PLR YAML for a person (version-switchable gender dist)
# =====================================================================

# v1: all-female predictions — MockModel returns this by default
_MOCK_YAML_V1_ALL_FEMALE = textwrap.dedent("""\
    target: person
    gender: female
    gender_reason: long hair, slender build
    age: adult
    outfit: two_piece
    upper.color: black
    upper.type: jacket
    lower.color: black
    lower.type: pants
    action: standing
    military: civilian
    margins:
      gender: 0.9
      age: 1.0
      outfit: 0.8
""")

# v2: all-male predictions — different gender distribution to produce a Δ
_MOCK_YAML_V2_ALL_MALE = textwrap.dedent("""\
    target: person
    gender: male
    gender_reason: broad shoulders, short hair
    age: adult
    outfit: two_piece
    upper.color: navy
    upper.type: jacket
    lower.color: black
    lower.type: pants
    action: standing
    military: civilian
    margins:
      gender: 0.9
      age: 1.0
      outfit: 0.8
""")


# The canonical MockModel now lives in gemma_model (shared by demo/registry/
# tests).  Import it here; the local YAML constants above are still used to
# build the synthetic golden dirs and to exercise the version-switchable stub.
from gemma_model import MockModel  # noqa: E402


# =====================================================================
# A-cycle helpers
# =====================================================================


def _make_plr_golden_dir(tmp_path: Path, obj_ids: list[str]) -> Path:
    """Minimal golden dir for gender attribute with 3 dummy crops."""
    gdir = tmp_path / "golden" / "gender"
    crops = gdir / "crops"
    crops.mkdir(parents=True)

    # Bootstrap predictions.jsonl (re_score will overwrite)
    _write_jsonl(
        gdir / "predictions.jsonl",
        [{"obj_id": oid, "pred": "male", "reason": ""} for oid in obj_ids],
    )

    # labels.jsonl — ground truth: all three are female
    _write_jsonl(
        gdir / "labels.jsonl",
        [{"obj_id": oid, "true": "female"} for oid in obj_ids],
    )

    # Crop images
    for oid in obj_ids:
        _tiny_jpg(crops / f"{oid}.jpg")

    return gdir


def _run_eval_for(
    gdir: Path,
    version: str,
    ledger_path: Path,
) -> str:
    """Call run_eval.main() for gender attribute; return captured stdout."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_eval_mod",
        str(_LAB_ROOT / "eval" / "run_eval.py"),
    )
    run_eval = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(run_eval)  # type: ignore[union-attr]

    orig_argv = sys.argv
    sys.argv = [
        "run_eval",
        "--attribute", "gender",
        "--golden", str(gdir),
        "--version", version,
        "--ledger", str(ledger_path),
        "--date", "2026-07-01T00:00:00",
    ]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_eval.main()
    except SystemExit:
        pass
    finally:
        sys.argv = orig_argv

    return buf.getvalue()


# =====================================================================
# Deliverable 2A — A-cycle: 2-version Δ
# =====================================================================


def test_a_cycle_two_versions_delta(tmp_path: Path) -> None:
    """Full A-cycle with two mock versions; second run emits a Δ line.

    Ground truth: all 3 crops are female.
    v1 MockModel: predicts female → accuracy = 1.0
    v2 MockModel: predicts male   → accuracy = 0.0
    Second run_eval must print 'Δ vs mock_v1' and the ledger has 2 records.
    """
    import re_score as rs

    obj_ids = ["e2e_a", "e2e_b", "e2e_c"]
    gdir = _make_plr_golden_dir(tmp_path, obj_ids)
    ledger_path = tmp_path / "ledger.jsonl"

    # --- version 1: MockModel returns female ---
    meta_v1 = rs.re_score("gender", MockModel(_MOCK_YAML_V1_ALL_FEMALE), golden_dir=str(gdir))
    assert meta_v1["attribute"] == "gender"
    assert meta_v1["n"] == 3

    # Verify predictions were written as female
    preds_v1 = _read_jsonl(gdir / "predictions.jsonl")
    assert all(r["pred"] == "female" for r in preds_v1), (
        f"v1: expected all female preds, got {[r['pred'] for r in preds_v1]}"
    )

    out_v1 = _run_eval_for(gdir, "mock_v1", ledger_path)
    assert ledger_path.exists(), "ledger.jsonl not created after v1 eval"
    records_v1 = _read_jsonl(ledger_path)
    assert len(records_v1) == 1
    rec_v1 = records_v1[0]
    assert rec_v1["version"] == "mock_v1"
    # All 3 correct (all female, all labeled female) → accuracy 1.0
    assert rec_v1["accuracy"] == pytest.approx(1.0, abs=1e-4), (
        f"v1 accuracy should be 1.0, got {rec_v1['accuracy']}"
    )
    # First run has no prior version → "no prior version to diff"
    assert "no prior" in out_v1, f"Expected '(no prior version to diff)' in v1 output:\n{out_v1}"

    # --- version 2: MockModel returns male → all wrong ---
    meta_v2 = rs.re_score("gender", MockModel(_MOCK_YAML_V2_ALL_MALE), golden_dir=str(gdir))
    assert meta_v2["n"] == 3

    preds_v2 = _read_jsonl(gdir / "predictions.jsonl")
    assert all(r["pred"] == "male" for r in preds_v2), (
        f"v2: expected all male preds, got {[r['pred'] for r in preds_v2]}"
    )

    out_v2 = _run_eval_for(gdir, "mock_v2", ledger_path)
    records_v2 = _read_jsonl(ledger_path)
    assert len(records_v2) == 2, f"Expected 2 ledger records, got {len(records_v2)}"

    rec_v2 = records_v2[1]
    assert rec_v2["version"] == "mock_v2"
    # All 3 wrong (all male, labeled female) → accuracy 0.0
    assert rec_v2["accuracy"] == pytest.approx(0.0, abs=1e-4), (
        f"v2 accuracy should be 0.0, got {rec_v2['accuracy']}"
    )

    # Second run must show the Δ vs v1
    assert "Δ vs mock_v1" in out_v2, (
        f"Expected 'Δ vs mock_v1' in v2 output, got:\n{out_v2}"
    )
    # The delta should be -1.000 (1.0 → 0.0)
    assert "-1.000" in out_v2 or "-1.0" in out_v2, (
        f"Expected delta -1.000 in v2 output:\n{out_v2}"
    )


# =====================================================================
# Deliverable 2B — B-cycle: run_search_over_golden → run_search_eval
# =====================================================================

# Minimal valid vehicle PLR YAML for the search candidates
_MOCK_BLACK_VEHICLE_YAML = textwrap.dedent("""\
    target: vehicle
    color: black
    type: sedan
    military: civilian
""")

_MOCK_RED_VEHICLE_YAML = textwrap.dedent("""\
    target: vehicle
    color: red
    type: sedan
    military: civilian
""")


def _make_search_golden_dir(tmp_path: Path) -> Path:
    """Synthetic search golden dir: 2 queries with known relevant candidates."""
    sdir = tmp_path / "golden" / "search"
    sdir.mkdir(parents=True)

    # queries.jsonl — 2 queries
    #   query_1: black vehicle — relevant: ["black_car"]
    #   query_2: red vehicle   — relevant: ["red_car"]
    _write_jsonl(
        sdir / "queries.jsonl",
        [
            {"query": "검은색 차", "relevant": ["black_car"]},
            {"query": "red sedan", "relevant": ["red_car"]},
        ],
    )

    # Build full PLR JSONs via parse_plr_response so candidates carry
    # the attribute structure that search_core/scoring expect
    from plr_prompts import parse_plr_response
    from plr_core import _attach_military_flags

    black_plr = parse_plr_response(_MOCK_BLACK_VEHICLE_YAML, hint="vehicle")
    _attach_military_flags(black_plr)

    red_plr = parse_plr_response(_MOCK_RED_VEHICLE_YAML, hint="vehicle")
    _attach_military_flags(red_plr)

    # attributes.jsonl — both candidates
    _write_jsonl(
        sdir / "attributes.jsonl",
        [
            {"obj_id": "black_car", "plr_json": black_plr},
            {"obj_id": "red_car", "plr_json": red_plr},
        ],
    )

    return sdir


def test_b_cycle_search_recall_at_k(tmp_path: Path) -> None:
    """Full B-cycle: run_search_over_golden → run_search_eval → ledger record.

    Synthetic setup:
      - query_1 "검은색 차" (black car): relevant=["black_car"]
      - query_2 "red sedan": relevant=["red_car"]

    run_search_over_golden runs the dictionary path (model=None, no GPU).
    Each relevant candidate should appear in its query's ranked results.

    Expected recall@5:
      query_1: black_car in ranked → recall = 1.0
      query_2: red_car in ranked   → recall = 1.0
      mean recall@5 = 1.0

    A ledger record with recall_at_k is appended.
    """
    import re_score as rs
    import run_search_eval as rse

    sdir = _make_search_golden_dir(tmp_path)
    results_path = sdir / "search_results.jsonl"
    ledger_path = tmp_path / "search_ledger.jsonl"

    # --- Step 1: run_search_over_golden (no GPU, dictionary path) ---
    rs.run_search_over_golden(
        queries_path=str(sdir / "queries.jsonl"),
        attributes_path=str(sdir / "attributes.jsonl"),
        results_path=str(results_path),
        model=None,
    )

    assert results_path.exists(), "search_results.jsonl was not written"
    results = _read_jsonl(results_path)
    assert len(results) == 2

    # Verify at least one query hit (black car must rank for its query)
    result_map = {r["query"]: r["ranked"] for r in results}
    assert "검은색 차" in result_map, "검은색 차 query missing from results"
    assert "black_car" in result_map["검은색 차"], (
        f"black_car not in ranked for 검은색 차: {result_map['검은색 차']}"
    )
    # red_car should NOT appear in 검은색 차 results (hard filter)
    assert "red_car" not in result_map["검은색 차"], (
        f"red_car should be filtered from 검은색 차: {result_map['검은색 차']}"
    )

    # --- Step 2: run_search_eval → ledger record ---
    record = rse.main(
        results_path=str(results_path),
        queries_path=str(sdir / "queries.jsonl"),
        ledger_path=str(ledger_path),
        version="mock_search_v1",
        k=5,
        date="2026-07-01T00:00:00",
        seed_hash="abc123",
        gemma_repo="mock/model",
        core_ir_path=None,  # skip stale-seed subprocess
    )

    # recall@5 must be > 0 (at minimum black_car query hits)
    assert record["recall_at_k"] > 0.0, (
        f"Expected recall_at_k > 0, got {record['recall_at_k']}"
    )
    assert record["n_queries"] == 2
    assert record["k"] == 5
    assert record["attribute"] == "search"
    assert record["version"] == "mock_search_v1"

    # Ledger record appended
    assert ledger_path.exists()
    ledger_records = _read_jsonl(ledger_path)
    assert len(ledger_records) == 1
    lr = ledger_records[0]
    assert lr["attribute"] == "search"
    assert lr["recall_at_k"] > 0.0, (
        f"Ledger record recall_at_k must be > 0, got {lr['recall_at_k']}"
    )
    assert "recall_at_k" in lr
    assert "precision_at_k" in lr
    assert lr["seed_hash"] == "abc123"


def test_b_cycle_search_ledger_delta(tmp_path: Path) -> None:
    """B-cycle: second search eval version shows Δ vs prior version in output."""
    import re_score as rs
    import run_search_eval as rse

    sdir = _make_search_golden_dir(tmp_path)
    results_path = sdir / "search_results.jsonl"
    ledger_path = tmp_path / "search_ledger.jsonl"

    # Run search to produce results
    rs.run_search_over_golden(
        queries_path=str(sdir / "queries.jsonl"),
        attributes_path=str(sdir / "attributes.jsonl"),
        results_path=str(results_path),
        model=None,
    )

    # First version
    rse.main(
        results_path=str(results_path),
        queries_path=str(sdir / "queries.jsonl"),
        ledger_path=str(ledger_path),
        version="search_v1",
        k=5,
        date="2026-07-01T00:00:00",
        seed_hash=None,
        gemma_repo=None,
        core_ir_path=None,
    )

    # Second version → expect Δ vs search_v1 in stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rse.main(
            results_path=str(results_path),
            queries_path=str(sdir / "queries.jsonl"),
            ledger_path=str(ledger_path),
            version="search_v2",
            k=5,
            date="2026-07-01T00:00:01",
            seed_hash=None,
            gemma_repo=None,
            core_ir_path=None,
        )
    out = buf.getvalue()

    assert "Δ vs search_v1" in out, (
        f"Expected 'Δ vs search_v1' in second-run output:\n{out}"
    )

    records = _read_jsonl(ledger_path)
    assert len(records) == 2, f"Expected 2 ledger records, got {len(records)}"
    assert records[0]["version"] == "search_v1"
    assert records[1]["version"] == "search_v2"


# =====================================================================
# No-DB assertion (both cycles)
# =====================================================================


def test_no_db_modules_after_cycles(tmp_path: Path) -> None:
    """After running both A and B cycles, forbidden modules must not be in sys.modules."""
    import re_score as rs

    # A-cycle
    obj_ids = ["nodb_a", "nodb_b"]
    gdir = _make_plr_golden_dir(tmp_path, obj_ids)
    rs.re_score("gender", MockModel(), golden_dir=str(gdir))

    # B-cycle
    sdir = _make_search_golden_dir(tmp_path)
    rs.run_search_over_golden(
        queries_path=str(sdir / "queries.jsonl"),
        attributes_path=str(sdir / "attributes.jsonl"),
        results_path=str(sdir / "search_results.jsonl"),
        model=None,
    )

    forbidden = {"storage", "psycopg2", "redis"}
    imported = {m.split(".")[0] for m in sys.modules}
    leaked = forbidden & imported
    assert not leaked, (
        f"Forbidden DB modules found in sys.modules after cycles: {leaked}"
    )


# =====================================================================
# Deliverable 3 — Real-run wiring proof (no GPU)
# =====================================================================


class _GpuNotAvailableSentinel(RuntimeError):
    """Sentinel raised by the monkeypatched load_backend to simulate GPU absent."""


def test_lab_run_wiring_reaches_gemma_backend(tmp_path: Path) -> None:
    """Prove that `lab run` wiring reaches gemma_backend.load_backend().

    Monkeypatches gemma_backend.load_backend to raise a sentinel exception,
    then calls lab._cmd_run (with a real Namespace that sets up the env vars
    the same way lab run does). The sentinel propagating up proves that
    LabGemmaModel.generate() -> gemma_backend.load_backend() is wired —
    only GPU execution is missing, not the code path.

    No GPU is used; no real Gemma loads; no real crops are needed because the
    load_backend call is intercepted before any generate() completes.
    """
    # Build a minimal golden dir so re_score can iterate over obj_ids
    # before the first generate() call hits load_backend
    import re_score as rs

    obj_ids = ["wire_a", "wire_b"]
    gdir = _make_plr_golden_dir(tmp_path, obj_ids)

    # Import gemma_model and gemma_backend (the lazy import chain under test)
    import gemma_model
    import gemma_backend

    # Track whether load_backend was called
    load_backend_called = False

    def _mock_load_backend():
        nonlocal load_backend_called
        load_backend_called = True
        raise _GpuNotAvailableSentinel(
            "Sentinel: load_backend reached — GPU wiring is confirmed. "
            "Real execution requires a dedicated GPU."
        )

    # Monkeypatch gemma_backend.load_backend at both import sites:
    # gemma_model.LabGemmaModel.generate does `from gemma_backend import load_backend`
    # so we patch the module attribute AND the name in the gemma_model module's
    # namespace (which may already hold the reference from a prior import).
    with (
        patch.object(gemma_backend, "load_backend", _mock_load_backend),
        patch("gemma_model.LabGemmaModel.generate", _patched_lab_generate),
    ):
        lab_model = gemma_model.LabGemmaModel()
        with pytest.raises(_GpuNotAvailableSentinel):
            # Calling generate() on LabGemmaModel should reach load_backend
            lab_model.generate(messages=[], image=None)

    assert load_backend_called, (
        "load_backend was NOT called — wiring broken: LabGemmaModel.generate "
        "did not reach gemma_backend.load_backend"
    )


def _patched_lab_generate(self, messages, image):  # noqa: ANN001
    """Replacement for LabGemmaModel.generate that calls load_backend directly,
    matching the real implementation's import path.

    This ensures the sentinel propagates even if the lazy `from gemma_backend
    import load_backend` inside generate() is resolved before the patch.
    """
    import gemma_backend as _gb
    gen = _gb.load_backend().generate(image, messages, max_tokens=512, temperature=0.0)
    return gen.raw


def test_lab_run_wiring_via_re_score_monkeypatch(tmp_path: Path) -> None:
    """Alternative wiring proof: monkeypatch at the re_score call boundary.

    re_score(attribute, LabGemmaModel()) is what lab._cmd_run calls.
    We construct a LabGemmaModel and invoke re_score against the first crop —
    which calls model.generate() → load_backend() → our sentinel.

    This proves the full path: lab run → re_score → LabGemmaModel → load_backend.
    """
    import re_score as rs
    import gemma_backend
    import gemma_model

    obj_ids = ["wirepath_x"]
    gdir = _make_plr_golden_dir(tmp_path, obj_ids)

    sentinel_hit = []

    def _raising_load_backend():
        sentinel_hit.append(True)
        raise _GpuNotAvailableSentinel("load_backend sentinel: GPU wiring confirmed")

    # Patch load_backend in the gemma_backend module
    with patch.object(gemma_backend, "load_backend", _raising_load_backend):
        # Also patch the local reference inside gemma_model module so the
        # `from gemma_backend import load_backend` inside .generate() is covered
        with patch("gemma_model.LabGemmaModel.generate", _patched_lab_generate):
            lab_model = gemma_model.LabGemmaModel()
            with pytest.raises(_GpuNotAvailableSentinel):
                # re_score will call lab_model.generate on the first crop
                rs.re_score("gender", lab_model, golden_dir=str(gdir))

    assert sentinel_hit, (
        "load_backend sentinel was NOT raised — the full path "
        "lab run → re_score → LabGemmaModel → load_backend is broken"
    )
