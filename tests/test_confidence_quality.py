"""Confidence (margin) + crop-quality signals in the eval loop.

plr_v1.5_cot replaced the unknown escape hatch with "commit + margin", so:
  1. re_score must RECORD the model's decision_margin and a crop-quality
     score (measurement only — never gating) in predictions.jsonl.
  2. run_eval must SPLIT accuracy by those signals (calibration check:
     do errors concentrate in low-margin / low-quality crops?).
  3. Old predictions files without the fields must still evaluate
     (stats fields None, no crash).

No GPU, no DB, no Redis.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

from PIL import Image

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

_MOCK_PLR_YAML = textwrap.dedent("""\
    target: person
    gender: female
    gender_reason: long hair
    age: adult
    outfit: two_piece
    upper.color: black
    upper.type: jacket
    lower.color: black
    lower.type: pants
    action: standing
    military: civilian
    margins:
      gender: 0.35
      age: 1.0
      outfit: 0.8
""")


class _MockModel:
    def generate(self, messages, image):  # noqa: ARG002
        return _MOCK_PLR_YAML


def _make_dataset(base: Path, obj_ids: list[str]) -> Path:
    crops = base / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    with open(base / "labels.jsonl", "w", encoding="utf-8") as f:
        for oid in obj_ids:
            f.write(json.dumps({"obj_id": oid, "label": "female"}) + "\n")
    for oid in obj_ids:
        Image.new("RGB", (100, 150), (128, 128, 128)).save(
            str(crops / f"{oid}.jpg"), format="JPEG")
    return base


def _run_eval(golden: Path, ledger: Path, extra_args: list[str] | None = None) -> dict:
    spec = importlib.util.spec_from_file_location(
        "run_eval_cq", str(_LAB_ROOT / "eval" / "run_eval.py"))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    orig = sys.argv
    sys.argv = ["run_eval", "--attribute", "gender", "--golden", str(golden),
                "--version", "cq_v1", "--ledger", str(ledger)] + (extra_args or [])
    try:
        mod.main()
    except SystemExit:
        pass
    finally:
        sys.argv = orig
    with open(ledger, encoding="utf-8") as f:
        return json.loads(f.readlines()[-1])


def test_re_score_records_margin_and_quality(tmp_path: Path) -> None:
    """Every predictions.jsonl row carries the model margin and a quality
    score in [0,1]."""
    from runners import re_score as rs

    gdir = _make_dataset(tmp_path / "ds", ["c1", "c2"])
    rs.re_score("gender", _MockModel(), golden_dir=str(gdir))

    rows = [json.loads(l) for l in open(gdir / "predictions.jsonl", encoding="utf-8")]
    assert len(rows) == 2
    for r in rows:
        assert r["margin"] == 0.35, f"margin not extracted: {r}"
        assert isinstance(r["quality"], float) and 0.0 <= r["quality"] <= 1.0, (
            f"quality missing/out of range: {r}"
        )


def test_run_eval_splits_by_margin_and_quality(tmp_path: Path) -> None:
    """Low-margin errors must show up as low-bucket accuracy 0 vs high-bucket
    accuracy 1, and mean_wrong < mean_correct."""
    gdir = tmp_path / "ds"
    gdir.mkdir()
    # 4 crops: two confident & correct, two unconfident & wrong.
    rows = [
        {"obj_id": "a", "pred": "female", "margin": 0.9, "quality": 0.8},
        {"obj_id": "b", "pred": "female", "margin": 0.8, "quality": 0.7},
        {"obj_id": "c", "pred": "male",   "margin": 0.2, "quality": 0.2},
        {"obj_id": "d", "pred": "male",   "margin": 0.3, "quality": 0.3},
    ]
    with open(gdir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(gdir / "labels.jsonl", "w", encoding="utf-8") as f:
        for oid in "abcd":
            f.write(json.dumps({"obj_id": oid, "label": "female"}) + "\n")

    rec = _run_eval(gdir, tmp_path / "ledger.jsonl")

    ms = rec["margin_stats"]
    assert ms["high"] == {"n": 2, "accuracy": 1.0}
    assert ms["low"] == {"n": 2, "accuracy": 0.0}
    assert ms["mean_wrong"] < ms["mean_correct"], (
        "calibration direction wrong: errors should have lower margins"
    )
    qs = rec["quality_stats"]
    assert qs["high"]["accuracy"] == 1.0 and qs["low"]["accuracy"] == 0.0


def test_run_eval_tolerates_missing_signals(tmp_path: Path) -> None:
    """Old predictions files (no margin/quality) evaluate fine; the stats
    fields are None."""
    gdir = tmp_path / "ds"
    gdir.mkdir()
    with open(gdir / "predictions.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"obj_id": "a", "pred": "female"}) + "\n")
    with open(gdir / "labels.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"obj_id": "a", "label": "female"}) + "\n")

    rec = _run_eval(gdir, tmp_path / "ledger.jsonl")
    assert rec["accuracy"] == 1.0
    assert rec["margin_stats"] is None and rec["quality_stats"] is None
