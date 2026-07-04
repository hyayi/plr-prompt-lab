"""Dataset-as-parameter tests — no GPU, no DB, no redis.

Verifies:
  1. Dataset(path) accessors + manifest + obj_ids().
  2. resolve_dataset_dir: --dataset wins; else backward-compat golden/<attr>.
  3. `run`-equivalent (re_score) + `eval`-equivalent (run_eval) work against a
     synthetic dataset dir built at an ARBITRARY path with a mock model.
  4. search `run` + `eval` work against an arbitrary dataset dir.
"""

from __future__ import annotations

import io
import json
import sys
import textwrap
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))


_MOCK_PLR_YAML = textwrap.dedent("""\
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
      gender: 0.8
      age: 1.0
      outfit: 0.8
""")

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


# Canonical MockModel now lives in gemma_model; bind this file's default YAML.
import functools  # noqa: E402

from gemma_model import MockModel as _MockModel  # noqa: E402

MockModel = functools.partial(_MockModel, _MOCK_PLR_YAML)


def _tiny_jpg(path: Path, rgb: tuple[int, int, int] = (128, 128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 150), rgb).save(str(path), format="JPEG")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# =====================================================================
# Test 1: Dataset accessors
# =====================================================================


def test_dataset_accessors_and_manifest(tmp_path: Path) -> None:
    from evalkit import dataset as ds_mod

    d = tmp_path / "my_dataset"
    (d / "crops").mkdir(parents=True)
    _write_jsonl(d / "predictions.jsonl", [
        {"obj_id": "o1", "pred": "male", "reason": ""},
        {"obj_id": "o2", "pred": "female", "reason": ""},
    ])
    (d / "manifest.yaml").write_text(
        "attribute: gender\nn: 2\ncreated: 2026-07-01\nsource_note: synthetic\n",
        encoding="utf-8",
    )

    ds = ds_mod.Dataset(d)
    assert ds.crops_dir == d / "crops"
    assert ds.labels_path == d / "labels.jsonl"
    assert ds.queries_path == d / "queries.jsonl"
    assert ds.obj_ids() == ["o1", "o2"]

    manifest = ds.manifest
    assert manifest["attribute"] == "gender"
    assert manifest["n"] == 2
    assert manifest["source_note"] == "synthetic"


def test_resolve_dataset_dir_precedence(tmp_path: Path) -> None:
    from evalkit import dataset as ds_mod

    # --dataset given → used verbatim
    explicit = tmp_path / "arbitrary"
    assert ds_mod.resolve_dataset_dir(_LAB_ROOT, "gender", str(explicit)) == explicit

    # --dataset None → backward-compat golden/<attribute>
    fallback = ds_mod.resolve_dataset_dir(_LAB_ROOT, "gender", None)
    assert fallback == Path(_LAB_ROOT) / "eval" / "golden" / "gender"


# =====================================================================
# Test 2: attr run + eval against an arbitrary dataset dir
# =====================================================================


def _run_eval_against(gdir: Path, version: str, ledger_path: Path) -> dict:
    """run_eval CLI 제거 후 score() 직접호출 (반환: 지표 dict)."""
    from tests.scoring_helper import score_record
    return score_record(gdir, "gender")


def test_run_and_eval_on_arbitrary_dataset(tmp_path: Path) -> None:
    """Build a synthetic dataset at an arbitrary path, re_score it with a mock
    model (the `run` core), then eval it (the `eval` core). GPU-free."""
    from runners import re_score as rs

    ds_dir = tmp_path / "some" / "where" / "genderset"
    (ds_dir / "crops").mkdir(parents=True)
    obj_ids = ["a1", "a2", "a3"]
    _write_jsonl(ds_dir / "predictions.jsonl",
                 [{"obj_id": o, "pred": "male", "reason": ""} for o in obj_ids])
    _write_jsonl(ds_dir / "labels.jsonl",
                 [{"obj_id": o, "true": "female"} for o in obj_ids])
    for o in obj_ids:
        _tiny_jpg(ds_dir / "crops" / f"{o}.jpg")

    # `run` core — re_score writes predictions.jsonl (female) + attributes.jsonl
    meta = rs.re_score("gender", MockModel(), golden_dir=str(ds_dir))
    assert meta["n"] == 3
    preds = _read_jsonl(ds_dir / "predictions.jsonl")
    assert all(r["pred"] == "female" for r in preds)
    assert (ds_dir / "attributes.jsonl").exists()

    # `eval` core — score() 직접호출 (run_eval CLI/ledger 제거 후, 지표 dict 반환)
    rec = _run_eval_against(ds_dir, "ds_v1", tmp_path / "ledger.jsonl")
    assert rec["accuracy"] == pytest.approx(1.0, abs=1e-4)
    assert "accuracy" in rec


# =====================================================================
# Test 3: search run + eval against an arbitrary dataset dir
# =====================================================================


