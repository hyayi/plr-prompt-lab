"""Lab runner tests — no GPU, no DB, no redis.

Tests:
  1. MockModel round-trip: re_score writes predictions.jsonl with mock pred,
     len == n.
  2. Missing-crop fail-loud: removing a crop raises FileNotFoundError.
  3. No-DB: storage / psycopg2 / redis NOT in sys.modules after re_score.
  4. run_search_over_golden: black vehicle ranks above red vehicle for
     "검은색 차" query; red vehicle excluded.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


# =====================================================================
# Helpers
# =====================================================================

# Minimal valid PLR YAML for a person, gender=female.
# parse_plr_yaml reads SCALAR fields: gender, upper.color, upper.type,
# lower.color, lower.type, outfit, action, military, margins.*
# (NOT nested color_topk/type_topk arrays — those are built by the parser
# from the scalar labels using _topk_one).
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

# Minimal valid PLR YAML for a vehicle, color=black (for search test).
# parse_plr_yaml vehicle path reads scalar `color` and `type` fields.
_MOCK_BLACK_VEHICLE_YAML = textwrap.dedent("""\
    target: vehicle
    color: black
    type: sedan
    military: civilian
""")

# Minimal valid PLR YAML for a vehicle, color=red (for search test).
_MOCK_RED_VEHICLE_YAML = textwrap.dedent("""\
    target: vehicle
    color: red
    type: sedan
    military: civilian
""")


def _tiny_jpg(path: Path, rgb: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Write a tiny 100×150 JPEG at path (large enough for normal_plr mode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (100, 150), rgb)
    img.save(str(path), format="JPEG")


# =====================================================================
# MockModel — returns canned YAML; ignores messages and image
# =====================================================================


# The canonical MockModel now lives in gemma_model (shared by demo/registry/
# tests).  Import it and preserve this file's default YAML by binding it as the
# stub's default via a thin partial so existing call sites (MockModel()) still
# return _MOCK_PLR_YAML.
import functools  # noqa: E402

from gemma_model import MockModel as _MockModel  # noqa: E402

MockModel = functools.partial(_MockModel, _MOCK_PLR_YAML)


# =====================================================================
# Fixture: build a minimal golden dir with 3 obj_ids
# =====================================================================


def _make_golden_dir(tmp_path: Path, obj_ids: list[str]) -> Path:
    """Create a minimal golden dir structure under tmp_path."""
    gdir = tmp_path / "golden" / "gender"
    crops = gdir / "crops"
    crops.mkdir(parents=True)

    # predictions.jsonl — initial (will be overwritten by re_score)
    with open(gdir / "predictions.jsonl", "w") as f:
        for oid in obj_ids:
            f.write(json.dumps({"obj_id": oid, "pred": "male", "reason": ""}) + "\n")

    # crop images
    for oid in obj_ids:
        _tiny_jpg(crops / f"{oid}.jpg")

    return gdir


# =====================================================================
# Test 1: MockModel round-trip
# =====================================================================


def test_re_score_mock_writes_correct_preds(tmp_path: Path) -> None:
    """re_score with MockModel overwrites predictions.jsonl with mock pred (female),
    and the number of written rows equals the number of obj_ids."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    obj_ids = ["obj_a", "obj_b", "obj_c"]
    gdir = _make_golden_dir(tmp_path, obj_ids)

    import re_score as rs

    meta = rs.re_score("gender", MockModel(), golden_dir=str(gdir))

    # --- meta shape
    assert meta["attribute"] == "gender"
    assert meta["n"] == 3

    # --- predictions.jsonl was overwritten with female pred
    with open(gdir / "predictions.jsonl") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    assert len(rows) == 3
    written_ids = {r["obj_id"] for r in rows}
    assert written_ids == set(obj_ids)

    for row in rows:
        assert row["pred"] == "female", f"Expected female, got {row['pred']!r}"
        assert "reason" in row

    # --- attributes.jsonl also written
    with open(gdir / "attributes.jsonl") as f:
        attr_rows = [json.loads(l) for l in f if l.strip()]
    assert len(attr_rows) == 3
    for ar in attr_rows:
        assert "obj_id" in ar
        assert "plr_json" in ar
        assert isinstance(ar["plr_json"], dict)


# =====================================================================
# Test 2: Missing crop → fail-loud (FileNotFoundError)
# =====================================================================


def test_re_score_missing_crop_raises(tmp_path: Path) -> None:
    """Removing one crop jpg causes re_score to raise FileNotFoundError,
    not silently skip."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    obj_ids = ["p1", "p2", "p3"]
    gdir = _make_golden_dir(tmp_path, obj_ids)

    # Remove one crop
    (gdir / "crops" / "p2.jpg").unlink()

    import re_score as rs

    with pytest.raises(FileNotFoundError):
        rs.re_score("gender", MockModel(), golden_dir=str(gdir))


# =====================================================================
# Test 3: No DB — storage / psycopg2 / redis NOT in sys.modules
# =====================================================================


def test_re_score_no_db_imports(tmp_path: Path) -> None:
    """After running re_score, the forbidden modules must not be in sys.modules."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    obj_ids = ["x1", "x2"]
    gdir = _make_golden_dir(tmp_path, obj_ids)

    import re_score as rs

    rs.re_score("gender", MockModel(), golden_dir=str(gdir))

    forbidden = {"storage", "psycopg2", "redis"}
    imported = {m.split(".")[0] for m in sys.modules}
    leaked = forbidden & imported
    assert not leaked, (
        f"Forbidden DB modules found in sys.modules after re_score: {leaked}"
    )


# =====================================================================
# Test 4: run_search_over_golden — black vehicle ranks, red excluded
# =====================================================================


def _make_search_golden_dir(tmp_path: Path) -> Path:
    """Create a minimal search golden dir with two vehicle candidates."""
    sdir = tmp_path / "golden" / "search"
    sdir.mkdir(parents=True)

    # queries.jsonl: one query for a black vehicle
    with open(sdir / "queries.jsonl", "w") as f:
        f.write(json.dumps({
            "query": "검은색 차",
            "relevant": ["black_car"],
        }, ensure_ascii=False) + "\n")

    # Build full PLR JSONs for the two candidates via parse_plr_response
    from plr_prompts import parse_plr_response
    from plr_core import _attach_military_flags

    black_plr = parse_plr_response(_MOCK_BLACK_VEHICLE_YAML, hint="vehicle")
    _attach_military_flags(black_plr)

    red_plr = parse_plr_response(_MOCK_RED_VEHICLE_YAML, hint="vehicle")
    _attach_military_flags(red_plr)

    # attributes.jsonl: full plr_json for both candidates
    with open(sdir / "attributes.jsonl", "w") as f:
        f.write(json.dumps({"obj_id": "black_car", "plr_json": black_plr}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"obj_id": "red_car", "plr_json": red_plr}, ensure_ascii=False) + "\n")

    return sdir


def test_run_search_black_vehicle_ranks_red_excluded(tmp_path: Path) -> None:
    """'검은색 차' query: black car passes hard filter, red car is excluded."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    sdir = _make_search_golden_dir(tmp_path)
    out_path = sdir / "search_results.jsonl"

    import re_score as rs

    rs.run_search_over_golden(
        queries_path=str(sdir / "queries.jsonl"),
        attributes_path=str(sdir / "attributes.jsonl"),
        model=None,  # dictionary path — no GPU
    )

    assert out_path.exists(), "search_results.jsonl was not written"

    with open(out_path) as f:
        results = [json.loads(l) for l in f if l.strip()]

    assert len(results) == 1
    result = results[0]
    assert result["query"] == "검은색 차"

    ranked = result["ranked"]
    assert "black_car" in ranked, f"black_car missing from ranked: {ranked}"
    assert "red_car" not in ranked, f"red_car should be excluded by hard filter: {ranked}"
