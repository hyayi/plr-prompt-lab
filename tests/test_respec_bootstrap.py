"""Regression: re_score must score a spec-only dataset (crops + labels, no
predictions.jsonl bootstrap).

DATASET_SPEC defines a dataset as crops/ + labels.jsonl (+ queries + manifest);
predictions.jsonl is a GENERATED artifact, not a required input. The
prepare-dataset "arbitrary crops" path produces exactly such a dataset. re_score
therefore derives the obj_id set from predictions.jsonl if present, else from
crops/, else labels.jsonl — so the experiment matrix can run on a recipient's
own dataset.
"""
import json
from pathlib import Path

from PIL import Image

import re_score
from registry import get_model


def _spec_only_dataset(tmp_path):
    ds = tmp_path / "ds"
    (ds / "crops").mkdir(parents=True)
    for oid in ["a1", "a2", "a3"]:
        Image.new("RGB", (32, 32)).save(ds / "crops" / f"{oid}.jpg")
    (ds / "labels.jsonl").write_text(
        "".join(f'{{"obj_id": "{o}", "label": "female"}}\n' for o in ["a1", "a2", "a3"])
    )
    (ds / "manifest.yaml").write_text(
        "attribute: gender\nn: 3\ncreated: 2026-07-02\nsource_note: test\n"
    )
    return ds


def test_re_score_derives_obj_ids_from_crops_without_predictions(tmp_path):
    ds = _spec_only_dataset(tmp_path)
    assert not (ds / "predictions.jsonl").exists()  # spec-only: no bootstrap

    meta = re_score.re_score("gender", get_model("mock"), golden_dir=str(ds))

    preds = {
        json.loads(l)["obj_id"]
        for l in (ds / "predictions.jsonl").read_text().splitlines()
        if l.strip()
    }
    assert preds == {"a1", "a2", "a3"}, "obj_ids not derived from crops/"
    assert meta["n"] == 3
