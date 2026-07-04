"""Generic, manifest-driven datasets — the label SET is the author's choice.

The dataset template fixes the STRUCTURE (crops/ + labels.jsonl +
manifest.yaml); a dataset may declare its own attribute via manifest.yaml
(`labels`, `pred_path`, optional `margin_path` / `bias_pair` /
`object_type_hint`). gender / vehicle_type / military stay as built-in
presets and reference examples.

No GPU, no DB, no Redis.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest
from PIL import Image

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

# Mock person PLR with a helmet in equipment — the custom attribute's pred
# lives at attributes.equipment[0].type after parsing.
_MOCK_PLR_YAML = textwrap.dedent("""\
    target: person
    gender: male
    gender_reason: short hair
    age: adult
    outfit: two_piece
    upper.color: black
    upper.type: jacket
    lower.color: black
    lower.type: pants
    equipment: [helmet]
    action: standing
    military: civilian
    margins:
      gender: 0.9
      age: 1.0
      outfit: 0.8
""")


class _MockModel:
    def generate(self, messages, image):  # noqa: ARG002
        return _MOCK_PLR_YAML


def _make_helmet_dataset(base: Path, obj_ids: list[str]) -> Path:
    crops = base / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    (base / "manifest.yaml").write_text(textwrap.dedent("""\
        attribute: helmet
        n: 2
        created: "2026-07-02"
        source_note: synthetic generic-dataset test
        labels: [helmet, no_helmet]
        pred_path: attributes.equipment[0].type
        bias_pair: [no_helmet, helmet]
        object_type_hint: person
        """), encoding="utf-8")
    with open(base / "labels.jsonl", "w", encoding="utf-8") as f:
        for oid, lab in zip(obj_ids, ["helmet", "no_helmet"]):
            f.write(json.dumps({"obj_id": oid, "label": lab}) + "\n")
    for oid in obj_ids:
        Image.new("RGB", (100, 150), (128, 128, 128)).save(
            str(crops / f"{oid}.jpg"), format="JPEG")
    return base


def test_resolve_json_path() -> None:
    from evalkit.dataset import resolve_json_path

    data = {"attributes": {"equipment": [{"type": "helmet", "score": 1.0}],
                           "gender_scores": {"selected": "male"}}}
    assert resolve_json_path(data, "attributes.equipment[0].type") == "helmet"
    assert resolve_json_path(data, "attributes.gender_scores.selected") == "male"
    assert resolve_json_path(data, "attributes.equipment[5].type") is None
    assert resolve_json_path(data, "attributes.nope.x") is None


def test_custom_attribute_re_score_extracts_pred(tmp_path: Path) -> None:
    """helmet dataset: 클라이언트 re_score 가 manifest pred_path 로 예측을 추출해
    predictions.jsonl 에 쓴다 (채점은 서버 레포 소관 — 여기선 추출만 검증)."""
    from runners import re_score as rs

    gdir = _make_helmet_dataset(tmp_path / "ds", ["h1", "h2"])
    rs.re_score("helmet", _MockModel(), golden_dir=str(gdir))

    rows = [json.loads(l) for l in open(gdir / "predictions.jsonl", encoding="utf-8")]
    assert all(r["pred"] == "helmet" for r in rows), rows


def test_custom_attribute_without_pred_path_fails_loud(tmp_path: Path) -> None:
    from runners import re_score as rs

    gdir = tmp_path / "ds"
    (gdir / "crops").mkdir(parents=True)
    Image.new("RGB", (100, 150)).save(str(gdir / "crops" / "x1.jpg"), format="JPEG")
    (gdir / "manifest.yaml").write_text(
        "attribute: helmet\nn: 1\ncreated: '2026-07-02'\nsource_note: t\n",
        encoding="utf-8")

    with pytest.raises(ValueError, match="pred_path"):
        rs.re_score("helmet", _MockModel(), golden_dir=str(gdir))


def test_validate_uses_manifest_labels(tmp_path: Path, capsys) -> None:
    """Declared labels replace the preset vocabulary; out-of-set labels fail."""
    from evalkit.validate import validate_dataset

    gdir = _make_helmet_dataset(tmp_path / "ds", ["h1", "h2"])
    assert validate_dataset(str(gdir)) is True
    capsys.readouterr()

    # An out-of-vocabulary label must now be an error against the DECLARED set.
    with open(gdir / "labels.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"obj_id": "h1", "label": "banana"}) + "\n")
    assert validate_dataset(str(gdir)) is False
    out = capsys.readouterr().out
    assert "banana" in out
