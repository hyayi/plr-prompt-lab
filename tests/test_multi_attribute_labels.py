"""Multi-attribute labels: one crop set, one model run, N attribute evals.

labels.jsonl의 다속성 형식 {"obj_id", "labels": {attr: label}}과
manifest `attributes:` 맵이 파이프라인 전체(로더→eval→validate→gallery)에서
동작하는지 검증한다. 핵심 계약:
  1. load_labels는 단일(legacy)/다속성 두 형식을 모두 읽고, 요청 속성
     라벨이 없는 행은 제외한다 (미라벨 ≠ unknown).
  2. run_eval은 predictions.jsonl의 attribute 스탬프가 요청 속성과 다르면
     attributes.jsonl(전체 plr_json 캐시)에서 pred/margin을 재추출한다
     — 모델 1회 실행으로 전 속성 평가.
  3. validate는 선언 안 된 속성 키·vocab 밖 라벨을 error로 잡는다.

No GPU, no DB, no Redis.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from evalkit.dataset import attribute_spec, declared_attributes, load_labels  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _plr_json(gender: str, equip_type: str | None) -> dict:
    equipment = [{"type": equip_type, "score": 0.9}] if equip_type else []
    return {
        "object_type": "person",
        "attributes": {
            "gender_scores": {"male": 0.0, "female": 0.0, "selected": gender,
                              "decision_margin": 0.8},
            "equipment": equipment,
        },
    }


def _make_multi_dataset(base: Path) -> Path:
    """gender(프리셋) + helmet(커스텀) 2속성 데이터셋. 모델 실행은 이미 끝난
    상태를 흉내낸다: predictions.jsonl은 gender 추출물(스탬프 포함),
    attributes.jsonl은 크롭당 plr_json 전체."""
    from PIL import Image

    base.mkdir(parents=True, exist_ok=True)
    (base / "crops").mkdir(exist_ok=True)
    for oid in ("a", "b", "c"):
        Image.new("RGB", (60, 90), (100, 100, 100)).save(str(base / "crops" / f"{oid}.jpg"))
    (base / "manifest.yaml").write_text(
        "n: 3\ncreated: '2026-07-03'\nsource_note: test\n"
        "attributes:\n"
        "  gender: {}\n"
        "  helmet:\n"
        "    labels: [helmet, no_helmet]\n"
        "    pred_path: attributes.equipment[0].type\n",
        encoding="utf-8",
    )
    _write_jsonl(base / "labels.jsonl", [
        {"obj_id": "a", "labels": {"gender": "male", "helmet": "helmet"}},
        {"obj_id": "b", "labels": {"gender": "female", "helmet": "no_helmet"}},
        {"obj_id": "c", "labels": {"helmet": "helmet"}},  # gender 미라벨
    ])
    _write_jsonl(base / "predictions.jsonl", [
        {"obj_id": "a", "attribute": "gender", "pred": "male", "margin": 0.9, "quality": 0.8},
        {"obj_id": "b", "attribute": "gender", "pred": "female", "margin": 0.7, "quality": 0.6},
        {"obj_id": "c", "attribute": "gender", "pred": "male", "margin": 0.5, "quality": 0.4},
    ])
    _write_jsonl(base / "attributes.jsonl", [
        {"obj_id": "a", "plr_json": _plr_json("male", "helmet")},
        {"obj_id": "b", "plr_json": _plr_json("female", None)},   # equipment 없음 → pred None
        {"obj_id": "c", "plr_json": _plr_json("male", "helmet")},
    ])
    return base


def test_load_labels_both_shapes(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "labels.jsonl", [
        {"obj_id": "x", "label": "male"},                       # legacy 단일
        {"obj_id": "y", "labels": {"gender": "female", "helmet": "helmet"}},
        {"obj_id": "z", "labels": {"helmet": "no_helmet"}},     # gender 없음
    ])
    g = load_labels(tmp_path, "gender")
    assert g == {"x": "male", "y": "female"}, "legacy 행은 모든 속성에, dict 행은 키 있을 때만"
    h = load_labels(tmp_path, "helmet")
    assert h == {"x": "male", "y": "helmet", "z": "no_helmet"}


def test_declared_attributes_and_spec(tmp_path: Path) -> None:
    ds = _make_multi_dataset(tmp_path / "ds")
    assert declared_attributes(ds) == ["gender", "helmet"]
    spec = attribute_spec(ds, "helmet")
    assert spec["pred_path"] == "attributes.equipment[0].type"
    assert list(spec["labels"]) == ["helmet", "no_helmet"]
    # 프리셋 속성은 맵의 빈 dict로도 프리셋 스펙 유지
    assert attribute_spec(ds, "gender")["pred_path"] == "attributes.gender_scores.selected"


def test_validate_multi_attribute(tmp_path: Path) -> None:
    from evalkit.validate import validate_dataset

    ds = _make_multi_dataset(tmp_path / "ok")
    assert validate_dataset(ds, verbose=False) is True

    bad = _make_multi_dataset(tmp_path / "bad")
    _write_jsonl(bad / "labels.jsonl", [
        {"obj_id": "a", "labels": {"gendre": "male"}},          # 오타 속성 키
        {"obj_id": "b", "labels": {"helmet": "purple"}},        # vocab 밖 라벨
    ])
    assert validate_dataset(bad, verbose=False) is False


def test_eval_all_skips_declared_but_unlabeled(tmp_path: Path) -> None:
    """"라벨이 실제로 있는 속성"만 평가 대상: 선언만 되고 라벨 없는 속성은 skipped.
    (lab eval CLI 제거 후 — 선택 로직 eval_attributes는 서버가 공유하는
    evalkit/dataset.py에 유지, 여기서 직접 검증.)"""
    from evalkit.dataset import eval_attributes

    ds = _make_multi_dataset(tmp_path / "ds")
    mp = ds / "manifest.yaml"
    mp.write_text(mp.read_text(encoding="utf-8") + "  military: {}\n", encoding="utf-8")

    attributes, skipped, undeclared = eval_attributes(ds)
    assert set(attributes) == {"gender", "helmet"}
    assert "military" in skipped
    assert undeclared == []


class _HintAwareModel:
    """받은 프롬프트가 person용인지 vehicle용인지에 따라 다른 YAML을 반환하고,
    어떤 종류를 받았는지 기록한다 — 크롭별 힌트 라우팅의 검증 장치."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate(self, messages, image):  # noqa: ARG002
        user_text = " ".join(
            c.get("text", "") for m in messages
            for c in (m["content"] if isinstance(m["content"], list) else [])
        )
        if "gender" in user_text:
            self.seen.append("person")
            return ("target: person\ngender: female\nage: adult\noutfit: two_piece\n"
                    "upper:\n  color: black\n  type: jacket\n"
                    "lower:\n  color: black\n  type: pants\naction: standing\n"
                    "military: civilian\nmargins: {gender: 0.9}")
        self.seen.append("vehicle")
        return ("target: vehicle\nvehicle_type: sedan\ncolor: white\n"
                "state: parked\nmilitary: civilian\nmargins: {vehicle_type: 0.9}")


def test_mixed_dataset_routes_prompt_per_crop(tmp_path: Path) -> None:
    """labels.jsonl 행의 object_type이 크롭별 person/vehicle 프롬프트를
    결정해야 한다 (데이터셋 단위 힌트 일괄 적용이면 이 테스트가 죽는다)."""
    from PIL import Image

    from runners import re_score as rs

    ds = tmp_path / "mixed"
    (ds / "crops").mkdir(parents=True)
    for oid in ("p1", "v1"):
        Image.new("RGB", (100, 150), (120, 120, 120)).save(str(ds / "crops" / f"{oid}.jpg"))
    (ds / "manifest.yaml").write_text(
        "n: 2\ncreated: '2026-07-03'\nsource_note: test\n"
        "attributes:\n  gender: {}\n  vehicle_type: {}\n",
        encoding="utf-8",
    )
    _write_jsonl(ds / "labels.jsonl", [
        {"obj_id": "p1", "object_type": "person", "labels": {"gender": "female"}},
        {"obj_id": "v1", "object_type": "vehicle", "labels": {"vehicle_type": "sedan"}},
    ])

    model = _HintAwareModel()
    # attribute=gender → 데이터셋 단위 힌트는 person이지만, v1 행의
    # object_type=vehicle이 크롭 단위로 이겨야 한다.
    rs.re_score("gender", model, golden_dir=str(ds))

    assert sorted(model.seen) == ["person", "vehicle"], (
        f"두 크롭이 서로 다른 프롬프트를 받아야 하는데: {model.seen}"
    )
    attrs = {json.loads(l)["obj_id"]: json.loads(l)["plr_json"]["object_type"]
             for l in open(ds / "attributes.jsonl", encoding="utf-8")}
    assert attrs == {"p1": "person", "v1": "vehicle"}
