"""format/reason experiment axes (IR_PLR_FORMAT / IR_PLR_REASON).

Optional `formats:` / `reasons:` keys in experiment.yaml cross into per-cell
env, stamp distinguishable ledger version tags, never leak env across cells,
and fail loudly (per cell) when the format axis conflicts with a yaml-pinned
prompt version.

(The search-pipeline wiring tests that used to live alongside these were
removed with the text-search pipeline — the lab is PLR-only since 2026-07.)

No GPU, no DB, no Redis.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest
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
      gender: 0.8
      age: 1.0
      outfit: 0.8
""")


class EnvRecordingModel:
    """Model stub that records the IR_PLR_FORMAT/IR_PLR_REASON env seen at
    generate() time (i.e. what the cell actually ran under)."""

    def __init__(self, sink: list[tuple[str | None, str | None]]) -> None:
        self._sink = sink

    def generate(self, messages, image):  # noqa: ARG002
        self._sink.append((os.environ.get("IR_PLR_FORMAT"),
                           os.environ.get("IR_PLR_REASON")))
        return _MOCK_PLR_YAML


def _make_gender_dataset(base: Path, obj_ids: list[str]) -> Path:
    crops = base / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    with open(base / "predictions.jsonl", "w", encoding="utf-8") as f:
        for oid in obj_ids:
            f.write(json.dumps({"obj_id": oid, "pred": "unknown", "reason": ""}) + "\n")
    with open(base / "labels.jsonl", "w", encoding="utf-8") as f:
        for oid in obj_ids:
            f.write(json.dumps({"obj_id": oid, "label": "female"}) + "\n")
    for oid in obj_ids:
        Image.new("RGB", (100, 150), (128, 128, 128)).save(
            str(crops / f"{oid}.jpg"), format="JPEG")
    return base


def test_enumerate_cells_crosses_reason_axis() -> None:
    """plr cells cross the reasons axis (formats died with the JSON path)."""
    from runners import experiment as ex

    cfg = {
        "datasets": ["d"], "models": ["mock"], "prompts": ["p"],
        "pipelines": ["plr"], "attributes": ["gender"],
        "reasons": ["on", "off"],
    }
    cells = ex.enumerate_cells(cfg)
    assert len(cells) == 2, f"expected 2 cells, got {len(cells)}"
    assert {c.reason for c in cells} == {"on", "off"}
    assert {c.version_tag() for c in cells} == {"p+reason-on", "p+reason-off"}


def test_validate_schema_rejects_bad_axis_values() -> None:
    from runners import experiment as ex

    base = {"datasets": ["d"], "models": ["m"], "prompts": ["p"], "pipelines": ["plr"]}
    with pytest.raises(ValueError, match="reasons"):
        ex._validate_schema({**base, "reasons": ["maybe"]}, "x.yaml")
    # the formats axis was removed with the legacy JSON prompt path
    with pytest.raises(ValueError, match="removed"):
        ex._validate_schema({**base, "formats": ["yaml"]}, "x.yaml")


def test_experiment_reason_axis_sets_env_and_stamps_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end matrix with reasons [on, off]: each cell runs under its own
    IR_PLR_REASON, the ledger stamps distinct version tags, and the env is
    restored after the matrix."""
    import registry
    from runners import experiment

    ds_dir = _make_gender_dataset(tmp_path / "ds", ["m1", "m2"])
    ledger_path = tmp_path / "ledger.jsonl"

    seen_env: list[tuple[str | None, str | None]] = []
    monkeypatch.setitem(registry.MODELS, "envrec", lambda: EnvRecordingModel(seen_env))
    monkeypatch.setenv("IR_PLR_REASON", "sentinel")  # must be restored afterwards

    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(textwrap.dedent(f"""\
        datasets:
          - {ds_dir}
        models:
          - envrec
        prompts:
          - plr_v1.5_cot
        pipelines:
          - plr
        attributes:
          - gender
        reasons:
          - "on"
          - "off"
        ledger: {ledger_path}
        """), encoding="utf-8")

    exit_code = experiment.run_experiment(str(yaml_path))
    assert exit_code == 0, f"expected exit 0, got {exit_code}"

    # 2 cells × 2 crops = 4 generate calls; first two under on, last two off.
    reasons_seen = [r for _f, r in seen_env]
    assert reasons_seen == ["on", "on", "off", "off"], (
        f"cells did not run under their own IR_PLR_REASON: {reasons_seen}"
    )
    assert os.environ.get("IR_PLR_REASON") == "sentinel", (
        "matrix leaked IR_PLR_REASON instead of restoring the pre-matrix env"
    )

    with open(ledger_path, encoding="utf-8") as f:
        versions = {json.loads(line)["version"] for line in f if line.strip()}
    assert versions == {"plr_v1.5_cot+reason-on", "plr_v1.5_cot+reason-off"}, (
        f"ledger version tags must distinguish the reason axis, got {versions}"
    )
