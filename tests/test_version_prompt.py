"""Prompt-axis tests — prove that `prompt_version` genuinely changes the prompt.

Before this feature `re_score` always built the MAIN PLR prompt from the
module-level constants (chosen by IR_PLR_FORMAT / IR_PLR_REASON) and ignored the
version, so two experiment cells with `prompts: [plr_v1.3_cot, plr_v1.4_cot]`
sent the SAME prompt and only the ledger label differed. These tests lock in the
fix:

  1. re_score(..., prompt_version="plr_v1.3_cot") vs "plr_v1.4_cot" send
     DIFFERENT user prompts on the SAME dataset (the whole point). Both also
     differ from "plr_v0.4".
  2. A yaml-less version ("mock_v1") falls back to the constants without error
     (build_messages None path).
  3. run_plr(..., build_messages=None) output is identical to the old default
     (no regression / byte-identical to the live path).
  4. End-to-end matrix: `run_experiment` with prompts [plr_v1.3_cot,
     plr_v1.4_cot] → the two cells' recorded prompts differ AND the ledger has
     2 records stamped with the two distinct versions.

No GPU, no DB, no Redis.  For the person (gender) attribute the CoT person
prompt differs between v1.3 and v1.4 only when IR_PLR_REASON=on (the plain
no-reason person template is shared), so these tests set IR_PLR_REASON=on to
exercise the divergent CoT templates — exactly the axis an experiment compares.
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
# Helpers
# =====================================================================

# Minimal valid PLR YAML for a person, gender=female (shared with the other
# lab-runner tests). RecordingMockModel returns this on every .generate.
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


def _tiny_jpg(path: Path, rgb: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Write a tiny 100×150 JPEG (large enough for quality_gate normal_plr mode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 150), rgb).save(str(path), format="JPEG")


def _make_golden_dir(base: Path, obj_ids: list[str]) -> Path:
    """Create a minimal gender golden dir (crops + predictions + labels)."""
    gdir = base
    crops = gdir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    with open(gdir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for oid in obj_ids:
            f.write(json.dumps({"obj_id": oid, "pred": "unknown", "reason": ""}) + "\n")
    with open(gdir / "labels.jsonl", "w", encoding="utf-8") as f:
        for oid in obj_ids:
            f.write(json.dumps({"obj_id": oid, "label": "female"}) + "\n")
    for oid in obj_ids:
        _tiny_jpg(crops / f"{oid}.jpg")
    return gdir


class RecordingMockModel:
    """Deterministic Model stub that RECORDS every `messages` it receives.

    Satisfies the Model protocol: generate(messages, image) -> str. Returns the
    canned PLR YAML (ignoring the image) so re_score parses a valid person PLR,
    while capturing the messages so a test can assert what prompt was sent.
    """

    def __init__(self, yaml_text: str = _MOCK_PLR_YAML) -> None:
        self._yaml = yaml_text
        self.calls: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]], image: Any) -> str:  # noqa: ARG002
        self.calls.append(messages)
        return self._yaml


def _user_text(messages: list[dict[str, Any]]) -> str:
    """Extract the concatenated user-role text from an OpenAI-style message list."""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    parts.append(str(chunk.get("text", "")))
    return "\n".join(parts)


def _first_user_text(rec: RecordingMockModel) -> str:
    assert rec.calls, "RecordingMockModel captured no .generate calls"
    return _user_text(rec.calls[0])


@pytest.fixture(autouse=True)
def _reason_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force IR_PLR_REASON=on so the CoT person prompt (which differs between
    v1.3 and v1.4) is exercised. Without this the plain no-reason person template
    is shared across those two versions."""
    monkeypatch.setenv("IR_PLR_REASON", "on")


# =====================================================================
# Test 1: versions genuinely send DIFFERENT prompts
# =====================================================================


def test_prompt_versions_produce_different_prompts(tmp_path: Path) -> None:
    """re_score with plr_v1.3_cot vs plr_v1.4_cot vs plr_v0.4 sends distinct
    user prompts on the SAME dataset — the whole point of the prompt axis."""
    import re_score as rs

    obj_ids = ["p1", "p2"]

    rec13 = RecordingMockModel()
    rs.re_score("gender", rec13,
                golden_dir=str(_make_golden_dir(tmp_path / "v13", obj_ids)),
                prompt_version="plr_v1.3_cot")

    rec14 = RecordingMockModel()
    rs.re_score("gender", rec14,
                golden_dir=str(_make_golden_dir(tmp_path / "v14", obj_ids)),
                prompt_version="plr_v1.4_cot")

    rec04 = RecordingMockModel()
    rs.re_score("gender", rec04,
                golden_dir=str(_make_golden_dir(tmp_path / "v04", obj_ids)),
                prompt_version="plr_v0.4")

    t13 = _first_user_text(rec13)
    t14 = _first_user_text(rec14)
    t04 = _first_user_text(rec04)

    assert t13 != t14, (
        "plr_v1.3_cot and plr_v1.4_cot produced IDENTICAL user prompts — "
        "the prompt axis is not actually changing the prompt.\n"
        f"len(v1.3)={len(t13)} len(v1.4)={len(t14)}"
    )
    assert t13 != t04, "plr_v1.3_cot and plr_v0.4 produced identical prompts"
    assert t14 != t04, "plr_v1.4_cot and plr_v0.4 produced identical prompts"


# =====================================================================
# Test 2: yaml-less version → constants fallback (build_messages None path)
# =====================================================================


def test_unknown_version_falls_back_to_constants(tmp_path: Path) -> None:
    """A version with no prompts/<v>.yaml (e.g. 'mock_v1') falls back to the
    module constants without error and matches prompt_version=None exactly."""
    import re_score as rs

    obj_ids = ["a1", "a2"]

    assert not (_LAB_ROOT / "prompts" / "mock_v1.yaml").exists(), (
        "test assumes mock_v1 has no backing yaml"
    )

    rec_mock = RecordingMockModel()
    rs.re_score("gender", rec_mock,
                golden_dir=str(_make_golden_dir(tmp_path / "mock", obj_ids)),
                prompt_version="mock_v1")

    rec_none = RecordingMockModel()
    rs.re_score("gender", rec_none,
                golden_dir=str(_make_golden_dir(tmp_path / "none", obj_ids)),
                prompt_version=None)

    assert _first_user_text(rec_mock) == _first_user_text(rec_none), (
        "yaml-less version 'mock_v1' should fall back to the same constants "
        "prompt as prompt_version=None"
    )


# =====================================================================
# Test 3: run_plr(build_messages=None) is byte-identical to the old default
# =====================================================================


def test_run_plr_build_messages_none_matches_constants() -> None:
    """run_plr with build_messages=None must send exactly plr_prompts'
    build_plr_messages output — no regression vs the live path."""
    import plr_core
    from plr_prompts import build_plr_messages

    class _QReport:
        mode = "normal_plr"

    rec = RecordingMockModel()
    pil = Image.new("RGB", (100, 150), (128, 128, 128))
    plr_core.run_plr(pil, _QReport(), rec, object_type_hint="person",
                     build_messages=None)

    assert rec.calls, "run_plr did not call model.generate"
    assert rec.calls[0] == build_plr_messages("person"), (
        "run_plr(build_messages=None) diverged from plr_prompts.build_plr_messages"
    )


# =====================================================================
# Test 4: end-to-end matrix — two prompt cells differ + 2 ledger records
# =====================================================================


def test_experiment_matrix_prompts_differ_per_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_experiment` with prompts [plr_v1.3_cot, plr_v1.4_cot] → the two
    cells' recorded prompts DIFFER and the ledger stamps two distinct versions."""
    import registry
    import experiment

    # Build a gender dataset.
    ds_dir = _make_golden_dir(tmp_path / "ds", ["m1", "m2", "m3"])
    ledger_path = tmp_path / "ledger.jsonl"

    # Register a recording model whose every constructed instance is captured,
    # so we can inspect the prompt each cell sent. Each cell calls get_model()
    # afresh → a fresh RecordingMockModel appended to `instances`.
    instances: list[RecordingMockModel] = []

    def _factory() -> RecordingMockModel:
        m = RecordingMockModel()
        instances.append(m)
        return m

    monkeypatch.setitem(registry.MODELS, "recording", _factory)

    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(textwrap.dedent(f"""\
        datasets:
          - {str(ds_dir)}
        models:
          - recording
        prompts:
          - plr_v1.3_cot
          - plr_v1.4_cot
        pipelines:
          - plr
        attributes:
          - gender
        ledger: {str(ledger_path)}
        """), encoding="utf-8")

    exit_code = experiment.run_experiment(str(yaml_path))
    assert exit_code == 0, f"expected exit 0, got {exit_code}"

    # Two cells → two recording instances that captured at least one call each.
    recorded = [m for m in instances if m.calls]
    assert len(recorded) == 2, (
        f"expected 2 cells to record prompts, got {len(recorded)}"
    )
    t_a = _first_user_text(recorded[0])
    t_b = _first_user_text(recorded[1])
    assert t_a != t_b, (
        "the two prompt-version cells sent IDENTICAL prompts — the prompt axis "
        "is a no-op in the matrix runner"
    )

    # Ledger: 2 records stamped with the two distinct prompt versions.
    with open(ledger_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2, f"expected 2 ledger records, got {len(records)}: {records}"
    versions = {r["version"] for r in records}
    assert versions == {"plr_v1.3_cot", "plr_v1.4_cot"}, (
        f"ledger versions should be the two prompt versions, got {versions}"
    )
