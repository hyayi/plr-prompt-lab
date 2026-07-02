"""Variant bundle — one prompts/<V>.yaml versions the whole input combination.

Knobs: prompt templates (existing), `enums:` overrides (provider),
`preprocess.marker` (image pre-processing), `sampling:` (model params).
`prompt_hash` hashes prompts/*.yaml, so any knob change re-stamps provenance
automatically.

No GPU, no DB, no Redis.
"""
from __future__ import annotations

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
      gender: 0.8
      age: 1.0
      outfit: 0.8
""")


class _SamplingAwareMock:
    """Mock with LabGemmaModel-like sampling attributes."""

    def __init__(self) -> None:
        self.max_tokens = 512
        self.temperature = 0.0

    def generate(self, messages, image):  # noqa: ARG002
        return _MOCK_PLR_YAML


def _write_variant_yaml(version: str) -> Path:
    """Derive a variant from plr_v1.5_cot with every knob overridden."""
    import yaml

    base = yaml.safe_load(
        (_LAB_ROOT / "prompts" / "plr_v1.5_cot.yaml").read_text(encoding="utf-8"))
    base["enums"] = {"colors": ["black", "white", "red"]}
    base["preprocess"] = {"marker": False}
    base["sampling"] = {"max_tokens": 256, "temperature": 0.2}
    out = _LAB_ROOT / "prompts" / f"{version}.yaml"
    out.write_text(yaml.dump(base, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    return out


def _make_ds(base: Path) -> Path:
    (base / "crops").mkdir(parents=True)
    with open(base / "labels.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"obj_id": "v1", "label": "female"}) + "\n")
    Image.new("RGB", (100, 150), (128, 128, 128)).save(
        str(base / "crops" / "v1.jpg"), format="JPEG")
    return base


def test_variant_bundle_applies_all_knobs(tmp_path: Path, monkeypatch) -> None:
    from providers.file_prompt_provider import FilePromptProvider
    from runners import re_score as rs
    import plr_core

    version = "test_variant"
    yaml_path = _write_variant_yaml(version)
    try:
        # 1. enum override lands in the built prompt
        monkeypatch.setenv("IR_PLR_REASON", "on")
        text = FilePromptProvider(version_override=version).build_plr_messages(
            "person")[1]["content"][1]["text"]
        assert "- color: black, white, red\n" in text, "enum override not injected"

        # 2+3. marker skipped + sampling forwarded, via a real re_score run
        marker_calls: list = []
        real_marker = plr_core._draw_target_marker
        monkeypatch.setattr(plr_core, "_draw_target_marker",
                            lambda pil: marker_calls.append(1) or real_marker(pil))
        model = _SamplingAwareMock()
        rs.re_score("gender", model, golden_dir=str(_make_ds(tmp_path / "ds")),
                    prompt_version=version)
        assert marker_calls == [], "preprocess.marker=false must skip drawing"
        assert model.max_tokens == 256 and model.temperature == 0.2, (
            "sampling knobs not forwarded to the model"
        )
    finally:
        yaml_path.unlink()


def test_default_variant_keeps_production_behaviour(tmp_path: Path, monkeypatch) -> None:
    """plr_v1.5_cot (no variant keys): marker drawn, sampling untouched."""
    from runners import re_score as rs
    import plr_core

    marker_calls: list = []
    real_marker = plr_core._draw_target_marker
    monkeypatch.setattr(plr_core, "_draw_target_marker",
                        lambda pil: marker_calls.append(1) or real_marker(pil))
    model = _SamplingAwareMock()
    rs.re_score("gender", model, golden_dir=str(_make_ds(tmp_path / "ds")),
                prompt_version="plr_v1.5_cot")
    assert marker_calls == [1], "default variant must draw the marker"
    assert model.max_tokens == 512 and model.temperature == 0.0


def test_variant_changes_prompt_hash(tmp_path: Path) -> None:
    """A knob change re-stamps provenance (prompt_hash covers the yaml)."""
    from evalkit.provenance import prompt_hash

    before = prompt_hash(_LAB_ROOT)
    yaml_path = _write_variant_yaml("test_varhash")
    try:
        after = prompt_hash(_LAB_ROOT)
        assert before != after, "variant yaml must change the prompt surface hash"
    finally:
        yaml_path.unlink()
