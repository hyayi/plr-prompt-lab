"""Experiment configs — configs/<name>.yaml crosses a prompt version with
input knobs WITHOUT copying template text (components referenced by path).

`prompts/<V>.yaml` stays a pure prompt version; a config references it
(`prompt: prompts/plr_v1.5_cot.yaml`) and adds `enums:`/`preprocess:`/`sampling:`.
`lab run --version <config-name>` resolves the combination, the ledger
stamps the config name, and prompt_hash covers configs/*.yaml so any knob
change re-stamps provenance.

No GPU, no DB, no Redis.
"""
from __future__ import annotations

import json
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


class _SamplingAwareMock:
    """Mock with LabGemmaModel-like sampling attributes."""

    def __init__(self) -> None:
        self.max_tokens = 512
        self.temperature = 0.0

    def generate(self, messages, image):  # noqa: ARG002
        return _MOCK_PLR_YAML


def _write_config(name: str, body: str) -> Path:
    vdir = _LAB_ROOT / "configs"
    vdir.mkdir(exist_ok=True)
    out = vdir / f"{name}.yaml"
    out.write_text(textwrap.dedent(body), encoding="utf-8")
    return out


def _make_ds(base: Path) -> Path:
    (base / "crops").mkdir(parents=True)
    with open(base / "labels.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"obj_id": "v1", "label": "female"}) + "\n")
    Image.new("RGB", (100, 150), (128, 128, 128)).save(
        str(base / "crops" / "v1.jpg"), format="JPEG")
    return base


def test_config_composition_applies_all_knobs(tmp_path: Path, monkeypatch) -> None:
    """A config referencing prompts/plr_v1.5_cot.yaml (path form) applies
    enum/marker/sampling knobs without any copy of the template text."""
    from runners import re_score as rs
    import plr_core

    path = _write_config("test_combo", """\
        prompt: prompts/plr_v1.5_cot.yaml
        enums:
          colors: [black, white, red]
        preprocess:
          marker: false
        sampling:
          max_tokens: 256
          temperature: 0.2
        """)
    try:
        monkeypatch.setenv("IR_PLR_REASON", "on")
        marker_calls: list = []
        real_marker = plr_core._draw_target_marker
        monkeypatch.setattr(plr_core, "_draw_target_marker",
                            lambda pil: marker_calls.append(1) or real_marker(pil))
        # Record the prompt actually sent (enum override must be inside).
        sent: list = []
        model = _SamplingAwareMock()
        real_gen = model.generate
        model.generate = lambda msgs, img: (sent.append(msgs), real_gen(msgs, img))[1]

        meta = rs.re_score("gender", model,
                           golden_dir=str(_make_ds(tmp_path / "ds")),
                           prompt_version="test_combo")

        text = sent[0][1]["content"][1]["text"]
        assert "- color: black, white, red\n" in text, "enum override not injected"
        assert marker_calls == [], "preprocess.marker=false must skip drawing"
        assert model.max_tokens == 256 and model.temperature == 0.2
        assert meta["version"] == "test_combo", "ledger tag must be the config name"
    finally:
        path.unlink()


def test_same_prompt_two_configs_differ_only_by_knobs(tmp_path: Path, monkeypatch) -> None:
    """Two configs over the SAME prompt version — no template copies, prompts
    differ exactly by the overridden enum list."""
    from providers.file_prompt_provider import FilePromptProvider
    from runners.exp_config import load_config

    a = _write_config("test_va", "prompt: plr_v1.5_cot\n")
    b = _write_config("test_vb", """\
        prompt: plr_v1.5_cot
        enums:
          colors: [red, blue]
        """)
    try:
        monkeypatch.setenv("IR_PLR_REASON", "on")
        va, vb = load_config(_LAB_ROOT, "test_va"), load_config(_LAB_ROOT, "test_vb")
        ta = FilePromptProvider(version_override=va.prompt,
                                enum_overrides=va.enums).build_plr_messages("person")
        tb = FilePromptProvider(version_override=vb.prompt,
                                enum_overrides=vb.enums).build_plr_messages("person")
        sa, sb = ta[1]["content"][1]["text"], tb[1]["content"][1]["text"]
        assert sa != sb
        assert "- color: red, blue\n" in sb and "- color: red, blue\n" not in sa
    finally:
        a.unlink(); b.unlink()


def test_variant_dangling_prompt_fails_loud() -> None:
    from runners.exp_config import load_config

    path = _write_config("test_dangling", "prompt: no_such_version\n")
    try:
        with pytest.raises(ValueError, match="dangling"):
            load_config(_LAB_ROOT, "test_dangling")
    finally:
        path.unlink()


def test_default_prompt_version_keeps_production_behaviour(tmp_path: Path, monkeypatch) -> None:
    """Plain prompt version (no config): marker drawn, sampling untouched."""
    from runners import re_score as rs
    import plr_core

    marker_calls: list = []
    real_marker = plr_core._draw_target_marker
    monkeypatch.setattr(plr_core, "_draw_target_marker",
                        lambda pil: marker_calls.append(1) or real_marker(pil))
    model = _SamplingAwareMock()
    rs.re_score("gender", model, golden_dir=str(_make_ds(tmp_path / "ds")),
                prompt_version="plr_v1.5_cot")
    assert marker_calls == [1], "default must draw the marker"
    assert model.max_tokens == 512 and model.temperature == 0.0


def test_config_changes_prompt_hash() -> None:
    """A new/changed experiment config re-stamps provenance."""
    from evalkit.provenance import prompt_hash

    before = prompt_hash(_LAB_ROOT)
    path = _write_config("test_hash", "prompt: plr_v1.5_cot\n")
    try:
        assert prompt_hash(_LAB_ROOT) != before, "configs/ must be hashed"
    finally:
        path.unlink()


def test_enum_override_extension_fails_loud() -> None:
    """Config enums may only NARROW: extending past the schema vocabulary is
    rejected (the parser would coerce the new answers away)."""
    from runners.exp_config import load_config

    path = _write_config("test_extend", """\
        prompt: plr_v1.5_cot
        enums:
          colors: [black, crimson]
        """)
    try:
        with pytest.raises(ValueError, match="EXTENDS"):
            load_config(_LAB_ROOT, "test_extend")
    finally:
        path.unlink()


def test_enum_override_unknown_key_fails_loud() -> None:
    from runners.exp_config import load_config

    path = _write_config("test_badkey", """\
        prompt: plr_v1.5_cot
        enums:
          colours: [black]
        """)
    try:
        with pytest.raises(ValueError, match="unknown enums key"):
            load_config(_LAB_ROOT, "test_badkey")
    finally:
        path.unlink()
