"""Tests for the two SESSION_HANDOFF §8 gap closures:

A. search `--version` wiring — run_search_over_golden(prompt_version=V) must
   send the QUERY-PARSER prompt from prompts/<V>.yaml when a Gemma backend is
   supplied, and keep the constants byte-identical when no version (or a
   yaml-less version) is given. The dictionary path (model=None) stays
   untouched — it sends no prompt at all.

B. format/reason experiment axes — optional `formats:` / `reasons:` keys in
   experiment.yaml cross into per-cell IR_PLR_FORMAT / IR_PLR_REASON env,
   stamp distinguishable ledger version tags, never leak env across cells,
   and fail loudly (per cell) when the format axis conflicts with a
   yaml-pinned prompt version.

No GPU, no DB, no Redis.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))


# =====================================================================
# Shared fixtures
# =====================================================================

# Gemma query-parser backends return an object with .raw (see
# query_parser.parse_with_gemma). qp_v0.4 shape: raw entities + residue.
_QP_JSON = json.dumps({
    "query_type": "person_search",
    "attributes": {"upper_color": "검정"},
    "free_form_residue": [],
    "raw_clean_query": "검정 옷 사람",
})


class RecordingQPBackend:
    """Query-parser backend stub: records every messages list it receives and
    returns a canned, parseable qp_v0.4 JSON response."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def generate(self, pil, messages, max_tokens=512, temperature=0.0):  # noqa: ARG002
        self.calls.append(messages)
        return SimpleNamespace(raw=_QP_JSON)


def _make_search_golden(base: Path) -> Path:
    """Minimal search golden dir: 1 query, 2 candidates."""
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "queries.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"query": "검정 옷 사람", "relevant": ["p1"]},
                           ensure_ascii=False) + "\n")
    with open(base / "attributes.jsonl", "w", encoding="utf-8") as f:
        for oid, color in (("p1", "black"), ("p2", "red")):
            plr = {"object_type": "person",
                   "attributes": {"upper": {"color": color}}}
            f.write(json.dumps({"obj_id": oid, "plr_json": plr},
                               ensure_ascii=False) + "\n")
    return base


# =====================================================================
# A. search --version wiring
# =====================================================================


def test_search_version_sends_yaml_query_parser_prompt(tmp_path: Path) -> None:
    """prompt_version=<yaml-backed V> + backend → the query-parser messages
    sent to the backend are exactly FilePromptProvider(V)'s output."""
    import re_score as rs
    from providers.file_prompt_provider import FilePromptProvider

    sdir = _make_search_golden(tmp_path / "search")
    backend = RecordingQPBackend()
    rs.run_search_over_golden(
        queries_path=str(sdir / "queries.jsonl"),
        attributes_path=str(sdir / "attributes.jsonl"),
        model=backend,
        prompt_version="plr_v1.4_cot",
    )

    assert backend.calls, "backend.generate was never called"
    expected = FilePromptProvider(
        version_override="plr_v1.4_cot"
    ).build_query_parser_messages("검정 옷 사람")
    assert backend.calls[0] == expected, (
        "search --version did not route the query-parser prompt through "
        "prompts/plr_v1.4_cot.yaml"
    )


def test_search_no_version_keeps_constants(tmp_path: Path) -> None:
    """No prompt_version (and a yaml-less version) → constants prompt,
    byte-identical to plr_prompts.build_query_parser_messages."""
    import re_score as rs
    from plr_prompts import build_query_parser_messages

    for tag, version in (("none", None), ("mockv", "mock_v1")):
        sdir = _make_search_golden(tmp_path / tag)
        backend = RecordingQPBackend()
        rs.run_search_over_golden(
            queries_path=str(sdir / "queries.jsonl"),
            attributes_path=str(sdir / "attributes.jsonl"),
            model=backend,
            prompt_version=version,
        )
        assert backend.calls[0] == build_query_parser_messages("검정 옷 사람"), (
            f"prompt_version={version!r} must fall back to the constants prompt"
        )


def test_search_dictionary_path_unaffected(tmp_path: Path) -> None:
    """model=None + prompt_version → identical results to the pre-wiring call
    (dictionary parse; no prompt is sent, version is a no-op)."""
    import re_score as rs

    sdir = _make_search_golden(tmp_path / "dict")
    rs.run_search_over_golden(
        queries_path=str(sdir / "queries.jsonl"),
        attributes_path=str(sdir / "attributes.jsonl"),
        model=None,
        prompt_version="plr_v1.4_cot",
    )
    with open(sdir / "search_results.jsonl", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    assert results and results[0]["ranked"][0] == "p1", (
        "dictionary-path search should still rank the black-upper person first"
    )


# =====================================================================
# B. format/reason experiment axes
# =====================================================================

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


def test_enumerate_cells_crosses_format_reason_axes() -> None:
    """plr cells cross formats × reasons; search cells do not."""
    import experiment as ex

    cfg = {
        "datasets": ["d"], "models": ["mock"], "prompts": ["p"],
        "pipelines": ["plr", "search"], "attributes": ["gender"],
        "formats": ["yaml", "json"], "reasons": ["on", "off"],
    }
    cells = ex.enumerate_cells(cfg)
    plr_cells = [c for c in cells if c.pipeline == "plr"]
    search_cells = [c for c in cells if c.pipeline == "search"]
    assert len(plr_cells) == 4, f"expected 2×2 plr cells, got {len(plr_cells)}"
    assert {(c.fmt, c.reason) for c in plr_cells} == {
        ("yaml", "on"), ("yaml", "off"), ("json", "on"), ("json", "off")}
    assert len(search_cells) == 1 and search_cells[0].fmt == "" == search_cells[0].reason


def test_validate_schema_rejects_bad_axis_values() -> None:
    import experiment as ex

    base = {"datasets": ["d"], "models": ["m"], "prompts": ["p"], "pipelines": ["plr"]}
    with pytest.raises(ValueError, match="formats"):
        ex._validate_schema({**base, "formats": ["xml"]}, "x.yaml")
    with pytest.raises(ValueError, match="reasons"):
        ex._validate_schema({**base, "reasons": ["maybe"]}, "x.yaml")
    with pytest.raises(ValueError, match="formats"):
        ex._validate_schema({**base, "formats": []}, "x.yaml")


def test_experiment_reason_axis_sets_env_and_stamps_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end matrix with reasons [on, off]: each cell runs under its own
    IR_PLR_REASON, the ledger stamps distinct version tags, and the env is
    restored after the matrix."""
    import registry
    import experiment

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
          - plr_v1.4_cot
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
    assert versions == {"plr_v1.4_cot+reason-on", "plr_v1.4_cot+reason-off"}, (
        f"ledger version tags must distinguish the reason axis, got {versions}"
    )


def test_format_axis_conflict_with_yaml_pinned_version_fails_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """formats: [json] against plr_v1.4_cot (yaml-pinned) must fail that cell
    loudly — and NOT crash the whole matrix (exit 2 = all cells failed)."""
    import registry
    import experiment

    ds_dir = _make_gender_dataset(tmp_path / "ds", ["m1"])
    monkeypatch.setitem(
        registry.MODELS, "envrec", lambda: EnvRecordingModel([]))

    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(textwrap.dedent(f"""\
        datasets:
          - {ds_dir}
        models:
          - envrec
        prompts:
          - plr_v1.4_cot
        pipelines:
          - plr
        attributes:
          - gender
        formats:
          - json
        ledger: {tmp_path / 'ledger.jsonl'}
        """), encoding="utf-8")

    exit_code = experiment.run_experiment(str(yaml_path))
    assert exit_code == 2, (
        "the conflicting (yaml-pinned prompt × json format) cell must fail "
        f"and, being the only cell, yield exit 2 — got {exit_code}"
    )
