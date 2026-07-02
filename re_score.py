"""Lab runner A: re-score a golden attribute set with a mock or real model.

Decision (candidates / attributes):
  Option (a) — re_score ALSO writes `eval/golden/<attr>/attributes.jsonl`
  (one line per obj_id: {"obj_id": ..., "plr_json": {...}}).
  run_search_over_golden reads that file to build the full-attribute
  candidate rows that search_core.run_search needs. This keeps every
  runner self-contained: no side-channel file is needed, and the search
  runner always gets the PLR attributes that correspond to the latest
  re_score pass.

No DB / redis / gemma_backend is imported at module level. The only heavy
dependency at import time is PIL. The model is injected by the caller
(LabGemmaModel or a MockModel for tests). The quality gate was removed with
the plr_v1.5_cot single-view contract — every crop goes to the model.

Military note: HINT["military"] = "person" because the military attribute
lives on the person's attributes dict (plr_v1.4_cot: Gemma judges
military/civilian from camouflage / field-uniform cues on the person crop,
not a separate object type). The vehicle military flag (_attach_military_flags)
is also populated by plr_core for vehicle crops — if a future military golden
set covers vehicles, pass object_type_hint="vehicle" explicitly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


# =====================================================================
# Attribute → object_type_hint mapping
# =====================================================================

HINT: dict[str, str] = {
    "gender": "person",
    "vehicle_type": "vehicle",
    "military": "person",   # see module docstring
}


# =====================================================================
# Attribute extraction helpers
# =====================================================================


def _extract_pred_reason(attribute: str, plr_json: dict[str, Any]) -> tuple[str, str]:
    """Extract (pred, reason) for one attribute from a full PLR JSON.

    Mirrors the SQL expressions in eval/build_golden.py ATTR dict:
      gender      -> attributes.gender_scores.{selected, reason}
      vehicle_type-> attributes.type_topk[0].label
      military    -> attributes.military
    """
    attrs = plr_json.get("attributes") or {}

    if attribute == "gender":
        gs = attrs.get("gender_scores") or {}
        pred = gs.get("selected") or "unknown"
        reason = gs.get("reason") or ""
        # reason may be stored as an evidence list — normalise to str
        if isinstance(reason, list):
            reason = ", ".join(str(x) for x in reason)
        return str(pred), str(reason)

    if attribute == "vehicle_type":
        topk = attrs.get("type_topk") or []
        pred = topk[0].get("label", "unknown") if topk else "unknown"
        return str(pred), ""

    if attribute == "military":
        pred = attrs.get("military") or "unknown"
        return str(pred), ""

    raise ValueError(f"Unknown attribute: {attribute!r}. Add it to HINT and _extract_pred_reason.")


# =====================================================================
# PLR lab runner (Deliverable 1)
# =====================================================================


def re_score(
    attribute: str,
    model: Any,
    golden_dir: str | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Re-score every object in the golden set for one attribute.

    Args:
      attribute:  "gender" | "vehicle_type" | "military"
      model:      Any object with .generate(messages, image) -> str
                  (gemma_model.Model protocol). May be a MockModel for tests.
      golden_dir: path to the golden dir (default: eval/golden/<attribute>).
      prompt_version: optional prompt version tag (e.g. "plr_v1.3_cot"). When
                  given AND prompts/<prompt_version>.yaml exists, the MAIN PLR
                  prompt is built from that version's YAML via FilePromptProvider,
                  so two cells with different prompt versions genuinely send
                  DIFFERENT prompts (not just different ledger labels). When
                  None, or the version has no YAML (e.g. a mock/demo version like
                  "mock_v1"), the module-level constants are used (byte-identical
                  to the live path). The returned meta["version"] is stamped with
                  prompt_version in the YAML-backed case so the comparison is
                  labeled correctly.

    Returns:
      meta dict: {attribute, n, version, gemma_repo}

    Side-effects (writes inside golden_dir):
      predictions.jsonl  — one line per obj_id: {obj_id, pred, reason}
      attributes.jsonl   — one line per obj_id: {obj_id, plr_json}
                           (used by run_search_over_golden as full-attribute
                           candidates; see module docstring decision (a)).

    Raises:
      FileNotFoundError if a crop image is missing (fail-loud — no silent skip).
      AssertionError    if the written count != the golden obj_id count.
    """
    from types import SimpleNamespace

    import plr_core

    here = Path(__file__).parent
    gdir = Path(golden_dir) if golden_dir else here / "eval" / "golden" / attribute

    preds_path = gdir / "predictions.jsonl"
    attrs_path = gdir / "attributes.jsonl"
    crops_dir = gdir / "crops"
    labels_path = gdir / "labels.jsonl"

    # Resolve the obj_id set to score. Preferred source: an existing
    # predictions.jsonl (the golden bootstrap that build-golden writes) —
    # keeps order for deterministic output. For a spec-only dataset (crops +
    # labels, no predictions.jsonl — e.g. the prepare-dataset "arbitrary
    # crops" path), fall back to the crop files, then to labels.jsonl.
    obj_ids: list[str] = []
    if preds_path.exists():
        with open(preds_path) as f:
            obj_ids = [json.loads(line)["obj_id"] for line in f if line.strip()]
    if not obj_ids and crops_dir.is_dir():
        obj_ids = sorted(p.stem for p in crops_dir.glob("*.jpg"))
    if not obj_ids and labels_path.exists():
        with open(labels_path) as f:
            obj_ids = sorted(json.loads(line)["obj_id"] for line in f if line.strip())
    if not obj_ids:
        raise FileNotFoundError(
            f"No obj_ids to score in {gdir}: provide a non-empty "
            f"predictions.jsonl, crops/*.jpg, or labels.jsonl."
        )
    obj_id_set: set[str] = set(obj_ids)

    object_type_hint = HINT.get(attribute, "person")

    # Version-specific prompt wiring. Only when a real yaml-backed version is
    # requested do we build a per-version message builder; otherwise build_messages
    # stays None so run_plr uses the module-level constants (byte-identical to the
    # live path, and covering mock/demo versions with no yaml). FilePromptProvider
    # is imported lazily here (not at module top) to keep `import re_score`
    # import-clean and to avoid pulling the registry/bootstrap.
    build_messages = None
    stamped_version: str | None = None
    if prompt_version and (here / "prompts" / f"{prompt_version}.yaml").exists():
        from providers.file_prompt_provider import FilePromptProvider

        build_messages = (
            lambda hint: FilePromptProvider(version_override=prompt_version)
            .build_plr_messages(hint)
        )
        stamped_version = prompt_version

    new_preds: list[dict[str, Any]] = []
    new_attrs: list[dict[str, Any]] = []

    for obj_id in obj_ids:
        crop_path = crops_dir / f"{obj_id}.jpg"
        if not crop_path.exists():
            raise FileNotFoundError(
                f"Missing crop for obj_id={obj_id!r}: {crop_path}\n"
                "re_score is fail-loud — add the crop or remove the obj_id from predictions.jsonl."
            )

        pil = Image.open(crop_path).convert("RGB")

        # PLR inference — single-view contract (plr_v1.5_cot): the quality
        # gate no longer withholds crops from the model, so every crop gets
        # exactly one call. run_plr's qreport parameter only steers its (now
        # unused) coarse_only branch — pin the normal mode.
        plr_json = plr_core.run_plr(
            pil,
            SimpleNamespace(mode="normal_plr"),
            model,
            object_type_hint=object_type_hint,
            build_messages=build_messages,
        )

        pred, reason = _extract_pred_reason(attribute, plr_json)

        new_preds.append({"obj_id": obj_id, "pred": pred, "reason": reason})
        new_attrs.append({"obj_id": obj_id, "plr_json": plr_json})

    # Sanity check before any writes
    assert len(new_preds) == len(obj_id_set), (
        f"re_score wrote {len(new_preds)} rows but golden set has {len(obj_id_set)} obj_ids"
    )

    # Overwrite predictions.jsonl (single file per run; version in ledger)
    with open(preds_path, "w", encoding="utf-8") as f:
        for row in new_preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write attributes.jsonl (full PLR JSON, for search runner)
    with open(attrs_path, "w", encoding="utf-8") as f:
        for row in new_attrs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stamp the ledger version with the ACTUAL prompt version when a yaml-backed
    # version drove this run (so a prompt-axis comparison is labeled correctly);
    # otherwise fall back to the env-derived "format+reason" tag.
    version = stamped_version or (
        os.environ.get("IR_PLR_FORMAT", "yaml")
        + "+" + os.environ.get("IR_PLR_REASON", "")
    ).rstrip("+")
    gemma_repo = os.environ.get("IR_GEMMA_REPO", "")

    return {
        "attribute": attribute,
        "n": len(new_preds),
        "version": version,
        "gemma_repo": gemma_repo,
    }


# =====================================================================
# Candidate row shim — satisfies search_core.run_search's row protocol
# =====================================================================


@dataclass
class _Row:
    """Minimal candidate row for search_core.run_search.

    run_search accesses row.plr_json (dict) and row.object_type (str).
    """

    obj_id: str
    plr_json: dict[str, Any]
    object_type: str


# =====================================================================
# Search runner (Deliverable 2)
# =====================================================================


def run_search_over_golden(
    queries_path: str | None = None,
    predictions_path: str | None = None,
    attributes_path: str | None = None,
    results_path: str | None = None,
    model: Any = None,
    prompt_version: str | None = None,
) -> None:
    """Run search_core.run_search over the golden search set.

    Reads:
      queries.jsonl     — lines {query: str, relevant: [obj_id, ...]}
      attributes.jsonl  — lines {obj_id: str, plr_json: dict}
                          (written by re_score; carries full PLR attributes)

    Writes:
      search_results.jsonl — lines {query: str, ranked: [obj_id, ...]}
                             Written alongside queries.jsonl (same directory)
                             unless results_path is given.

    Candidates carry full attribute dicts (decision (a) — see module docstring).
    parse_query runs on the regex/dictionary path when backend=None; it is
    deterministic and requires no GPU.

    Args:
      queries_path:     path to queries.jsonl (default: eval/golden/search/queries.jsonl)
      predictions_path: unused (reserved for future per-object score overlay)
      attributes_path:  path to attributes.jsonl (default: eval/golden/search/attributes.jsonl)
      results_path:     path to write search_results.jsonl
                        (default: same directory as queries_path)
      model:            passed to parse_query as backend; None = dictionary path (no GPU needed)
      prompt_version:   optional prompt version tag (e.g. "plr_v1.4_cot"). When
                        given AND prompts/<prompt_version>.yaml exists, the
                        QUERY-PARSER prompt is built from that version's YAML via
                        FilePromptProvider (same rule as re_score's PLR prompt).
                        Only effective on the Gemma parse path (model != None) —
                        the dictionary path sends no prompt at all. None, or a
                        version with no YAML, keeps the module-level constants.
    """
    import search_core
    from query_parser import parse_query

    here = Path(__file__).parent
    search_dir = here / "eval" / "golden" / "search"

    # Version-specific query-parser prompt wiring — mirrors re_score's PLR
    # wiring exactly (yaml-backed version → FilePromptProvider, else constants).
    build_messages = None
    if prompt_version and (here / "prompts" / f"{prompt_version}.yaml").exists():
        from providers.file_prompt_provider import FilePromptProvider

        build_messages = (
            lambda q: FilePromptProvider(version_override=prompt_version)
            .build_query_parser_messages(q)
        )

    q_path = Path(queries_path) if queries_path else search_dir / "queries.jsonl"
    a_path = Path(attributes_path) if attributes_path else search_dir / "attributes.jsonl"
    # Output lives next to queries.jsonl by default so callers that pass a
    # temp queries_path get their output in the same temp directory.
    if results_path:
        out_path = Path(results_path)
    else:
        out_path = q_path.parent / "search_results.jsonl"

    # Load queries
    with open(q_path) as f:
        queries = [json.loads(line) for line in f if line.strip()]

    # Load candidates from attributes.jsonl (full plr_json per obj_id)
    with open(a_path) as f:
        attr_rows = [json.loads(line) for line in f if line.strip()]

    candidates: list[_Row] = [
        _Row(
            obj_id=row["obj_id"],
            plr_json=row["plr_json"],
            object_type=row["plr_json"].get("object_type", "person"),
        )
        for row in attr_rows
    ]

    results: list[dict[str, Any]] = []

    for q_entry in queries:
        query_text = q_entry["query"]

        # Parse query — dictionary path when model=None (no GPU required)
        query_json = parse_query(query_text, backend=model, build_messages=build_messages)

        # Run search (hard-filter + attribute-match rank)
        ranked_rows = search_core.run_search(query_json, candidates)

        results.append({
            "query": query_text,
            "ranked": [row.obj_id for row in ranked_rows],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
