"""lab 러너 — 골든셋을 (mock 또는 실제) 모델로 재채점하는 크롭 루프.

크롭마다: quality 스코어 측정(게이팅 아님) → run_plr 1회 → 스펙 기반
pred/margin 추출 → 기록. 산출물 3형제:
  predictions.jsonl   {"obj_id","pred","reason","margin","quality"} — 채점 뷰
  attributes.jsonl    {"obj_id","plr_json"}  — 전체 슬롯 (per-slot 분석 원료)
  raw_responses.jsonl {"obj_id","raw",토큰수} — 모델 원문 + 비용 (증거/비용분석)

(원문 설명)
Besides predictions.jsonl, re_score also writes `attributes.jsonl` — the full
PLR output per crop, raw material for per-slot analysis.

No DB / redis / gemma_backend is imported at module level. The only heavy
dependency at import time is PIL. The model is injected by the caller
(LabGemmaModel or a MockModel for tests). The quality gate was removed with
the plr_v1.5_cot single-view contract — every crop goes to the model.

Attribute handling is SPEC-driven (evalkit.dataset.attribute_spec): the
dataset's manifest.yaml may declare its own labels / pred_path / margin_path /
bias_pair / object_type_hint; gender / vehicle_type / military are built-in
presets and double as the reference examples. (Military note: the preset uses
object_type_hint="person" because the military attribute lives on the person
attributes dict; for a vehicle military set declare object_type_hint: vehicle
in the manifest.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


# =====================================================================
# Reason extraction (presets only — pred/margin come from the attribute spec)
# =====================================================================


def _extract_reason(attribute: str, plr_json: dict[str, Any]) -> str:
    """예측의 근거 문구 추출 — gender 프리셋만 프롬프트가 gender_reason을
    emit하므로 그 외 속성은 "".
    예) → "broad shoulders, dark hair" """
    if attribute != "gender":
        return ""
    reason = ((plr_json.get("attributes") or {}).get("gender_scores") or {}).get("reason") or ""
    if isinstance(reason, list):
        reason = ", ".join(str(x) for x in reason)
    return str(reason)


class _RawCapture:
    """모델 래퍼 — 호출마다 원문 응답 + 토큰 사용량을 기록하는 스파이.
    (Model 프로토콜이 str만 반환해 raw가 run_plr 안에서 버려지므로, parity
    파일(plr_core)을 건드리지 않고 여기서 가로챈다. 정확 토큰 수는
    LabGemmaModel.last_result에서; 없으면(mock) chars/4 추정 + exact=false 표기.)

    Model wrapper that records each call's raw response + token usage.

    The Model protocol returns only the raw string, so raw text and token
    counts would otherwise be discarded inside run_plr. Wrapping here keeps
    plr_core (parity surface) untouched. Exact token counts come from
    LabGemmaModel.last_result (llama.cpp usage); models without it (mock)
    fall back to a chars/4 estimate, flagged via `tokens_exact: false`.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self.records: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:  # delegate sampling attrs etc.
        return getattr(self._model, name)

    @staticmethod
    def _text_of(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                parts.extend(str(ch.get("text", "")) for ch in c if isinstance(ch, dict))
        return "\n".join(parts)

    def generate(self, messages: list[dict[str, Any]], image: Any) -> str:
        raw = self._model.generate(messages, image)
        prompt_text = self._text_of(messages)
        gen = getattr(self._model, "last_result", None)
        exact = gen is not None and getattr(gen, "output_tokens", 0) > 0
        self.records.append({
            "prompt_chars": len(prompt_text),
            "raw_chars": len(raw),
            "input_tokens": int(gen.input_tokens) if exact else len(prompt_text) // 4,
            "output_tokens": int(gen.output_tokens) if exact else len(raw) // 4,
            "tokens_exact": bool(exact),
            "raw": raw,
        })
        return raw


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
                           (full PLR output, for per-slot analysis).

    Raises:
      FileNotFoundError if a crop image is missing (fail-loud — no silent skip).
      AssertionError    if the written count != the golden obj_id count.
    """
    from types import SimpleNamespace

    import plr_core
    import quality_gate

    here = Path(__file__).resolve().parent.parent  # lab root (runners/ is one below)
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

    # Attribute spec: manifest declaration (labels/pred_path/...) wins,
    # built-in PLR presets (gender/vehicle_type/military) fill the gaps —
    # the label SET is the dataset author's choice, the STRUCTURE is fixed.
    from evalkit.dataset import attribute_spec, resolve_json_path
    spec = attribute_spec(gdir, attribute)
    pred_path = spec.get("pred_path")
    if not pred_path:
        raise ValueError(
            f"No pred_path for attribute {attribute!r}: it is not a built-in "
            "preset (gender/vehicle_type/military), so the dataset's "
            "manifest.yaml must declare `labels:` and `pred_path:` "
            "(see docs/DATASET_SPEC.md)."
        )
    object_type_hint = spec.get("object_type_hint") or "person"

    # Version-specific prompt wiring. Only when a real yaml-backed version is
    # requested do we build a per-version message builder; otherwise build_messages
    # stays None so run_plr uses the module-level constants (byte-identical to the
    # live path, and covering mock/demo versions with no yaml). FilePromptProvider
    # is imported lazily here (not at module top) to keep `import re_score`
    # import-clean and to avoid pulling the registry/bootstrap.
    build_messages = None
    stamped_version: str | None = None
    variant = None
    if prompt_version:
        from runners.exp_config import apply_sampling, load_config

        # --version accepts an EXPERIMENT CONFIG name (configs/<n>.yaml — a
        # composed combination referencing a prompt version) or a plain
        # prompt version (prompts/<V>.yaml). Names matching neither
        # (mock_v1, demo tags) fall back to the module constants.
        variant = load_config(here, prompt_version)
        prompt_yaml_ver = variant.prompt if variant else prompt_version
        if ((here / "prompts" / prompt_yaml_ver).is_dir()
                or (here / "prompts" / f"{prompt_yaml_ver}.yaml").exists()):
            from providers.file_prompt_provider import FilePromptProvider

            _enum_overrides = dict(variant.enums) if variant else None
            build_messages = (
                lambda hint: FilePromptProvider(
                    version_override=prompt_yaml_ver,
                    enum_overrides=_enum_overrides,
                ).build_plr_messages(hint)
            )
            # Ledger tag = what the user named: the experiment-config
            # name, or the bare prompt version.
            stamped_version = prompt_version
            apply_sampling(model, variant)

    new_preds: list[dict[str, Any]] = []
    new_attrs: list[dict[str, Any]] = []

    # Raw capture: record every model response verbatim + token usage.
    # Written to raw_responses.jsonl (run artifact, gitignored) — evidence
    # for improve-prompt and for token-cost analysis across versions.
    model = _RawCapture(model)

    for obj_id in obj_ids:
        crop_path = crops_dir / f"{obj_id}.jpg"
        if not crop_path.exists():
            raise FileNotFoundError(
                f"Missing crop for obj_id={obj_id!r}: {crop_path}\n"
                "re_score is fail-loud — add the crop or remove the obj_id from predictions.jsonl."
            )

        pil = Image.open(crop_path).convert("RGB")

        # Quality score — MEASUREMENT ONLY, never gating (single-view
        # contract: every crop still goes to the model). Recorded so eval can
        # split accuracy by crop quality and check where errors concentrate.
        quality = round(float(quality_gate.evaluate(pil).score), 4)

        # PLR inference — single-view contract (plr_v1.5_cot): the quality
        # gate no longer withholds crops from the model, so every crop gets
        # exactly one call. run_plr's qreport parameter only steers its (now
        # unused) coarse_only branch — pin the normal mode.
        # preprocess.marker=false (config knob): pass the RAW crop with
        # _pre_marked=True so run_plr skips drawing the corner marker.
        skip_marker = variant is not None and not variant.marker
        plr_json = plr_core.run_plr(
            pil,
            SimpleNamespace(mode="normal_plr"),
            model,
            object_type_hint=object_type_hint,
            build_messages=build_messages,
            _pre_marked=skip_marker,
        )

        pred_val = resolve_json_path(plr_json, pred_path)
        pred = str(pred_val) if pred_val not in (None, "") else "unknown"
        reason = _extract_reason(attribute, plr_json)
        margin = None
        if spec.get("margin_path"):
            m = resolve_json_path(plr_json, spec["margin_path"])
            try:
                margin = float(m) if m is not None else None
            except (TypeError, ValueError):
                margin = None

        new_preds.append({
            "obj_id": obj_id, "pred": pred, "reason": reason,
            "margin": margin, "quality": quality,
        })
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

    # Raw responses + token usage (one row per crop, same order as obj_ids)
    raw_path = gdir / "raw_responses.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for oid, rec in zip(obj_ids, model.records):
            f.write(json.dumps({"obj_id": oid, **rec}, ensure_ascii=False) + "\n")

    _n = max(len(model.records), 1)
    tokens = {
        "exact": all(r["tokens_exact"] for r in model.records) if model.records else False,
        "input_total": sum(r["input_tokens"] for r in model.records),
        "output_total": sum(r["output_tokens"] for r in model.records),
        "input_avg": round(sum(r["input_tokens"] for r in model.records) / _n, 1),
        "output_avg": round(sum(r["output_tokens"] for r in model.records) / _n, 1),
    }

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
        # Token usage summary (exact=False means chars/4 estimate, e.g. mock).
        "tokens": tokens,
    }
