"""Pure single-view PLR inference core.

`run_plr(pil, qreport, model, object_type_hint) -> plr_json` is the storage-free
PLR pipeline the live indexer and the offline lab share:

  coarse_only  -> quality_gate.coarse_only_plr_json(...)  (skip generate)
  otherwise    -> draw target marker -> build_plr_messages -> model.generate
                  -> parse_plr_response -> _attach_military_flags

It depends ONLY on: plr_prompts, quality_gate, PIL, and the `Model` protocol
(gemma_model.Model). It has ZERO direct `gemma_backend` reference — generation
goes through `model`. The marker + military helpers live HERE (canonical home,
pure stdlib/PIL) and `indexing.py` imports them from this module, so a bare
`import plr_core` never transitively pulls storage / psycopg2 / redis.

Behavior-preserving note for the live path: `run_plr` exposes two PRIVATE
toggles, `_pre_marked` and `_attach`. The public/lab contract is the defaults
(`_pre_marked=False, _attach=True`) — marker drawn + flags attached, exactly as
the spec describes. The live indexer (`indexing._run_plr_with_optional_sr`)
already draws the marker once per object and attaches flags once post-merge, so
it invokes `run_plr(..., _pre_marked=True, _attach=False)` to avoid double-work,
keeping `_process_one`'s observable behavior byte-identical.
"""

from __future__ import annotations

from typing import Any, Callable

from PIL import Image, ImageDraw


# =====================================================================
# Target marker (pure PIL) — canonical home. indexing.py imports from here.
# =====================================================================


def _draw_target_marker(
    pil: Image.Image,
    box_ratio: float = 0.65,
    color: tuple[int, int, int] = (255, 230, 0),  # bright yellow
    line_width: int | None = None,
    arm_ratio: float = 0.20,
) -> Image.Image:
    """Draw four yellow corner brackets around the centre of the crop.

    v1.1_cot used a full rectangle — outline closure activated an
    "enclosure" semantic (uniform / one-piece / jumpsuit hallucinations
    accounted for 24% of upper_type regressions on the 50-object PoC).
    Corner brackets keep the "look here" cue without enclosing the
    subject, removing the false-containment prior.

    Args:
      box_ratio: side length of the conceptual box (corners are placed
        on its perimeter) as a fraction of the shorter crop dimension.
      color: RGB of the bracket strokes. Yellow (255, 230, 0) is
        uncommon on Korean street clothing and car paint.
      line_width: stroke thickness. None → auto from shorter side.
      arm_ratio: length of each bracket arm as a fraction of box_side.
    """
    w, h = pil.size
    side = min(w, h)
    if side < 16:                    # tiny thumbnail — skip, marker would dominate
        return pil
    if line_width is None:
        line_width = max(2, side // 80)
    box_side = int(side * box_ratio)
    arm = max(line_width * 2, int(box_side * arm_ratio))
    x0 = (w - box_side) // 2
    y0 = (h - box_side) // 2
    x1 = x0 + box_side
    y1 = y0 + box_side
    out = pil.copy()
    draw = ImageDraw.Draw(out)
    for cx, cy, dx, dy in (
        (x0, y0, +1, +1),    # top-left  L-shape opens down-right
        (x1, y0, -1, +1),    # top-right
        (x0, y1, +1, -1),    # bottom-left
        (x1, y1, -1, -1),    # bottom-right
    ):
        draw.line((cx, cy, cx + dx * arm, cy), fill=color, width=line_width)
        draw.line((cx, cy, cx, cy + dy * arm), fill=color, width=line_width)
    return out


# =====================================================================
# Military / color scalar enrichment (pure) — canonical home.
# =====================================================================

_MILITARY_COLOR = "military_olive"


def _top_color(color_topk: Any) -> str | None:
    """Return the label of the highest-scoring entry in a color_topk array.

    Returns None when the array is missing, empty, or malformed — never raises.
    The array is assumed to be already sorted descending by score (as Gemma
    outputs it), but we take max() defensively.
    """
    if not isinstance(color_topk, list) or not color_topk:
        return None
    try:
        best = max(color_topk, key=lambda e: float(e.get("score", 0.0)) if isinstance(e, dict) else 0.0)
        label = best.get("label") if isinstance(best, dict) else None
        return str(label) if label else None
    except Exception:  # noqa: BLE001
        return None


def _attach_military_flags(plr_json: dict[str, Any]) -> None:
    """Populate scalar color / military-flag fields in-place on *plr_json*.

    Written fields (all optional in the schema):
      vehicle -> attributes.primary_color  (str)
               -> attributes.is_military   (bool)
      person  -> attributes.upper_clothing.primary_color  (str)
               -> attributes.lower_clothing.primary_color  (str)
               -> attributes.is_soldier                    (bool)

    Defensive: never raises — silently skips any field when the source
    color_topk is missing/empty/malformed.  Intended to be called after
    PLR parse but before storage.upsert_row so the scalars live in
    plr_json (JSONB) and are containment-queryable.
    """
    obj_type = plr_json.get("object_type")
    attrs = plr_json.get("attributes")
    if not isinstance(attrs, dict):
        return

    # plr_v1.4_cot: Gemma now judges military/civilian directly from
    # camouflage / field-uniform / load-bearing cues (`military` field). We
    # OR that judgment with the legacy military_olive colour rule so olive
    # still works as a recall fallback on older rows or when military=="unknown".
    is_military_judgment = (attrs.get("military") == "military")

    if obj_type == "vehicle":
        pc = _top_color(attrs.get("color_topk"))
        if pc is not None:
            attrs["primary_color"] = pc
        attrs["is_military"] = (
            is_military_judgment or (pc == _MILITARY_COLOR)
        )

    elif obj_type == "person":
        upper = attrs.get("upper_clothing")
        lower = attrs.get("lower_clothing")
        upc = _top_color(upper.get("color_topk") if isinstance(upper, dict) else None)
        lpc = _top_color(lower.get("color_topk") if isinstance(lower, dict) else None)
        if isinstance(upper, dict) and upc is not None:
            upper["primary_color"] = upc
        if isinstance(lower, dict) and lpc is not None:
            lower["primary_color"] = lpc
        attrs["is_soldier"] = (
            is_military_judgment
            or upc == _MILITARY_COLOR
            or lpc == _MILITARY_COLOR
        )


def _plr_generate_parse(model: Any, image: Any, msgs: list[dict[str, Any]], hint: str) -> dict[str, Any]:
    """Shared inner core: run the model then parse the raw output.

    Pure in (model output, hint). Does NOT validate — schema validation and
    retry/DLQ orchestration stay in the caller so those semantics are unchanged.
    """
    from plr_prompts import parse_plr_response

    raw = model.generate(msgs, image)
    return parse_plr_response(raw, hint=hint)


def run_plr(
    pil: Any,
    qreport: Any,
    model: Any,
    object_type_hint: str,
    *,
    build_messages: Callable[[str], list[dict[str, Any]]] | None = None,
    _pre_marked: bool = False,
    _attach: bool = True,
) -> dict[str, Any]:
    """Single-view PLR inference.

    Args:
      pil: the object crop (PIL.Image). When `_pre_marked` is False the target
        marker is drawn here; when True the caller has already marked it.
      qreport: a QualityReport. If `qreport.mode == "coarse_only"` the model is
        skipped and a minimal coarse PLR JSON is returned.
      model: a `gemma_model.Model` — `generate(messages, image) -> str`.
      object_type_hint: "person" | "vehicle" prompt template hint.
      build_messages: (lab-only) optional builder `hint -> messages` for the
        MAIN prompt. When None (the default, and the ONLY behaviour in core/ir)
        the module-level `plr_prompts.build_plr_messages` is used, so the output
        is byte-identical to the live path. The lab supplies a version-specific
        builder (e.g. FilePromptProvider(version_override=...).build_plr_messages)
        so a single checkout can genuinely compare prompt versions. The
        schema-retry path is intentionally left on the constants builder.
      _pre_marked: (private) skip the marker draw when the caller already marked.
      _attach: (private) run `_attach_military_flags` on the result when True.

    Returns the PLR JSON dict.
    """
    import quality_gate

    if qreport.mode == "coarse_only":
        return quality_gate.coarse_only_plr_json(
            obj_type_hint=object_type_hint,
            report=qreport,
            dominant_color=None,
        )

    from plr_prompts import build_plr_messages

    marked = pil if _pre_marked else _draw_target_marker(pil)
    msgs = (build_messages or build_plr_messages)(object_type_hint)
    plr = _plr_generate_parse(model, marked, msgs, object_type_hint)
    if _attach:
        _attach_military_flags(plr)
    return plr
