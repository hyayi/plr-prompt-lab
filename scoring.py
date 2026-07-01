"""Multi-signal scoring for the PLR search.

Final score per candidate (redesign 2026-06):

    final_score = 0.85 * template_embedding
                + 0.10 * raw_query_embedding
                + 0.05 * quality_score

Attribute matching is now a HARD GATE (passes_hard_filter), not a ranking
term — every survivor is already attribute-valid, so ranking falls to
embedding similarity + quality. `attribute_match()` is still computed and
surfaced in ScoreBreakdown.attribute_match for diagnostics / the
results_detailed API (failure_log EMBEDDING_DRIFT, etc.), but it no longer
feeds final_score. It is a weighted sum over slots (age_group, gender,
upper_color, upper_type, lower_color, lower_type, equipment, action) of
per-slot score in [0,1].

Inputs:
  - query_json   : query_parser.QueryJSON.to_dict()
  - candidate    : storage.IndexedRow (has plr_json + embedding + quality_score)
  - q_template_emb, q_raw_emb : numpy 1-d vectors (already L2-normalized)

Output:
  ScoreBreakdown — final_score plus per-component scores (used in API response).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plr_schema import (
    color_group,
    lower_type_group,
    upper_type_group,
    vehicle_type_group,
)

log = logging.getLogger(__name__)


# =====================================================================
# Public weights (also exposed in API response for transparency)
# =====================================================================

SCORING_VERSION = "score_v0.5"  # 2026-06: attribute_match demoted to hard gate

# Why this re-weighting (v0.1 → v0.2):
# v0.1 gave embedding similarity 40% combined weight and attribute_match 55%.
# In practice the template caption ("어른 남성, 검정 재킷") and the user
# query ("청자켓을 입은 여자") both describe people in dark jackets — BGE-m3
# cosine reaches 0.95 even when gender flips, and 0.30 × 0.95 = 0.285 alone
# clears the 0.4 threshold no matter what the attribute slots say. Symptom:
# Redesign 2026-06: attribute matching is now a HARD GATE (passes_hard_filter),
# not a ranking score. Survivors are all attribute-valid, so ranking falls to
# embedding similarity + quality. `attribute_match` is therefore dropped from
# the weighted sum (but ScoreBreakdown.attribute_match is still populated for
# diagnostics / the results_detailed API — see final_score()).
WEIGHTS_FINAL = {
    "template_embedding": 0.85,
    "raw_query_embedding": 0.10,
    "quality_score": 0.05,
}
assert abs(sum(WEIGHTS_FINAL.values()) - 1.0) < 1e-9

# Slots that, when explicitly requested by the user, *must* match — a "여자"
# query should never surface a male candidate even with a 0.95 caption
# similarity. Strict semantics here is independent of the API `strict` flag,
# which still gates the broader required-slot conjunction. These are the
# attributes where embedding-driven slop is unacceptable:
#   gender / age_group     identity slots
#   action                 highly discriminative pose — "오토바이 탄 남자"
#                          must not return every male in the video just
#                          because gender filters pass and action is only
#                          scored softly.
HARD_FILTER_SLOTS: frozenset[str] = frozenset({
    "gender", "age_group", "action",
    # Rider-vehicle attributes are also discriminative — "빨간 오토바이
    # 탄 남자" should not surface a rider on a black motorcycle just
    # because the action + gender lines up. Empty queries skip
    # naturally (the `if not q_vals: continue` guard).
    "rider_vehicle_color", "rider_vehicle_type",
    # NOTE: lower_type removed here (redesign 2026-06). Garment FABRIC
    # (jeans/slacks) is no longer a hard filter — only coarse SHAPE
    # (lower_shape) is, handled in the person colour/shape block below.
    # Fine type moves to the refine stage.
})

# attribute_match slot weights (sum to 1.0 for person; vehicle uses a subset).
WEIGHTS_SLOT_PERSON: dict[str, float] = {
    "age_group": 0.10,
    "gender": 0.08,
    "outfit_type": 0.06,
    "upper_color": 0.16,
    "upper_type": 0.09,
    "lower_color": 0.13,
    "lower_type": 0.09,
    "equipment": 0.15,
    "action": 0.05,
    # Rider-vehicle attributes — pulled in by a small weight only; the
    # action hard filter is what actually scopes the result to riders.
    # These scores re-rank among riders by bike color/type.
    "rider_vehicle_color": 0.05,
    "rider_vehicle_type":  0.04,
}
WEIGHTS_SLOT_VEHICLE: dict[str, float] = {
    "vehicle_color": 0.55,
    "vehicle_type": 0.45,
}

# Per-slot bonus structure
SELECTED_BONUS = 0.40       # added when the candidate's "selected" label matches query
TOPK_SCORE_WEIGHT = 0.40    # weight on the candidate's topk score for exact match
COARSE_GROUP_BONUS = 0.20   # weight on candidate's topk score for coarse-group match
EXCLUDED_PENALTY = 0.30     # subtracted (per matching excluded slot value)
LOW_MARGIN_MULTIPLIER = 0.9 # multiplies score when decision_margin is below threshold
LOW_MARGIN_THRESHOLD = 0.20  # absolute margin below which we discount


# =====================================================================
# Data class
# =====================================================================


@dataclass
class ScoreBreakdown:
    """Per-candidate score breakdown used in API responses."""

    final_score: float
    attribute_match: float
    template_embedding: float
    raw_query_embedding: float
    quality_score: float
    slot_contributions: dict[str, float] = field(default_factory=dict)


# =====================================================================
# attribute_match
# =====================================================================


def attribute_match(
    query_required: dict[str, list[str]],
    query_excluded: dict[str, list[str]] | None,
    candidate_plr: dict[str, Any],
    is_vehicle: bool,
) -> tuple[float, dict[str, float]]:
    """Slot-by-slot matching, normalised by the query's slot footprint.

    The slot weight table sums to 1.0 across *all* slots, but a real query
    only mentions 2-3 of them. Without normalising, a perfect 2-slot match
    capped attribute_match around 0.18, which got drowned by the embedding
    term (cosine 0.95) at the final-score stage. Dividing by the sum of the
    requested slots' weights restores [0, 1] as the natural range.

    Returns (score in [0,1], per-slot contribution dict for logging).
    """
    if not query_required and not query_excluded:
        # The user wrote a broad category query like "자동차" / "사람" /
        # "남자" with no narrowing attributes. The target filter in the
        # caller has already kept only the right object_type, so every
        # surviving candidate is by definition a perfect match for what
        # was asked. Give them the maximum attribute score so ranking
        # falls back to embedding similarity + quality.
        return 1.0, {"__broad_target_query__": 1.0}

    weights = WEIGHTS_SLOT_VEHICLE if is_vehicle else WEIGHTS_SLOT_PERSON
    attrs = candidate_plr.get("attributes") or {}

    contributions: dict[str, float] = {}
    total = 0.0
    requested_weight_sum = 0.0

    for slot, slot_weight in weights.items():
        q_vals = query_required.get(slot) or []
        if not q_vals:
            continue
        requested_weight_sum += slot_weight
        slot_score = _slot_match_score(slot, q_vals, attrs)
        contributions[slot] = round(slot_score * slot_weight, 4)
        total += slot_score * slot_weight

    if requested_weight_sum > 0:
        total /= requested_weight_sum  # normalise to [0, 1]

    # Excluded penalty: if the candidate matches any excluded value in any slot
    if query_excluded:
        for slot, vals in query_excluded.items():
            if _candidate_has_any(slot, vals, attrs):
                total -= EXCLUDED_PENALTY
                contributions.setdefault(f"{slot}__excluded_penalty", -EXCLUDED_PENALTY)

    # Low decision_margin penalty (only meaningful for person)
    if not is_vehicle and _low_margin(attrs):
        total *= LOW_MARGIN_MULTIPLIER
        contributions["__low_margin_multiplier"] = LOW_MARGIN_MULTIPLIER

    return max(0.0, min(1.0, total)), contributions


def _slot_match_score(slot: str, q_vals: list[str], attrs: dict[str, Any]) -> float:
    """Score one slot in [0,1]."""
    if slot == "age_group":
        return _scalar_score(attrs.get("age_group_scores"), q_vals)
    if slot == "gender":
        return _scalar_score(attrs.get("gender_scores"), q_vals)
    if slot == "outfit_type":
        return _scalar_score(attrs.get("outfit_type_scores"), q_vals)
    if slot == "action":
        return _scalar_score(attrs.get("static_action_state_scores"), q_vals)

    if slot == "upper_color":
        return _topk_color_score(attrs.get("upper_clothing"), q_vals)
    if slot == "lower_color":
        return _topk_color_score(attrs.get("lower_clothing"), q_vals)
    if slot == "upper_type":
        return _topk_type_score(attrs.get("upper_clothing"), q_vals, upper_type_group)
    if slot == "lower_type":
        return _topk_type_score(attrs.get("lower_clothing"), q_vals, lower_type_group)

    if slot == "equipment":
        return _equipment_score(attrs.get("equipment"), q_vals)

    if slot == "vehicle_color":
        return _topk_color_score(attrs, q_vals, root_key="color_topk")
    if slot == "vehicle_type":
        return _topk_type_score(attrs, q_vals, vehicle_type_group, root_key="type_topk")

    if slot == "rider_vehicle_color":
        rv = attrs.get("rider_vehicle") or {}
        return _topk_color_score(rv, q_vals, root_key="color_topk")
    if slot == "rider_vehicle_type":
        rv = attrs.get("rider_vehicle") or {}
        sel = rv.get("type")
        if sel and sel in q_vals:
            return 1.0
        return 0.0

    return 0.0


def _scalar_score(scores_dict: dict[str, Any] | None, q_vals: list[str]) -> float:
    """Score a binary or N-way score-distribution slot (gender / age / outfit / action)."""
    if not scores_dict:
        return 0.0
    selected = scores_dict.get("selected")
    out = 0.0
    if selected and selected in q_vals:
        out += SELECTED_BONUS
        score_value = float(scores_dict.get(selected, 0.0))
        out += TOPK_SCORE_WEIGHT * score_value
    else:
        # Maybe the query value isn't selected but has a score we can use
        for v in q_vals:
            if v in scores_dict and isinstance(scores_dict[v], (int, float)):
                out += TOPK_SCORE_WEIGHT * float(scores_dict[v])
                break
    return min(1.0, out)


def _topk_color_score(
    clothing_or_attrs: dict[str, Any] | None,
    q_vals: list[str],
    *,
    root_key: str = "color_topk",
) -> float:
    """Color slot scoring with selected bonus + topk + coarse-group bonus."""
    if not clothing_or_attrs:
        return 0.0
    topk = clothing_or_attrs.get(root_key) or []
    if not topk:
        return 0.0

    q_set = set(q_vals)
    q_groups = {color_group(v) for v in q_vals}

    out = 0.0
    selected_label = topk[0].get("label")
    if selected_label in q_set:
        out += SELECTED_BONUS

    # Walk top-3 entries
    for entry in topk[:3]:
        label = entry.get("label")
        score_v = float(entry.get("score", 0.0))
        if not label:
            continue
        if label in q_set:
            out += TOPK_SCORE_WEIGHT * score_v
        elif color_group(label) in q_groups:
            out += COARSE_GROUP_BONUS * score_v
    return min(1.0, out)


def _topk_type_score(
    clothing_or_attrs: dict[str, Any] | None,
    q_vals: list[str],
    group_fn,
    *,
    root_key: str = "type_topk",
) -> float:
    """Type slot scoring with selected bonus + topk + coarse-group bonus."""
    if not clothing_or_attrs:
        return 0.0
    topk = clothing_or_attrs.get(root_key) or []
    if not topk:
        return 0.0

    q_set = set(q_vals)
    q_groups = {group_fn(v) for v in q_vals}

    out = 0.0
    selected_label = topk[0].get("label")
    if selected_label in q_set:
        out += SELECTED_BONUS

    for entry in topk[:3]:
        label = entry.get("label")
        score_v = float(entry.get("score", 0.0))
        if not label:
            continue
        if label in q_set:
            out += TOPK_SCORE_WEIGHT * score_v
        elif group_fn(label) in q_groups:
            out += COARSE_GROUP_BONUS * score_v
    return min(1.0, out)


def _equipment_score(equipment_list: list[dict[str, Any]] | None, q_vals: list[str]) -> float:
    """Equipment slot: any candidate equipment item with type in q_vals."""
    if not equipment_list:
        return 0.0
    q_set = set(q_vals)
    out = 0.0
    matched = False
    for eq in equipment_list:
        t = eq.get("type")
        s = float(eq.get("score", 0.0))
        if t in q_set:
            if not matched:
                out += SELECTED_BONUS
                matched = True
            out += TOPK_SCORE_WEIGHT * s
    return min(1.0, out)


def _candidate_has_any(slot: str, vals: list[str], attrs: dict[str, Any]) -> bool:
    """Check if the candidate has any of these excluded values in this slot."""
    if not vals:
        return False
    if slot == "equipment":
        for eq in attrs.get("equipment") or []:
            if eq.get("type") in vals:
                return True
        return False
    if slot in {"age_group", "gender", "outfit_type", "action"}:
        sd = {
            "age_group": "age_group_scores",
            "gender": "gender_scores",
            "outfit_type": "outfit_type_scores",
            "action": "static_action_state_scores",
        }[slot]
        return (attrs.get(sd) or {}).get("selected") in vals
    if slot == "upper_color":
        topk = (attrs.get("upper_clothing") or {}).get("color_topk") or []
        return any(t.get("label") in vals for t in topk[:3])
    if slot == "lower_color":
        topk = (attrs.get("lower_clothing") or {}).get("color_topk") or []
        return any(t.get("label") in vals for t in topk[:3])
    if slot == "upper_type":
        topk = (attrs.get("upper_clothing") or {}).get("type_topk") or []
        return any(t.get("label") in vals for t in topk[:3])
    if slot == "lower_type":
        topk = (attrs.get("lower_clothing") or {}).get("type_topk") or []
        return any(t.get("label") in vals for t in topk[:3])
    return False


def _low_margin(attrs: dict[str, Any]) -> bool:
    """True if any critical score-distribution field has a low decision_margin."""
    for field in ("gender_scores", "age_group_scores", "outfit_type_scores"):
        d = attrs.get(field) or {}
        m = d.get("decision_margin")
        if isinstance(m, (int, float)) and m < LOW_MARGIN_THRESHOLD:
            return True
    return False


# =====================================================================
# Embedding similarity
# =====================================================================


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity assuming both vectors are L2-normalized.

    Clamps to [-1, 1] for numerical safety.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    s = float(np.dot(a, b))
    if s > 1.0:
        s = 1.0
    elif s < -1.0:
        s = -1.0
    return s


def _emb_to_unit_range(cos_value: float) -> float:
    """Map cosine [-1, 1] → [0, 1]."""
    return max(0.0, min(1.0, (cos_value + 1.0) * 0.5))


# =====================================================================
# Final score combiner
# =====================================================================


def final_score(
    *,
    query_required: dict[str, list[str]],
    query_excluded: dict[str, list[str]] | None,
    candidate_plr: dict[str, Any],
    candidate_embedding: np.ndarray,
    candidate_quality: float,
    q_template_emb: np.ndarray,
    q_raw_emb: np.ndarray,
    is_vehicle: bool,
) -> ScoreBreakdown:
    """Combine all four signals into a single score."""
    attr, contributions = attribute_match(
        query_required=query_required,
        query_excluded=query_excluded,
        candidate_plr=candidate_plr,
        is_vehicle=is_vehicle,
    )

    tmpl_cos = cosine(q_template_emb, candidate_embedding)
    raw_cos = cosine(q_raw_emb, candidate_embedding)
    tmpl = _emb_to_unit_range(tmpl_cos)
    raw = _emb_to_unit_range(raw_cos)
    qual = float(max(0.0, min(1.0, candidate_quality)))

    # attr is still computed (above) and surfaced in ScoreBreakdown for
    # diagnostics, but no longer contributes to the ranking score — the hard
    # gate already guarantees every survivor is attribute-valid.
    fs = (
        WEIGHTS_FINAL["template_embedding"] * tmpl
        + WEIGHTS_FINAL["raw_query_embedding"] * raw
        + WEIGHTS_FINAL["quality_score"] * qual
    )

    return ScoreBreakdown(
        final_score=round(fs, 4),
        attribute_match=round(attr, 4),
        template_embedding=round(tmpl, 4),
        raw_query_embedding=round(raw, 4),
        quality_score=round(qual, 4),
        slot_contributions=contributions,
    )


def passes_hard_filter(
    query_required: dict[str, list[str]],
    candidate_plr: dict[str, Any],
    query_excluded: dict[str, list[str]] | None = None,
) -> bool:
    """Two-stage cascade: this is the cheap stage. Reject candidates whose
    coarse identity / color-group / type-group are clearly different from
    the query; survivors go on to the weighted-sum stage for fine ranking.

    The filters are grouped by tightness so that recall stays intact:

    - **gender, age_group**: strict label match (no group fallback). The PLR
      enum is already tight — you're one or the other.
    - **upper_color, lower_color, vehicle_color**: FINE `color_hard_group` match
      — black groups only with dark_gray (achromatic dark), NOT the coarse dark
      band (dark_brown/dark_green/military_olive). "검은색" therefore excludes
      brown/green/국방색. Pure mismatch (red vs black) is hard rejected. (The
      coarse `color_group` is still used for "어두운/밝은" brightness queries and
      soft-score bonuses.)
    - **any_color**: garment-agnostic "red anywhere" — pass if upper OR lower
      fine colour group overlaps (used when the query named no garment, "빨간 옷").
    - **lower_shape**: coarse shape (long_pants/shorts/long_skirt/short_skirt),
      derived from the fine lower_type label for pre-redesign rows; unknown
      shape wildcard-passes. Garment FABRIC (jeans vs slacks) is NOT gated here
      — it is a refine-stage signal (redesign 2026-06).
    - **upper_sleeve / upper_type group**: sleeve(long/short) exact, and the
      upper garment GROUP (outerwear/top/uniform/one_piece) so "코트" excludes
      tshirts; the fine upper_type (coat vs jacket) is a refine signal. Unknown
      wildcard-passes.

    Returns False to reject. Empty query slots are no-ops.
    """
    from plr_schema import color_hard_group, lower_shape_of, upper_type_group

    attrs = candidate_plr.get("attributes") or {}

    # --- Strict identity slots ------------------------------------------
    _SLOT_TO_PLR_KEY = {
        "gender": "gender_scores",
        "age_group": "age_group_scores",
        "action": "static_action_state_scores",
    }
    for slot in HARD_FILTER_SLOTS:
        q_vals = query_required.get(slot)
        if not q_vals:
            continue
        # rider_vehicle attributes live in a different shape than the
        # other selected-style slots — handle them specifically.
        if slot == "rider_vehicle_type":
            rv = attrs.get("rider_vehicle") or {}
            rv_type = rv.get("type")
            if rv_type and rv_type not in q_vals:
                return False
            if not rv_type:
                # PLR didn't populate rider_vehicle for this row, so it
                # cannot satisfy a rider-specific query. Reject.
                return False
            continue
        if slot == "rider_vehicle_color":
            rv = attrs.get("rider_vehicle") or {}
            cand = [t.get("label") for t in (rv.get("color_topk") or [])[:2]
                    if isinstance(t, dict)]
            if not cand:
                return False
            q_groups = {color_hard_group(v) for v in q_vals if v}
            c_groups = {color_hard_group(v) for v in cand if v}
            if not (q_groups & c_groups):
                return False
            continue
        plr_key = _SLOT_TO_PLR_KEY.get(slot)
        if not plr_key:
            continue
        d = attrs.get(plr_key) or {}
        sel = d.get("selected")
        if sel and sel not in q_vals:
            return False

    # --- Coarse-group slots ---------------------------------------------
    # Helper: collect topk labels (top-2) for a slot. Top-2 is a tighter
    # filter than top-3 — the user explicitly wanted PLR's strongest
    # picks to drive matching, not the long tail. Top-1 covers the
    # confident case; top-2 catches the close runner-up so a coat that
    # PLR labels gray/dark_gray still passes a "회색" query.
    def _topk_labels(parent: str, key: str) -> list[str]:
        node = attrs.get(parent) or {}
        topk = node.get(key) or []
        return [t.get("label") for t in topk[:2] if isinstance(t, dict)]

    def _group_overlap(q_labels: list[str], cand_labels: list[str],
                       grouper) -> bool:
        if not q_labels or not cand_labels:
            return True  # nothing to check
        q_groups = {grouper(v) for v in q_labels if v}
        c_groups = {grouper(v) for v in cand_labels if v}
        return bool(q_groups & c_groups)

    # Person colour / shape groups — only when the row is a person.
    # Reliable HARD axes only (redesign 2026-06): colour groups + coarse
    # lower SHAPE. Garment FABRIC (upper_type / lower_type) is NOT gated
    # here — it is a refine-stage signal.
    if candidate_plr.get("object_type") == "person":
        if (q := query_required.get("upper_color")):
            if not _group_overlap(q, _topk_labels("upper_clothing", "color_topk"),
                                  color_hard_group):
                return False
        if (q := query_required.get("lower_color")):
            if not _group_overlap(q, _topk_labels("lower_clothing", "color_topk"),
                                  color_hard_group):
                return False
        # any_color = garment-agnostic "red anywhere" (query named no garment,
        # e.g. "빨간 옷"). Pass if EITHER upper OR lower colour group overlaps.
        # Empty/unknown garment colours wildcard-pass inside _group_overlap,
        # so this never rejects on missing data (recall-preserving).
        if (q := query_required.get("any_color")):
            up_ok = _group_overlap(q, _topk_labels("upper_clothing", "color_topk"),
                                   color_hard_group)
            lo_ok = _group_overlap(q, _topk_labels("lower_clothing", "color_topk"),
                                   color_hard_group)
            if not (up_ok or lo_ok):
                return False
        # Lower SHAPE (긴바지/반바지/긴치마/짧은치마). Prefer the extracted
        # `lower_shape` field; for pre-redesign rows derive it from the fine
        # lower_type top-1 label (read-time, no reindex). Unknown shape
        # wildcard-passes; exact (non-group) match otherwise.
        if (q := query_required.get("lower_shape")):
            cand_shape = attrs.get("lower_shape")
            if not cand_shape:
                lo = _topk_labels("lower_clothing", "type_topk")
                cand_shape = lower_shape_of(lo[0]) if lo else "lower_unknown"
            if cand_shape != "lower_unknown" and cand_shape not in q:
                return False
        # Upper SLEEVE length (reliable axis). Extracted field on reindexed
        # rows (plr_prompts adds `upper.sleeve`); absent/unknown on old rows ->
        # wildcard-pass. Exact match otherwise.
        if (q := query_required.get("upper_sleeve")):
            cand_sleeve = (attrs.get("upper_clothing") or {}).get("sleeve")
            if cand_sleeve and cand_sleeve != "unknown" and cand_sleeve not in q:
                return False
        # Upper garment GROUP (outerwear / top / uniform / one_piece) — the
        # reliable coarse axis. "코트" must not surface tshirts. The FINE type
        # (coat vs jacket) is a refine signal, not gated here. upper_unknown /
        # empty candidate labels wildcard-pass (recall-preserving).
        if (q := query_required.get("upper_type")):
            cand_up = [
                l for l in _topk_labels("upper_clothing", "type_topk")
                if l and l != "upper_unknown"
            ]
            if not _group_overlap(q, cand_up, upper_type_group):
                return False

    # Vehicle color/type groups — vehicle attributes live directly under
    # `attributes` (no upper/lower split).
    elif candidate_plr.get("object_type") == "vehicle":
        from plr_schema import vehicle_type_group
        cand_colors = [t.get("label") for t in (attrs.get("color_topk") or [])[:2]
                       if isinstance(t, dict)]
        cand_types = [t.get("label") for t in (attrs.get("type_topk") or [])[:2]
                      if isinstance(t, dict)]
        if (q := query_required.get("vehicle_color")):
            # Dominant-colour hard gate on the FINE colour group (user choice
            # 2026-07): "검은색 차" matches black + dark_gray only, never the
            # coarse dark band (dark_brown / dark_green / military_olive) nor a
            # car whose #2 colour happened to be dark. Top-1 colour only;
            # unlabelled crops (cand_top1 None) wildcard-pass for recall.
            cand_top1 = cand_colors[0] if cand_colors else None
            if cand_top1 and color_hard_group(cand_top1) not in {
                color_hard_group(v) for v in q
            }:
                return False
        if (q := query_required.get("vehicle_type")):
            # An earlier revision (plan §3.19, v0.7_cot) treated
            # vehicle_unknown as a wildcard because Gemma E4B labelled
            # nearly every car as unknown. With the v1.0+ prompts
            # vehicle_unknown is down to ~1% of indexed vehicles, and
            # the wildcard now produces the opposite problem — minivans
            # and hatchbacks that PLR couldn't classify confidently
            # slip into precise queries like "흰색 트럭". Drop the
            # wildcard and require an actual group overlap. The handful
            # of true mis-classifications are an acceptable trade.
            # Broad-target queries are unaffected because they don't
            # populate vehicle_type at all (so this block is skipped).
            if not _group_overlap(q, cand_types, vehicle_type_group):
                return False

    # --- Excluded slots (negation hard reject) -------------------------
    # The user explicitly asked NOT to see X ("모자 안 쓴 사람" →
    # excluded.equipment=[hat]). If the candidate clearly carries X,
    # drop it before scoring even runs. This is a stricter version of
    # the EXCLUDED_PENALTY applied in compute_final_score: hard
    # rejection prevents penalised-but-still-top-ranked false matches.
    if query_excluded:
        from plr_schema import EQUIPMENT_TYPE_GROUP  # local import
        for slot, vals in query_excluded.items():
            if not vals:
                continue
            if slot == "equipment":
                eq_types = [
                    e.get("type") for e in (attrs.get("equipment") or [])
                    if isinstance(e, dict) and e.get("type")
                ]
                if any(v in eq_types for v in vals):
                    return False
            elif slot == "vehicle_type":
                cand_types = [
                    t.get("label") for t in (attrs.get("type_topk") or [])[:2]
                    if isinstance(t, dict)
                ]
                if any(v in cand_types for v in vals):
                    return False
            elif slot == "upper_type":
                cand = [t.get("label") for t in
                        ((attrs.get("upper_clothing") or {}).get("type_topk") or [])[:2]
                        if isinstance(t, dict)]
                if any(v in cand for v in vals):
                    return False
            elif slot == "lower_type":
                cand = [t.get("label") for t in
                        ((attrs.get("lower_clothing") or {}).get("type_topk") or [])[:2]
                        if isinstance(t, dict)]
                if any(v in cand for v in vals):
                    return False
            elif slot in ("upper_color", "lower_color", "vehicle_color"):
                # Negated colour: drop candidates whose dominant colour
                # group matches the negated one. Group-level (gray vs
                # dark_gray both → "gray") keeps recall reasonable —
                # "검은 옷 안 입은" should reject a candidate labelled
                # dark_gray too, since that visually IS the colour the
                # user wants to avoid.
                if slot == "upper_color":
                    cand = [t.get("label") for t in
                            ((attrs.get("upper_clothing") or {}).get("color_topk") or [])[:2]
                            if isinstance(t, dict)]
                elif slot == "lower_color":
                    cand = [t.get("label") for t in
                            ((attrs.get("lower_clothing") or {}).get("color_topk") or [])[:2]
                            if isinstance(t, dict)]
                else:  # vehicle_color
                    cand = [t.get("label") for t in (attrs.get("color_topk") or [])[:2]
                            if isinstance(t, dict)]
                q_groups = {color_hard_group(v) for v in vals if v}
                c_groups = {color_hard_group(v) for v in cand if v}
                if q_groups & c_groups:
                    return False
            # gender / action remain soft-penalty only.

    return True


def passes_strict_filter(
    query_required: dict[str, list[str]],
    candidate_plr: dict[str, Any],
    is_vehicle: bool,
) -> bool:
    """When strict=true in the API request, require every required slot to match.

    "Match" here means slot score > 0 (i.e. the candidate has at least one of
    the requested values somewhere, either as selected or in top-k).
    """
    if not query_required:
        return True
    attrs = candidate_plr.get("attributes") or {}
    for slot, vals in query_required.items():
        if not vals:
            continue
        slot_score = _slot_match_score(slot, vals, attrs)
        if slot_score <= 0:
            return False
    return True


# =====================================================================
# Refine stage (redesign 2026-06)
# =====================================================================

# The fine sub-type distinction the hard gate intentionally skips (jeans vs
# slacks, dress-shirt vs tee). The cheap first pass uses the PLR label; only
# 'uncertain' candidates are escalated to a VQA call by the caller.
REFINE_LABEL_CONFIDENT = 0.6   # a top-1 PLR score at/above this is "confident"

_REFINE_SLOT_TO_NODE: dict[str, tuple[str, str]] = {
    "lower_type": ("lower_clothing", "type_topk"),
    "upper_type": ("upper_clothing", "type_topk"),
}

# Generic "parent" labels that DON'T contradict a more specific sibling query.
# PLR very often labels a denim jean as the generic "pants" (it didn't commit to
# the fabric), so a confident "pants" vs a "jeans"/"slacks" query is UNCERTAIN
# (could be the queried sub-type — let VQA decide), NOT a confident mismatch.
# (Oracle check of real "청바지" drops: every one was a generic "pants", none a
# confident slacks — so treating "pants" as mismatch was a false-drop trap.)
_GENERIC_REFINE_LABELS: dict[str, frozenset[str]] = {
    "lower_type": frozenset({"pants"}),
}


def refine_verdict(
    candidate_plr: dict[str, Any], slot: str, value: str, *, topk: int = 3
) -> str:
    """Cheap PLR-label verdict for one fine refine signal.

    Returns 'match' | 'mismatch' | 'uncertain':
      - match     : value is present in the candidate's top-k for the slot
                    (recall-preferring — PLR considers it plausible).
      - mismatch  : value is absent AND a DIFFERENT SPECIFIC label is the
                    confident top-1 (PLR is sure it's something else). A
                    confident GENERIC parent (e.g. "pants") -> uncertain, not
                    mismatch (it might still be the queried sub-type).
      - uncertain : low confidence / no data → caller should escalate to VQA.
    """
    attrs = candidate_plr.get("attributes") or {}
    loc = _REFINE_SLOT_TO_NODE.get(slot)
    if not loc:
        return "uncertain"
    node = attrs.get(loc[0]) or {}
    arr = node.get(loc[1]) or []
    labels = [
        (t.get("label"), float(t.get("score", 0.0)))
        for t in arr[:topk]
        if isinstance(t, dict)
    ]
    if not labels:
        return "uncertain"
    top_label, top_score = labels[0]
    label_set = {lbl for lbl, _ in labels}
    if value in label_set:
        # Present in top-k → plausible. Confident only if PLR is reasonably sure.
        return "match" if top_score >= REFINE_LABEL_CONFIDENT else "uncertain"
    # Absent from top-k.
    if top_score >= REFINE_LABEL_CONFIDENT:
        # A confident GENERIC label (e.g. "pants") doesn't contradict a specific
        # sub-type query (jeans/slacks) — it just didn't commit. Treat as
        # uncertain (VQA can decide) instead of dropping it.
        if top_label in _GENERIC_REFINE_LABELS.get(slot, frozenset()):
            return "uncertain"
        return "mismatch"
    return "uncertain"
