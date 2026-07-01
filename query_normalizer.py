"""Normalise the raw entity-extraction JSON from qp_v0.4 into PLR enum values.

Background
----------
qp_v0.3 asked Gemma to do two things at once: pull visual concepts out
of the Korean query AND map them to specific English enum codes (black,
backpack, walking_like, ...). The combination was unstable on a 4B model
because Rule 4 (normalize) and Rule 7 (residue) compete — the model
hedges by emitting the same word in both slots.

qp_v0.4 cuts that knot. The Gemma prompt now extracts clean Korean
visual concepts only; this module turns those concepts into the
QueryJSON shape downstream code expects:
    required={"upper_color": ["black"], "equipment": [...], ...}
    free_form_residue=["호피무늬", ...]

Mapping strategy
----------------
- Per-slot synonym tables (the same ones the legacy parse_with_dictionary
  uses, re-exported here as the source of truth).
- Multi-word values ("갈색 가방", "어두운 빨강") are split on whitespace,
  each token is looked up independently, and a known colour token
  attaches to the noun it precedes/follows.
- Equipment group keywords ("가방" → bag-group) expand to the full
  member list, matching parse_with_dictionary's behaviour.
- Anything that survives every table without matching lands in
  free_form_residue.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Re-use the dictionaries that the legacy dict-based parser already
# maintains. Importing rather than duplicating keeps the two paths in
# sync — if someone adds a synonym for parse_with_dictionary it
# automatically applies to qp_v0.4 too.
from query_parser import (
    _COLOR_SYNONYMS,
    _UPPER_TYPE_SYNONYMS,
    _LOWER_TYPE_SYNONYMS,
    _EQUIPMENT_SYNONYMS,
    _EQUIPMENT_GROUP_KEYWORDS,
    _ACTION_SYNONYMS,
    _GENDER_SYNONYMS,
    _AGE_SYNONYMS,
    _OUTFIT_TYPE_SYNONYMS,
    _VEHICLE_TYPE_SYNONYMS,
)
from plr_schema import EQUIPMENT_TYPE_GROUP


# ---------------------------------------------------------------------
# Negation detection (Wave 3, FR-13.3)
# ---------------------------------------------------------------------
# Korean queries express negation in a handful of recognisable shapes:
#   "X 안 V"     — "모자 안 쓴 사람"        ("not wearing a hat")
#   "X 안 V는"   — "안 입은", "안 멘"
#   "X 없는"     — "배달통 없는 오토바이"
#   "X 없이"     — "헬멧 없이 탄"
#   "X 미착용"   — "마스크 미착용"
# plus the obvious English forms. We do NOT try a full parse — a few
# anchored regexes give us recall on the patterns we actually see.
_NEGATION_VERBS = ("쓴", "쓰는", "쓰고", "쓰지",
                   "입은", "입는", "입고",
                   "신은", "신는", "신고",
                   "멘", "메는", "메고",
                   "들고", "들은", "든",
                   "차고",
                   "탄", "타고",
                   "걸친", "걸치고")
# 단형 부정 — "안 V"
_NEG_VERB_RE = re.compile(
    r"안\s*(?:" + "|".join(re.escape(v) for v in _NEGATION_VERBS) + r")"
)
# 장형 부정 — "V-지 않-" (입지 않은 / 쓰지 않아요 / 메지 않고 / …)
_NEG_LONG_STEMS = ("입지", "쓰지", "신지", "메지", "들지", "차지", "타지",
                   "걸치지", "하지")
_NEG_LONG_RE = re.compile(
    r"(?:" + "|".join(re.escape(v) for v in _NEG_LONG_STEMS) + r")"
    r"\s*않(?:은|는|아|아요|았|으면|고|지)?"
)
_NEG_SUFFIX_RE = re.compile(r"(없는|없이|없음|미착용|안한|아닌)")
_NEG_EN_RE = re.compile(r"\b(without|no|not\s+wearing|not\s+carrying)\b", re.IGNORECASE)


def _detect_negation(text: str | None) -> bool:
    """True if the given fragment carries a negation cue.

    Kept conservative: a fragment must literally contain one of the
    negation surface forms above. Long-form queries are scanned per
    fragment (e.g. attributes.equipment_details and the user query as
    a whole), so an unrelated "안" inside a non-negation word ("안내문"
    etc.) doesn't trip the detector because we anchor on verb stems.
    """
    if not text:
        return False
    t = str(text)
    if _NEG_VERB_RE.search(t):
        return True
    if _NEG_LONG_RE.search(t):
        return True
    if _NEG_SUFFIX_RE.search(t):
        return True
    if _NEG_EN_RE.search(t):
        return True
    return False


# Korean noun → category-of-enum classification, used to decide whether
# a negated noun should be routed to `excluded.<slot>` or kept as a
# free-form residue with is_negative=true.
def _enum_slot_for_noun(noun: str) -> tuple[str, str] | None:
    """Return (slot_name, enum_value) if `noun` maps to a structured
    enum, else None. Slot names match the QueryJSON.required keys."""
    t = _norm(noun)
    if not t:
        return None
    if (v := _lookup_first(t, _EQUIPMENT_SYNONYMS)):
        return ("equipment", v)
    if (v := _lookup_first(t, _VEHICLE_TYPE_SYNONYMS)):
        return ("vehicle_type", v)
    if (v := _lookup_first(t, _UPPER_TYPE_SYNONYMS)):
        return ("upper_type", v)
    if (v := _lookup_first(t, _LOWER_TYPE_SYNONYMS)):
        return ("lower_type", v)
    if (v := _lookup_first(t, _COLOR_SYNONYMS)):
        # Colour negation is too noisy on its own (we wouldn't know
        # whether it's upper, lower, or vehicle). Leave it to a future
        # iteration.
        return None
    return None


def _norm(text: Any) -> str:
    """Coerce arbitrary Gemma output (None / dict / list / number) to a
    lowercase stripped string. Gemma occasionally wraps a value in a
    dict or returns the literal string "null"; this keeps the rest of
    the normaliser from crashing."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        return " ".join(_norm(x) for x in text)
    if isinstance(text, dict):
        # Sometimes Gemma emits {"value": "검정"} — flatten the values.
        return " ".join(_norm(v) for v in text.values())
    s = str(text).strip().lower()
    if s in {"null", "none", "n/a"}:
        return ""
    return s


def _lookup_first(text: str | None, table: dict[str, str]) -> str | None:
    """Return the enum value of the longest synonym surface inside `text`.

    Longest-first iteration prevents "검은색" → black AND "은색" → silver
    double-matching, same fix as scoring._scan_dict.
    """
    t = _norm(text)
    if not t:
        return None
    for surface in sorted(table.keys(), key=len, reverse=True):
        if surface in t:
            return table[surface]
    return None


def _lookup_all(text: str | None, table: dict[str, str]) -> list[str]:
    """Return every enum value whose synonym appears in `text` (consumed)."""
    t = _norm(text)
    if not t:
        return []
    hits: list[str] = []
    remaining = t
    for surface in sorted(table.keys(), key=len, reverse=True):
        canonical = table[surface]
        if surface in remaining and canonical not in hits:
            hits.append(canonical)
            remaining = remaining.replace(surface, " ")
    return hits


def _extract_colour_and_noun(
    detail: str | None,
    noun_table: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Pull a colour list + noun-enum list out of free-form "<colour> <noun>".

    Used for equipment_details ("갈색 가방") and similar colour-bound
    phrases. Either side may be empty if the user only wrote one.
    """
    colours = _lookup_all(detail, _COLOR_SYNONYMS)
    nouns = _lookup_all(detail, noun_table)
    return colours, nouns


def _expand_equipment_groups(text: str | None, equipment: list[str]) -> list[str]:
    """If the raw text contains a group keyword (가방/무기/...), expand the
    equipment list to the group's full membership — matches the
    behaviour the legacy dictionary parser already provides.
    """
    t = _norm(text)
    if not t:
        return equipment
    expanded = list(equipment)
    for grp_kw, grp_name in _EQUIPMENT_GROUP_KEYWORDS.items():
        if grp_kw in t:
            for member in EQUIPMENT_TYPE_GROUP.get(grp_name, ()):
                if member not in expanded:
                    expanded.append(member)
    return expanded


def normalize_query_entities(
    raw: dict[str, Any],
    user_text: str,
) -> dict[str, Any]:
    """Turn the qp_v0.4 entity JSON into the QueryJSON-style shape.

    Returns a dict with the same top-level keys QueryJSON expects:
        {
          "query_type": ...,
          "target": ...,
          "required": {slot: [enum, ...], ...},
          "optional": {},
          "excluded": {},
          "free_form_residue": [...],
          "template_caption_en": ...,
          "template_caption_ko": ...,
        }
    `parse_with_gemma`'s caller wraps this into a QueryJSON dataclass.
    """
    attrs = raw.get("attributes") or {}
    required: dict[str, list[str]] = {}

    # --- person identity ---
    if (g := _lookup_first(attrs.get("gender"), _GENDER_SYNONYMS)):
        required["gender"] = [g]
    if (a := _lookup_first(attrs.get("age_group"), _AGE_SYNONYMS)):
        required["age_group"] = [a]
    if (o := _lookup_first(attrs.get("outfit_type"), _OUTFIT_TYPE_SYNONYMS)):
        required["outfit_type"] = [o]

    # --- upper clothing ---
    upper_colours = _lookup_all(attrs.get("upper_color"), _COLOR_SYNONYMS)
    upper_types = _lookup_all(attrs.get("upper_clothing"), _UPPER_TYPE_SYNONYMS)
    if upper_colours:
        required["upper_color"] = upper_colours
    if upper_types:
        required["upper_type"] = upper_types

    # --- lower clothing ---
    lower_colours = _lookup_all(attrs.get("lower_color"), _COLOR_SYNONYMS)
    lower_types = _lookup_all(attrs.get("lower_clothing"), _LOWER_TYPE_SYNONYMS)
    if lower_colours:
        required["lower_color"] = lower_colours
    if lower_types:
        required["lower_type"] = lower_types

    # --- garment-agnostic colour ("빨간 옷": colour with no specific garment) ---
    # Routed to the any_color OR gate (redesign 2026-06): matches if EITHER the
    # upper OR lower garment colour group overlaps. Parity with the dictionary
    # path's `any_colors` bucket.
    any_colours = _lookup_all(attrs.get("any_color"), _COLOR_SYNONYMS)
    if any_colours:
        required["any_color"] = any_colours

    # --- equipment + colour-bound equipment ---
    equipment = []
    raw_equipment = attrs.get("equipment") or []
    if isinstance(raw_equipment, list):
        for item in raw_equipment:
            for hit in _lookup_all(item, _EQUIPMENT_SYNONYMS):
                if hit not in equipment:
                    equipment.append(hit)
    equipment = _expand_equipment_groups(user_text, equipment)

    # equipment_details holds "<colour> <equipment>" or
    # "<pattern> <equipment>" phrases. Colour → equipment_color slot
    # (fixes 갈색 가방 → upper_color hallucination). Any tokens that
    # match neither a colour nor an equipment noun (호피무늬, 줄무늬,
    # logo, etc.) become residue so the free-form VQA stage picks
    # them up.
    eq_details = attrs.get("equipment_details")
    eq_details_str = _norm(eq_details)
    eq_colours, eq_nouns = _extract_colour_and_noun(
        eq_details, _EQUIPMENT_SYNONYMS,
    )
    for n in eq_nouns:
        if n not in equipment:
            equipment.append(n)
    equipment = _expand_equipment_groups(user_text, equipment)
    if equipment:
        required["equipment"] = equipment
    if eq_colours:
        required["equipment_color"] = eq_colours

    # Pull unmatched tokens out of equipment_details into a residue
    # pool for the post-processing step below.
    eq_details_residue: list[str] = []
    if eq_details_str:
        consumed = eq_details_str
        # remove tokens that matched a colour or an equipment noun
        for surface in sorted(_COLOR_SYNONYMS.keys(), key=len, reverse=True):
            if surface in consumed:
                consumed = consumed.replace(surface, " ")
        for surface in sorted(_EQUIPMENT_SYNONYMS.keys(), key=len, reverse=True):
            if surface in consumed:
                consumed = consumed.replace(surface, " ")
        for surface in sorted(_EQUIPMENT_GROUP_KEYWORDS.keys(), key=len, reverse=True):
            if surface in consumed:
                consumed = consumed.replace(surface, " ")
        for tok in consumed.split():
            tok = tok.strip()
            if len(tok) >= 2 and not tok.isdigit():
                eq_details_residue.append(tok)

    # --- action / posture ---
    if (act := _lookup_first(attrs.get("action_or_posture"), _ACTION_SYNONYMS)):
        required["action"] = [act]

    # --- vehicle ---
    if (vc := _lookup_first(attrs.get("vehicle_color"), _COLOR_SYNONYMS)):
        required["vehicle_color"] = [vc]
    if (vt := _lookup_first(attrs.get("vehicle_type"), _VEHICLE_TYPE_SYNONYMS)):
        required["vehicle_type"] = [vt]

    # --- target ---
    target = (raw.get("target") or "").strip().lower() or "unknown"
    if target not in {"person", "vehicle", "event", "mixed"}:
        # Derive from required slots if Gemma didn't pick one.
        if required.get("vehicle_color") or required.get("vehicle_type"):
            target = "vehicle"
        elif any(k in required for k in (
            "upper_color", "upper_type", "lower_color", "lower_type",
            "equipment", "gender", "age_group", "outfit_type", "action",
        )):
            target = "person"
        else:
            target = "person"
    query_type = {
        "person": "person_search",
        "vehicle": "vehicle_search",
        "event": "event_search",
        "mixed": "mixed_search",
    }.get(target, "person_search")

    # Generic head nouns that the *_CONTEXT_RE in parse_with_dictionary
    # already routes elsewhere. If Gemma echoes them into residue we
    # drop them so the VQA stage doesn't spuriously activate.
    _GENERIC_HEADS = frozenset({
        "사람", "남자", "여자", "남성", "여성", "어른", "아이", "행인",
        "보행자", "person", "man", "woman",
        "차", "차량", "자동차", "vehicle", "car",
        "옷", "옷차림", "차림",
    })

    # --- free-form residue (context-bound objects) ---
    # qp_v0.5 residue is a list of {"subject": "...", "attribute": "..."}
    # objects. We accept the new shape and gracefully upgrade older
    # bare-string entries (and any rescued equipment_details tokens)
    # by attaching a sensible subject derived from the parsed target /
    # equipment so the downstream VQA stage always has a noun to ask
    # about ("Is this <subject> <attribute>?").
    def _default_subject() -> str:
        # equipment-bound residue (호피무늬 → bag): if we have an
        # equipment slot, use the first member's group keyword.
        if equipment:
            first = equipment[0]
            for grp_kw, grp_name in _EQUIPMENT_GROUP_KEYWORDS.items():
                members = EQUIPMENT_TYPE_GROUP.get(grp_name, ())
                if first in members:
                    # Return the English head noun (bag / headwear / weapon).
                    return grp_name
            return first
        if target == "vehicle":
            vt = required.get("vehicle_type") or []
            if vt:
                return vt[0]
            return "vehicle"
        if target == "person":
            return "person"
        return target

    _GENERIC_HEADS_STRICT = _GENERIC_HEADS  # alias for readability

    # --- excluded slot (negation routing) ------------------------------
    # Two paths feed `excluded`:
    #   (a) Gemma's own `excluded` block from the qp_v0.6+ output.
    #   (b) Whole-query scan: any enum noun that sits next to a Korean
    #       negation cue ("X 안 V", "X 없는", …) gets pulled out of
    #       `required` (if Gemma misclassified it) and into `excluded`.
    excluded: dict[str, list[str]] = {}
    raw_excluded = raw.get("excluded") or {}
    if isinstance(raw_excluded, dict):
        for slot, vals in raw_excluded.items():
            if not isinstance(vals, list):
                vals = [vals]
            mapped: list[str] = []
            for v in vals:
                hit = _lookup_first(v, _EQUIPMENT_SYNONYMS) \
                    or _lookup_first(v, _VEHICLE_TYPE_SYNONYMS) \
                    or _lookup_first(v, _UPPER_TYPE_SYNONYMS) \
                    or _lookup_first(v, _LOWER_TYPE_SYNONYMS) \
                    or _lookup_first(v, _COLOR_SYNONYMS)
                if hit and hit not in mapped:
                    mapped.append(hit)
            if mapped:
                slot_norm = slot.strip().lower()
                # Map common alias names; tolerate either "equipment" or
                # "equipment_type".
                if slot_norm in {"equipment", "equipment_type"}:
                    slot_norm = "equipment"
                excluded.setdefault(slot_norm, [])
                for v in mapped:
                    if v not in excluded[slot_norm]:
                        excluded[slot_norm].append(v)

    # Whole-query negation scan: catch "모자 안 쓴 사람" where Gemma
    # might still emit equipment=[모자]. Anchored on the negation
    # regex AND a neighbouring enum noun (within a few characters).
    if (_NEG_VERB_RE.search(user_text)
            or _NEG_LONG_RE.search(user_text)
            or _NEG_SUFFIX_RE.search(user_text)):
        # Window-based scan. For each Korean enum surface in the query
        # that is followed (within ~8 chars) by a negation cue, treat
        # the surface as excluded. Group keywords ("가방"/"모자") expand
        # to their full member list under the canonical equipment slot.
        eq_group_kw = dict(_EQUIPMENT_GROUP_KEYWORDS)
        for noun_table, slot_name in (
            (_EQUIPMENT_SYNONYMS, "equipment"),
            (eq_group_kw, "equipment_group"),
            (_VEHICLE_TYPE_SYNONYMS, "vehicle_type"),
            (_UPPER_TYPE_SYNONYMS, "upper_type"),
            (_LOWER_TYPE_SYNONYMS, "lower_type"),
            (_COLOR_SYNONYMS, "color"),
        ):
            for surface in sorted(noun_table.keys(), key=len, reverse=True):
                idx = user_text.find(surface)
                if idx < 0:
                    continue
                window = user_text[idx:idx + len(surface) + 8]
                if not _detect_negation(window):
                    continue
                mapped = noun_table[surface]
                if slot_name == "equipment_group":
                    members = EQUIPMENT_TYPE_GROUP.get(mapped, ())
                    excluded.setdefault("equipment", [])
                    for m in members:
                        if m not in excluded["equipment"]:
                            excluded["equipment"].append(m)
                        if "equipment" in required and m in required["equipment"]:
                            required["equipment"] = [
                                v for v in required["equipment"] if v != m
                            ]
                            if not required["equipment"]:
                                del required["equipment"]
                    continue
                if slot_name == "color":
                    enum_val = mapped
                    nxt = user_text[idx + len(surface):idx + len(surface) + 6]
                    is_vehicle_neighbour = any(
                        vw in nxt for vw in ("차량", "자동차", "오토바이",
                                            "트럭", "버스", "세단", "SUV",
                                            "suv", "밴", "택시")
                    ) or nxt.strip().startswith("차")
                    if is_vehicle_neighbour:
                        target_slot = "vehicle_color"
                    else:
                        target_slot = None
                        for color_slot in ("upper_color", "lower_color", "vehicle_color"):
                            if color_slot in required and enum_val in required[color_slot]:
                                target_slot = color_slot
                                break
                        if target_slot is None:
                            target_slot = (
                                "vehicle_color" if target == "vehicle"
                                else "upper_color"
                            )
                    for color_slot in ("upper_color", "lower_color", "vehicle_color"):
                        if color_slot in required and enum_val in required[color_slot]:
                            required[color_slot] = [
                                v for v in required[color_slot] if v != enum_val
                            ]
                            if not required[color_slot]:
                                del required[color_slot]
                    excluded.setdefault(target_slot, [])
                    if enum_val not in excluded[target_slot]:
                        excluded[target_slot].append(enum_val)
                    continue
                enum_val = mapped
                excluded.setdefault(slot_name, [])
                if enum_val not in excluded[slot_name]:
                    excluded[slot_name].append(enum_val)
                # If Gemma also put it in required, remove it.
                if slot_name in required and enum_val in required[slot_name]:
                    required[slot_name] = [
                        v for v in required[slot_name] if v != enum_val
                    ]
                    if not required[slot_name]:
                        del required[slot_name]

    # --- free-form residue (context-bound + polarity-aware) -----------
    # qp_v0.6+ shape:
    #   [{"subject", "attribute_ko", "attribute_en", "is_negative"}, ...]
    # We accept qp_v0.5 ({"subject", "attribute"}) and legacy bare
    # strings too — both upgraded in place with sensible defaults.
    raw_residue = list(raw.get("free_form_residue") or [])
    # Promote rescued equipment_details tokens to objects too. They
    # inherit the equipment-group subject and have no attribute_en /
    # negation by definition.
    for tok in eq_details_residue:
        raw_residue.append({
            "subject": _default_subject(),
            "attribute_ko": tok,
            "attribute_en": "",
            "is_negative": False,
        })

    residue: list[dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()
    for item in raw_residue:
        if isinstance(item, dict):
            subj = _norm(item.get("subject")) or _default_subject()
            attr_ko = (str(item.get("attribute_ko")
                          or item.get("attribute")
                          or "")).strip()
            attr_en = (str(item.get("attribute_en") or "")).strip()
            is_neg = bool(item.get("is_negative", False))
        else:
            attr_ko = (str(item) or "").strip()
            attr_en = ""
            subj = _default_subject()
            is_neg = False
        if not (attr_ko or attr_en):
            continue
        key_ko = attr_ko.lower()
        if key_ko and key_ko in _GENERIC_HEADS_STRICT:
            continue
        # Drop residue items that the synonym tables can already place
        # in a structured slot — they belong in `required` / `excluded`,
        # not in residue. Only the Korean form is checked because that's
        # what the dictionaries know about.
        if attr_ko and (
            _lookup_first(attr_ko, _EQUIPMENT_SYNONYMS)
            or _lookup_first(attr_ko, _GENDER_SYNONYMS)
            or _lookup_first(attr_ko, _VEHICLE_TYPE_SYNONYMS)
            or _lookup_first(attr_ko, _UPPER_TYPE_SYNONYMS)
            or _lookup_first(attr_ko, _LOWER_TYPE_SYNONYMS)
            or _lookup_first(attr_ko, _COLOR_SYNONYMS)
            or _lookup_first(attr_ko, _ACTION_SYNONYMS)
            or _lookup_first(attr_ko, _AGE_SYNONYMS)
            or _lookup_first(attr_ko, _OUTFIT_TYPE_SYNONYMS)
            or _lookup_first(attr_ko, _EQUIPMENT_GROUP_KEYWORDS)
        ):
            continue
        # Per-item negation detection — Gemma may forget the flag.
        if not is_neg and _detect_negation(attr_ko):
            is_neg = True
        key = (subj, key_ko or attr_en.lower(), is_neg)
        if key in seen:
            continue
        seen.add(key)
        residue.append({
            "subject": subj,
            "attribute_ko": attr_ko,
            "attribute_en": attr_en,
            "is_negative": is_neg,
        })

    return {
        "query_type": query_type,
        "target": target,
        "required": required,
        "optional": {},
        "excluded": excluded,
        "ambiguous": [],
        "free_form_residue": residue,
        "needs_temporal_search": False,
        "template_caption_en": raw.get("raw_clean_query") or user_text,
        "template_caption_ko": user_text,
    }
