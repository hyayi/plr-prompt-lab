"""Prompts and JSON parser for the Gemma 4 PLR backend.

Two prompt families:
  1. PLR extraction (image → structured JSON of person/vehicle attributes)
  2. Query parser (user text → required/optional/excluded slots)

Both prompts enforce English-only output for internal consistency. Korean
synonyms are normalized to the English enum by the dictionary in
query_parser.py (separate file).

The parser tolerates the most common Gemma format slips:
  - Wraps JSON in ```json ... ``` fences
  - Trailing commas
  - Smart quotes
  - Leading prose ("Here is the JSON:")
"""

from __future__ import annotations

import json
import re
from typing import Any

from plr_schema import (
    AGE_GROUP_ENUM,
    COLOR_ENUM,
    EQUIPMENT_TYPE_ENUM,
    GENDER_ENUM,
    LOWER_TYPE_ENUM,
    MILITARY_ENUM,
    OUTFIT_TYPE_ENUM,
    PROMPT_VERSION,
    STATIC_ACTION_ENUM,
    UPPER_TYPE_ENUM,
    VEHICLE_TYPE_ENUM,
)


# =====================================================================
# PLR extraction prompt
# =====================================================================

_PLR_SYSTEM_PROMPT = """Extract visual attributes from CCTV crops as JSON.

Rules:
- JSON only. No prose. No markdown. Stop immediately after the closing brace.
- Use only the listed enum values; pick "unknown" if uncertain.
- Score-distribution fields (gender, age_group, outfit_type) must include all options summing to ~1.0 and "selected" = argmax.
- topk fields return up to 3 entries sorted by score desc.
- Be conservative with gender/age — visual appearance only, not identity."""


_PLR_PERSON_USER_TEMPLATE = """Image: person crop. Output exactly this JSON shape:

{{"object_type":"person","attributes":{{
"gender_scores":{{"male":N,"female":N,"selected":"male|female","decision_margin":N}},
"age_group_scores":{{"adult":N,"child":N,"selected":"adult|child","decision_margin":N}},
"outfit_type_scores":{{"two_piece":N,"one_piece":N,"layered":N,"obscured":N,"selected":"...","decision_margin":N}},
"upper_clothing":{{"color_topk":[{{"label":"X","score":N}}],"type_topk":[{{"label":"X","score":N}}]}},
"lower_clothing":{{"color_topk":[],"type_topk":[]}},
"equipment":[{{"type":"X","score":N}}],
"static_action_state_scores":{{"selected":"X"}}
}}}}

Enums:
gender: {genders}
age_group: {ages}
outfit_type: {outfits}
color: {colors}
upper_type: {upper_types}
lower_type: {lower_types}
equipment: {equips}
static_action: {actions}

If outfit_type=one_piece: upper.type uses dress/jumpsuit/one_piece_uniform; lower.type=[{{"label":"none","score":1.0}}].
Output the JSON only and stop."""


_PLR_VEHICLE_USER_TEMPLATE = """Image: vehicle crop. Output exactly this JSON shape:

{{"object_type":"vehicle","attributes":{{
"color_topk":[{{"label":"X","score":N}}],
"type_topk":[{{"label":"X","score":N}}]
}}}}

Enums:
color: {colors}
vehicle_type: {vehicle_types}

No state field. Output the JSON only and stop."""


def plr_system_prompt() -> str:
    return _PLR_SYSTEM_PROMPT


def plr_user_prompt_person() -> str:
    return _PLR_PERSON_USER_TEMPLATE.format(
        genders=", ".join(GENDER_ENUM),
        ages=", ".join(AGE_GROUP_ENUM),
        outfits=", ".join(OUTFIT_TYPE_ENUM),
        colors=", ".join(COLOR_ENUM),
        upper_types=", ".join(UPPER_TYPE_ENUM),
        lower_types=", ".join(LOWER_TYPE_ENUM),
        equips=", ".join(EQUIPMENT_TYPE_ENUM),
        actions=", ".join(STATIC_ACTION_ENUM),
    )


def plr_user_prompt_vehicle() -> str:
    return _PLR_VEHICLE_USER_TEMPLATE.format(
        colors=", ".join(COLOR_ENUM),
        vehicle_types=", ".join(VEHICLE_TYPE_ENUM),
    )


# =====================================================================
# PLR extraction — YAML-style prompt (B-plan)
#
# JSON in v0.4 was burning ~80% of output tokens on structural formatting
# (keys, quotes, braces, escapes), and failing at char 158/267 on ~5% of
# objects when the model leaked a hedging word mid-structure. The YAML
# variant moves all values into bare key: value lines that PyYAML can
# parse robustly, and shifts saved tokens to actual content. PROMPT_VERSION
# stays plr_v0.4 unless YAML output is enabled — when it is, callers
# should record plr_prompt_version="plr_v0.5_yaml".
# =====================================================================

# v0.6_yaml (2026-07): forced-commit — the unknown escape hatch is removed from
# the answer space (gender/age/sleeve and the enum lists offered to the model).
# Low confidence is expressed via `margins`, not by refusing to answer.
PROMPT_VERSION_YAML = "plr_v0.6_yaml"
# Keep under 16 chars — tb_cache_function_response.plr_prompt_version is
# varchar(16). v0.9 = v0.8 minus the ~30 Korean/foreign car model names
# (Gemma E4B identified only 2/214 as bmw, 0 for the rest), plus a
# `light_car` (경차) bucket. Also collapses headwear enum (beanie →
# absorbed into hat) since Gemma can't reliably distinguish soft
# headwear types in CCTV crops.
# v1.3_cot (redesign 2026-06): adds the `upper.sleeve` (long/short) extraction
# field. Bumping the version makes existing rows look stale so the lazy
# per-video reindex (_index_is_fresh) re-extracts them with sleeve. (<=16 chars.)
# v1.4_cot (2026-06): reconciles the constants<->yaml drift (the military_olive
# Color hint, previously yaml-only, is now in BOTH sources) and makes military
# detection prompt-native — Gemma emits `military: <military|civilian|unknown>`
# directly from camo/uniform/gear cues instead of a post-hoc single-olive rule.
# v1.5_cot (2026-07): forced-commit contract — `unknown` is removed from every
# answer the model gives (gender/age/sleeve literals, military, and the enum
# lists injected via _commit_enum). The model must always pick the most likely
# concrete value; low confidence goes into `margins`. rider_vehicle is OMITTED
# when not riding (the parser fills its N/A sentinel). Pairs with the
# indexing-side single-view contract (quality gate and SR dual-view removed —
# every crop gets exactly one Gemma call).
PROMPT_VERSION_YAML_COT = "plr_v1.5_cot"


# ---------------------------------------------------------------------------
# Declarative prompt source — the LIVE templates are loaded from
# prompts/<PROMPT_VERSION_YAML_COT>.yaml (single source; the historical
# constants were removed 2026-07). One file = one prompt version: the same
# yaml serves the runtime default here AND the --version/provider path, so
# the constants<->yaml drift class (the ca1a922 military_olive bug) no longer
# exists. Historical experiment rationale (v0.6/v0.6.1/v0.7 rollbacks, ...)
# lives in git history and the lab's author-prompt skill "Domain lessons".
# ---------------------------------------------------------------------------
from pathlib import Path as _Path

import yaml as _yaml

_LIVE_PROMPT_PATH = _Path(__file__).resolve().parent / "prompts" / f"{PROMPT_VERSION_YAML_COT}.yaml"
with open(_LIVE_PROMPT_PATH, encoding="utf-8") as _fh:
    _LIVE_PROMPT: dict[str, Any] = _yaml.safe_load(_fh)

_PLR_YAML_COT_PERSON_TEMPLATE = _LIVE_PROMPT["plr"]["person_user"].rstrip("\n")
_PLR_YAML_PERSON_TEMPLATE = _LIVE_PROMPT["plr"]["person_user_no_reason"].rstrip("\n")
_PLR_YAML_VEHICLE_TEMPLATE = _LIVE_PROMPT["plr"]["vehicle_user"].rstrip("\n")
_PLR_YAML_SYSTEM = _LIVE_PROMPT["plr"]["system"].rstrip("\n")
_QUERY_PARSER_SYSTEM_PROMPT = _LIVE_PROMPT["query_parser"]["system"].rstrip("\n")
_QUERY_PARSER_USER_TEMPLATE = _LIVE_PROMPT["query_parser"]["user"].rstrip("\n")


def _commit_enum(values) -> tuple[str, ...]:
    """Enum values as offered to the model — the `*unknown*` escape hatches are
    excluded (plr_v1.5_cot forced-commit contract). The full enums in
    plr_schema KEEP their unknown members: they are still needed to read
    pre-v1.5 indexed rows and as defensive normalisation targets."""
    return tuple(v for v in values if "unknown" not in v)


def plr_yaml_user_prompt_person(with_reason: bool = False) -> str:
    template = _PLR_YAML_COT_PERSON_TEMPLATE if with_reason else _PLR_YAML_PERSON_TEMPLATE
    return template.format(
        colors=", ".join(_commit_enum(COLOR_ENUM)),
        upper_types=", ".join(_commit_enum(UPPER_TYPE_ENUM)),
        lower_types=", ".join(_commit_enum(LOWER_TYPE_ENUM)),
        equips=", ".join(_commit_enum(EQUIPMENT_TYPE_ENUM)),
        actions=", ".join(_commit_enum(STATIC_ACTION_ENUM)),
        military_enum="|".join(_commit_enum(MILITARY_ENUM)),
    )


def plr_yaml_user_prompt_vehicle() -> str:
    return _PLR_YAML_VEHICLE_TEMPLATE.format(
        colors=", ".join(_commit_enum(COLOR_ENUM)),
        vehicle_types=", ".join(_commit_enum(VEHICLE_TYPE_ENUM)),
        military_enum="|".join(_commit_enum(MILITARY_ENUM)),
    )


def _plr_format() -> str:
    """Choose between 'yaml' (B-plan, default) and 'json' (v0.4 fallback)."""
    import os
    v = os.environ.get("IR_PLR_FORMAT", "yaml").strip().lower()
    return v if v in {"yaml", "json"} else "yaml"


def _plr_with_reason() -> bool:
    """B1 PoC: add gender_reason / age_reason lines before the labels.
    Toggle via IR_PLR_REASON=on (off by default — extra tokens cost ~35%
    latency, so we ship CoT only when validated)."""
    import os
    v = os.environ.get("IR_PLR_REASON", "off").strip().lower()
    return v in {"on", "true", "1", "yes"}


def build_plr_messages(object_hint: str = "person") -> list[dict[str, Any]]:
    """Build chat messages for one PLR call.

    object_hint: 'person' or 'vehicle' — selects the user template.
    Format chosen by IR_PLR_FORMAT env (default: yaml).
    The image is appended by gemma_backend.generate(pil, messages).
    """
    fmt = _plr_format()
    if fmt == "yaml":
        user_text = (
            plr_yaml_user_prompt_vehicle()
            if object_hint == "vehicle"
            else plr_yaml_user_prompt_person(with_reason=_plr_with_reason())
        )
        sys_text = _PLR_YAML_SYSTEM
    else:
        # Legacy JSON path (kept for A/B comparison via env).
        user_text = (
            plr_user_prompt_vehicle()
            if object_hint == "vehicle"
            else plr_user_prompt_person()
        )
        sys_text = plr_system_prompt()
    return [
        {"role": "system", "content": sys_text},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_freeform_vqa_messages(
    residue: list[dict[str, Any]] | list[str] | str,
) -> list[dict[str, Any]]:
    """Build chat messages for a single yes/no VQA call (RFC §9 Option X).

    Context-bound, polarity-aware dynamic question. `residue` items
    carry four fields:
        subject       — head noun ("motorcycle", "bag", "person", ...)
        attribute_ko  — original Korean phrase
        attribute_en  — concrete visual English description (preferred
                        for the prompt; Korean is appended in parens
                        as a fallback hint)
        is_negative   — true when the user wants the attribute to be
                        ABSENT ("배달통 없는 오토바이"). Negative items
                        flip into a "does NOT show" clause; the call
                        still returns yes for keep / no for drop.

    Accepted input shapes (newest first):
      - qp_v0.6+: list of dicts with the four fields above.
      - qp_v0.5: list of {"subject", "attribute"} — `attribute` is
                 mapped onto attribute_ko, no attribute_en, no negation.
      - Legacy bare list[str] / str — attribute-only, no subject, no
                 negation.

    Output contract:
      - max_tokens=4, temperature=0.0.
      - Anything other than a clear "no" is treated as "yes"
        (recall-preferring). Negation is baked into the prompt, so the
        caller still does yes→keep / no→drop without per-item flipping.
    """
    pos_clauses: list[str] = []
    neg_clauses: list[str] = []

    def _clause(subj: str, attr_en: str, attr_ko: str, negate: bool) -> str:
        # Compose "the <subj> clearly shows <en> (Korean: "<ko>")"
        # — or the negated form. en is the primary description (small
        # vision models ground English visuals more reliably); ko is a
        # safety net in case the translation is off.
        anchor = f"the {subj}" if subj else "the subject"
        descr = attr_en or attr_ko or ""
        if attr_en and attr_ko and attr_en != attr_ko:
            descr = f'{attr_en} (Korean: "{attr_ko}")'
        elif attr_ko and not attr_en:
            descr = f'"{attr_ko}"'
        verb = "does NOT show" if negate else "clearly shows"
        return f"{anchor} {verb} {descr}"

    def _collect(item: Any) -> None:
        if isinstance(item, dict):
            subj = (item.get("subject") or "").strip()
            attr_en = (item.get("attribute_en") or "").strip()
            attr_ko = (item.get("attribute_ko") or "").strip()
            if not (attr_en or attr_ko):
                # qp_v0.5 single-field fallback.
                attr_ko = (item.get("attribute") or "").strip()
            if not (attr_en or attr_ko):
                return
            negate = bool(item.get("is_negative"))
        else:
            attr_ko = (str(item) or "").strip()
            if not attr_ko:
                return
            subj, attr_en, negate = "", "", False
        bucket = neg_clauses if negate else pos_clauses
        bucket.append(_clause(subj, attr_en, attr_ko, negate))

    if isinstance(residue, str):
        _collect(residue)
    elif isinstance(residue, list):
        for r in residue:
            _collect(r)

    clauses = pos_clauses + neg_clauses
    if not clauses:
        clauses = ["the listed attribute is clearly visible"]

    sys_text = (
        "You are a visual matcher. The crop's subject (person/vehicle/"
        "bag/etc.) has already been verified by an upstream filter — "
        "your job is to confirm the listed attribute(s) on that "
        "subject. Be strict about which subject each attribute belongs "
        "to: if an attribute would normally apply to a different "
        "subject than the one named, answer 'no'. Pay close attention "
        "to clauses that say 'does NOT show' — for those, answer 'no' "
        "if the attribute IS visible. Answer with one lowercase word: "
        "yes or no. No punctuation, no explanation."
    )
    user_text = "Does this image satisfy ALL of the following: " + \
        " AND ".join(clauses) + "? Answer yes or no."
    return [
        {"role": "system", "content": sys_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                # The actual image is appended by gemma_backend.generate(pil, …).
            ],
        },
    ]


def build_plr_retry_messages(
    object_hint: str, original_response: str, error_reason: str
) -> list[dict[str, Any]]:
    """Build retry messages when first PLR response fails parse/schema.

    Shows the model its own bad output + the failure reason and asks for
    a corrected response in whichever format the current run is using.
    """
    fmt = _plr_format()
    format_word = "YAML" if fmt == "yaml" else "JSON"
    retry_text = (
        f"Your previous response did not match the required schema:\n"
        f"  Error: {error_reason}\n\n"
        f"Your previous output was:\n{original_response[:500]}\n\n"
        f"Output ONLY a corrected {format_word} response that conforms to the "
        f"schema. No markdown fences, no leading prose."
    )
    base = build_plr_messages(object_hint)
    base.append({"role": "user", "content": [{"type": "text", "text": retry_text}]})
    return base


# =====================================================================
# Query parser prompt
# =====================================================================

# Query-parser prompt constants are loaded from the live prompt yaml above
# (query_parser block) — see the declarative-prompt-source note.


def query_parser_system_prompt() -> str:
    return _QUERY_PARSER_SYSTEM_PROMPT


def query_parser_user_prompt(user_query: str) -> str:
    # qp_v0.4: no enum injection. Python normalizer handles it.
    return _QUERY_PARSER_USER_TEMPLATE.format(
        user_query=user_query.replace('"', '\\"'),
    )


def build_query_parser_messages(user_query: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": query_parser_system_prompt()},
        {"role": "user", "content": query_parser_user_prompt(user_query)},
    ]


# =====================================================================
# JSON parsing — tolerant of common Gemma formatting quirks
# =====================================================================

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _extract_json_block(text: str) -> str:
    """Return the substring that looks most like a top-level JSON object."""
    s = text.strip()

    # Strip markdown fences if any.
    s = _MARKDOWN_FENCE_RE.sub("", s).strip()

    # If there's prose before the JSON, find the first '{' and matching '}'.
    if not s.startswith("{"):
        first = s.find("{")
        if first == -1:
            return s  # nothing to do
        s = s[first:]

    # Bracket-match to find the matching closing brace (handles nested braces
    # and strings).
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != -1:
        s = s[: end + 1]
    return s


def parse_plr_json(raw: str) -> dict[str, Any]:
    """Parse a (possibly slightly malformed) JSON string into a dict.

    Raises ValueError if parsing ultimately fails.
    """
    if not raw:
        raise ValueError("empty response")

    candidate = _extract_json_block(raw)

    # Try strict parse first
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # Try after removing trailing commas
        cleaned = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON: {e.msg} (around char {e.pos})"
            ) from e

    # Normalize Gemma's free-form deviations against the strict enum schema:
    #   - object_type: 'car'/'truck'/'pedestrian' → 'vehicle'/'person'
    #   - *_scores.selected: 'unknown' → '<group>_unknown' enum value
    #   - missing/empty *_topk arrays → single 'unknown' placeholder
    # Models occasionally output the synonym even with explicit enum lists in
    # the prompt; normalizing on parse keeps indexing flowing instead of
    # dumping the row into the DLQ on every minor deviation.
    return _normalize_plr_json(data)


# =====================================================================
# YAML parser (B-plan)
# =====================================================================

# Stripping any leading prose: YAML starts at the first line that is a
# bare "key: value" or a recognised top-level marker.
_YAML_HEAD_RE = re.compile(r"^(target|gender|age|outfit|upper|lower|equipment|action|margins|color|type)\s*:", re.M)
_YAML_FENCE_RE = re.compile(r"^```(?:yaml|yml)?\s*\n?|\n?```\s*$", re.I | re.M)
# Fallback flat-line parser for cases where the model loses YAML indentation
# but still emits "key: value" lines. dotted keys like "upper.color: black"
# are tolerated.
_FLAT_LINE_RE = re.compile(
    r"^\s*([a-z_.]+)\s*:\s*(.+?)\s*$", re.I
)


def _trim_to_yaml(raw: str) -> str:
    s = _YAML_FENCE_RE.sub("", raw).strip()
    m = _YAML_HEAD_RE.search(s)
    if m:
        s = s[m.start():]
    return s


def _scores_dict(
    enum: tuple[str, ...], selected: str, margin: float, default: str
) -> dict[str, Any]:
    """Build the *_scores shape that scoring.py expects, from a single
    selected label + a margin. We assign 1.0 to the selected enum value,
    0.0 to the others — scoring already weights by `decision_margin` to
    soften low-confidence rows, so the distribution itself is just a
    placeholder."""
    sel = (selected or "").strip().lower()
    if sel not in {e.lower() for e in enum}:
        sel = default
    dist = {e: (1.0 if e.lower() == sel else 0.0) for e in enum}
    dist["selected"] = sel
    dist["decision_margin"] = float(margin) if margin is not None else 0.5
    return dist


def _topk_one(label: str, fallback: str) -> list[dict[str, Any]]:
    s = (label or "").strip().lower()
    return [{"label": s or fallback, "score": 1.0}]


def _norm_sleeve(value: str) -> str:
    """Coerce the new upper.sleeve field to {long|short|unknown}."""
    s = (value or "").strip().lower()
    return s if s in {"long", "short"} else "unknown"


def _norm_military(value: str) -> str:
    """Coerce the plr_v1.4_cot `military` field to {military|civilian|unknown}.
    Defaults to 'unknown' when absent or out of enum (defensive)."""
    s = (value or "").strip().lower()
    return s if s in {m.lower() for m in MILITARY_ENUM} else "unknown"


def parse_plr_yaml(raw: str, hint: str = "person") -> dict[str, Any]:
    """Parse a YAML-formatted PLR response into the same dict shape that
    plr_schema.PLR_PERSON_SCHEMA / PLR_VEHICLE_SCHEMA + scoring.py and
    template_caption.py expect. Falls back to a flat key:value parser
    if PyYAML errors out, then to defaults — never raises for content,
    only for an empty string."""
    if not raw or not raw.strip():
        raise ValueError("empty response")

    import yaml
    body = _trim_to_yaml(raw)
    data: Any = None
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError:
        data = None

    flat: dict[str, str] = {}
    if not isinstance(data, dict):
        # Flat fallback — grab every "key: value" line.
        for line in body.splitlines():
            m = _FLAT_LINE_RE.match(line)
            if m:
                flat[m.group(1).strip().lower()] = m.group(2).strip()
        data = {}

    def g(key: str, default: str = "") -> str:
        """Read a scalar from either the YAML dict (nested under
        upper/lower/margins, OR flat top-level with dotted/underscored
        names) or the flat fallback table (when YAML parse missed it)."""
        if "." in key:
            top, sub = key.split(".", 1)
            if isinstance(data, dict):
                sub_d = data.get(top)
                if isinstance(sub_d, dict):
                    v = sub_d.get(sub)
                    if v is not None:
                        return str(v).strip()
                # yaml.safe_load happily keeps "upper.color: blue" as the
                # top-level key "upper.color" — check both spellings.
                for variant in (key, f"{top}_{sub}", f"{top}-{sub}"):
                    v = data.get(variant)
                    if v is not None:
                        return str(v).strip()
            v = flat.get(key.lower()) or flat.get(f"{top}_{sub}".lower())
            return (v or default).strip()
        v = data.get(key) if isinstance(data, dict) else None
        if v is None:
            v = flat.get(key.lower(), default)
        return (str(v) if v is not None else default).strip()

    def g_margin(key: str) -> float:
        m = data.get("margins") if isinstance(data, dict) else None
        if isinstance(m, dict):
            v = m.get(key)
            if v is not None:
                try:
                    return max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    pass
        v = flat.get(f"margins.{key}".lower()) or flat.get(f"margin_{key}".lower())
        if v is not None:
            try:
                return max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                pass
        return 0.5  # neutral confidence when the model omitted the margin

    target = g("target", hint).lower()
    if "vehicle" in target or target == "car":
        target = "vehicle"
    elif "person" in target or target in {"pedestrian", "human"}:
        target = "person"
    else:
        target = hint

    if target == "vehicle":
        color = g("color", "gray")
        vtype = g("type", "vehicle_unknown")
        return _normalize_plr_json({
            "object_type": "vehicle",
            "attributes": {
                "color_topk": _topk_one(color, "gray"),
                "type_topk": _topk_one(vtype, "vehicle_unknown"),
                # plr_v1.4_cot: prompt-native military judgment.
                "military": _norm_military(g("military")),
            },
        })

    # person path
    equip_raw = data.get("equipment") if isinstance(data, dict) else None
    if isinstance(equip_raw, str):
        # Flow list as a single string ("[backpack, umbrella]") or csv.
        equip_raw = equip_raw.strip().strip("[]")
        equip_list = [s.strip() for s in equip_raw.split(",") if s.strip()]
    elif isinstance(equip_raw, list):
        equip_list = [str(s).strip() for s in equip_raw if str(s).strip()]
    else:
        # Flat fallback
        v = flat.get("equipment", "")
        equip_list = [s.strip() for s in v.strip("[]").split(",") if s.strip()]

    equip_list = [e.lower() for e in equip_list if e.lower() not in {"none", "null", ""}]

    gender_reason = g("gender_reason")
    age_reason = g("age_reason")

    gender_dict = _scores_dict(
        GENDER_ENUM, g("gender"), g_margin("gender"), "female"
    )
    if gender_reason:
        gender_dict["reason"] = gender_reason

    age_dict = _scores_dict(
        AGE_GROUP_ENUM, g("age"), g_margin("age"), "adult"
    )
    if age_reason:
        age_dict["reason"] = age_reason

    # rider_vehicle is only meaningful when the action is riding_*; we
    # still emit the field so downstream filters can rely on its shape.
    rider_color = (g("rider_vehicle.color") or "").strip()
    rider_type = (g("rider_vehicle.type") or "").lower().strip()
    if rider_type not in {"motorcycle", "bicycle", "scooter", "kickboard"}:
        rider_type = "unknown"

    out = {
        "object_type": "person",
        "attributes": {
            "gender_scores": gender_dict,
            "age_group_scores": age_dict,
            "outfit_type_scores": _scores_dict(
                OUTFIT_TYPE_ENUM, g("outfit"), g_margin("outfit"), "obscured"
            ),
            "upper_clothing": {
                "color_topk": _topk_one(g("upper.color"), "gray"),
                "type_topk": _topk_one(g("upper.type"), "upper_unknown"),
                # NEW (redesign 2026-06): sleeve length — the one genuinely new
                # extracted hard axis (outer / location / mask are derivable
                # from type or already in equipment). Absent on pre-redesign
                # rows -> the gate wildcard-passes.
                "sleeve": _norm_sleeve(g("upper.sleeve")),
            },
            "lower_clothing": {
                "color_topk": _topk_one(g("lower.color"), "gray"),
                "type_topk": _topk_one(g("lower.type"), "lower_unknown"),
            },
            "equipment": [{"type": e, "score": 1.0} for e in equip_list],
            # plr_v1.4_cot: prompt-native military judgment.
            "military": _norm_military(g("military")),
            "static_action_state_scores": _scores_dict(
                STATIC_ACTION_ENUM, g("action"), 0.5, "posture_unknown"
            ),
            "rider_vehicle": {
                "color_topk": _topk_one(rider_color, "gray") if rider_color else [],
                "type": rider_type,
            },
        },
    }
    return _normalize_plr_json(out)


def parse_plr_response(
    raw: str, hint: str = "person", *, fmt: str | None = None
) -> dict[str, Any]:
    """Dispatch to the right parser based on IR_PLR_FORMAT (or override)."""
    import os
    chosen = (fmt or os.environ.get("IR_PLR_FORMAT", "yaml")).strip().lower()
    if chosen == "yaml":
        return parse_plr_yaml(raw, hint=hint)
    return parse_plr_json(raw)


# =====================================================================
# Normalization (post-parse, pre-validate)
# =====================================================================

# Synonyms that Gemma occasionally returns instead of the canonical enum.
_OBJECT_TYPE_SYNONYMS: dict[str, str] = {
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "van": "vehicle",
    "suv": "vehicle", "motorcycle": "vehicle", "motorbike": "vehicle",
    "bike": "vehicle", "bicycle": "vehicle", "scooter": "vehicle",
    "pedestrian": "person", "human": "person", "people": "person",
    "man": "person", "woman": "person", "child": "person",
}

# Selected-field 'unknown' → the matching *_unknown sentinel.
_UNKNOWN_FALLBACKS: dict[str, str] = {
    "static_action_state_scores": "posture_unknown",
    "outfit_type_scores": "obscured",
    "gender_scores": "female",   # default-unknown handled by decision_margin=0
    "age_group_scores": "adult", # same
}

# Placeholder rows when *_topk is missing or empty.
_TOPK_FALLBACKS: dict[str, list[dict[str, Any]]] = {
    "upper_clothing.color_topk": [{"label": "gray", "score": 0.0}],
    "upper_clothing.type_topk":  [{"label": "upper_unknown", "score": 1.0}],
    "lower_clothing.color_topk": [{"label": "gray", "score": 0.0}],
    "lower_clothing.type_topk":  [{"label": "lower_unknown", "score": 1.0}],
    # Vehicle:
    "attributes.color_topk":     [{"label": "gray", "score": 0.0}],
    "attributes.type_topk":      [{"label": "vehicle_unknown", "score": 1.0}],
}


def _coerce_topk_labels(
    arr: list[Any], enum: set[str], fallback: str
) -> list[dict[str, Any]]:
    """Coerce each topk entry's label to be inside enum, otherwise to fallback.

    Gemma routinely emits a topk array shaped correctly (label + score) but with
    `'unknown'` or a near-synonym that isn't in the strict enum. Without this
    step every such row drops into the DLQ even though the surrounding fields
    are usable.
    """
    out = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str) or label.strip().lower() not in enum:
            item = {**item, "label": fallback}
        out.append(item)
    return out or [{"label": fallback, "score": 1.0}]


def _normalize_plr_json(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce common Gemma deviations into the strict schema."""
    if not isinstance(data, dict):
        return data

    # Local import to avoid a circular dep at module load time.
    from plr_schema import (
        UPPER_TYPE_ENUM, LOWER_TYPE_ENUM, COLOR_ENUM,
        VEHICLE_TYPE_ENUM, EQUIPMENT_TYPE_ENUM,
    )

    # 1. object_type synonyms
    ot = str(data.get("object_type", "")).strip().lower()
    if ot in _OBJECT_TYPE_SYNONYMS:
        data["object_type"] = _OBJECT_TYPE_SYNONYMS[ot]
    elif ot in ("person", "vehicle"):
        data["object_type"] = ot

    attrs = data.get("attributes")
    if not isinstance(attrs, dict):
        return data

    # 2. *_scores.selected='unknown' → matching enum
    for field, fallback in _UNKNOWN_FALLBACKS.items():
        node = attrs.get(field)
        if isinstance(node, dict):
            sel = node.get("selected")
            if isinstance(sel, str) and sel.strip().lower() == "unknown":
                node["selected"] = fallback

    color_set = {c.lower() for c in COLOR_ENUM}
    upper_set = {c.lower() for c in UPPER_TYPE_ENUM}
    lower_set = {c.lower() for c in LOWER_TYPE_ENUM}
    vehicle_set = {c.lower() for c in VEHICLE_TYPE_ENUM}
    equipment_set = {c.lower() for c in EQUIPMENT_TYPE_ENUM}

    # 3. Person clothing — fill empty arrays AND coerce out-of-enum labels.
    person_cfg = (
        ("upper_clothing", color_set, upper_set, "upper_unknown", "gray"),
        ("lower_clothing", color_set, lower_set, "lower_unknown", "gray"),
    )
    for parent, cset, tset, type_fb, color_fb in person_cfg:
        node = attrs.get(parent)
        if not isinstance(node, dict):
            attrs[parent] = node = {}
        # color_topk
        arr = node.get("color_topk")
        if not isinstance(arr, list) or len(arr) == 0:
            node["color_topk"] = _TOPK_FALLBACKS[f"{parent}.color_topk"]
        else:
            node["color_topk"] = _coerce_topk_labels(arr, cset, color_fb)
        # type_topk
        arr = node.get("type_topk")
        if not isinstance(arr, list) or len(arr) == 0:
            node["type_topk"] = _TOPK_FALLBACKS[f"{parent}.type_topk"]
        else:
            node["type_topk"] = _coerce_topk_labels(arr, tset, type_fb)

    # 4. Vehicle attributes — same coercion against vehicle enums.
    if data.get("object_type") == "vehicle":
        arr = attrs.get("color_topk")
        if not isinstance(arr, list) or len(arr) == 0:
            attrs["color_topk"] = _TOPK_FALLBACKS["attributes.color_topk"]
        else:
            attrs["color_topk"] = _coerce_topk_labels(arr, color_set, "gray")
        arr = attrs.get("type_topk")
        if not isinstance(arr, list) or len(arr) == 0:
            attrs["type_topk"] = _TOPK_FALLBACKS["attributes.type_topk"]
        else:
            attrs["type_topk"] = _coerce_topk_labels(
                arr, vehicle_set, "vehicle_unknown"
            )

    # 4b. military judgment (plr_v1.4_cot): coerce to the 3-value enum,
    # defaulting to 'unknown' when absent/invalid. Present on both person and
    # vehicle attributes; harmless 'unknown' default on older prompt versions.
    mil = attrs.get("military")
    if mil is not None:
        attrs["military"] = _norm_military(mil if isinstance(mil, str) else "")

    # 5. equipment[]: type out-of-enum → 'other_equipment'; color_topk coerce.
    eq = attrs.get("equipment")
    if isinstance(eq, list):
        cleaned: list[dict[str, Any]] = []
        for item in eq:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if isinstance(t, str) and t.strip().lower() not in equipment_set:
                item["type"] = "other_equipment"
            ct = item.get("color_topk")
            if isinstance(ct, list) and len(ct) > 0:
                item["color_topk"] = _coerce_topk_labels(ct, color_set, "gray")
            cleaned.append(item)
        attrs["equipment"] = cleaned

    return data


def attach_prompt_version(data: dict[str, Any]) -> dict[str, Any]:
    """Inject the current prompt_version into the parsed PLR dict."""
    data.setdefault("prompt_version", PROMPT_VERSION)
    return data


# =====================================================================
# Backward-compat shim — pluggable-modules layer (A3)
#
# Importing providers.file_prompt_provider registers FilePromptProvider
# for both "plr_v0.4" and "plr_v1.3_cot" in the registry so callers can
# do:
#   from registry import get_provider
#   p = get_provider("prompt")
#   p.build_plr_messages("person")
#
# All existing module-level functions (build_plr_messages, etc.) remain
# intact — they are not delegated to the provider so that:
#   1. Circular imports are avoided (file_prompt_provider imports
#      plr_prompts indirectly via plr_schema).
#   2. parse_plr_yaml / parse_plr_json callers keep working unchanged.
#   3. tests/test_prompts.py (which imports parse_plr_yaml directly) passes
#      without modification.
#
# The registration is a best-effort import; if yaml or the prompts/ dir
# are missing in a stripped environment the module still loads cleanly.
# =====================================================================

try:
    import providers.file_prompt_provider as _fpp  # noqa: F401  side-effect: registers providers
except Exception:  # pragma: no cover
    pass
