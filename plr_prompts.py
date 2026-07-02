"""plr_prompts — PLR prompt ASSEMBLY only (composition layer).

Prompt TEXT lives in yaml, one file per function, one directory per version:

    prompts/<PROMPT_VERSION_YAML_COT>/
        person.yaml         # system + user_cot + user_plain
        vehicle.yaml        # system + user
        query_parser.yaml   # search query parser (core/ir search only)
        vqa.yaml            # search VQA re-rank system prompt
        retry.yaml          # schema-failure retry template

This module only LOADS those files and COMPOSES messages (enum injection via
plr_schema/vocab, CoT toggle, dynamic VQA clauses). Response parsing lives in
plr_parse.py (re-exported here for backward compatibility). The legacy JSON
prompt path (v0.4) was removed 2026-07 — YAML is the only wire format.
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

import yaml as _yaml

from plr_schema import (
    COLOR_ENUM,
    EQUIPMENT_TYPE_ENUM,
    LOWER_TYPE_ENUM,
    MILITARY_ENUM,
    STATIC_ACTION_ENUM,
    UPPER_TYPE_ENUM,
    VEHICLE_TYPE_ENUM,
)

# Backward-compat re-exports — the parsers moved to plr_parse.py.
from plr_parse import (  # noqa: F401
    attach_prompt_version,
    parse_plr_json,
    parse_plr_response,
    parse_plr_yaml,
    _normalize_plr_json,
    _norm_military,
    _norm_sleeve,
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
# Declarative prompt source — one directory per version, one file per
# function. plr_prompts holds ZERO prompt text; it loads and composes.
# ---------------------------------------------------------------------------

_PROMPTS_ROOT = _Path(__file__).resolve().parent / "prompts"


def _load_function(version: str, name: str) -> dict[str, Any]:
    path = _PROMPTS_ROOT / version / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        data = _yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


_P_PERSON = _load_function(PROMPT_VERSION_YAML_COT, "person")
_P_VEHICLE = _load_function(PROMPT_VERSION_YAML_COT, "vehicle")
_P_QP = _load_function(PROMPT_VERSION_YAML_COT, "query_parser")
_P_VQA = _load_function(PROMPT_VERSION_YAML_COT, "vqa")
_P_RETRY = _load_function(PROMPT_VERSION_YAML_COT, "retry")


def _commit_enum(values) -> tuple[str, ...]:
    """Enum values as offered to the model — the `*unknown*` escape hatches are
    excluded (plr_v1.5_cot forced-commit contract). The full enums in
    plr_schema KEEP their unknown members: they are still needed to read
    pre-v1.5 indexed rows and as defensive normalisation targets."""
    return tuple(v for v in values if "unknown" not in v)


def _plr_with_reason() -> bool:
    """CoT toggle: gender_reason / age_reason lines before the labels.
    Via IR_PLR_REASON=on (off by default — extra tokens cost ~35% latency)."""
    import os
    v = os.environ.get("IR_PLR_REASON", "off").strip().lower()
    return v in {"on", "true", "1", "yes"}


def plr_yaml_user_prompt_person(with_reason: bool = False) -> str:
    template = _P_PERSON["user_cot"] if with_reason else _P_PERSON["user_plain"]
    return template.rstrip("\n").format(
        colors=", ".join(_commit_enum(COLOR_ENUM)),
        upper_types=", ".join(_commit_enum(UPPER_TYPE_ENUM)),
        lower_types=", ".join(_commit_enum(LOWER_TYPE_ENUM)),
        equips=", ".join(_commit_enum(EQUIPMENT_TYPE_ENUM)),
        actions=", ".join(_commit_enum(STATIC_ACTION_ENUM)),
        military_enum="|".join(_commit_enum(MILITARY_ENUM)),
    )


def plr_yaml_user_prompt_vehicle() -> str:
    return _P_VEHICLE["user"].rstrip("\n").format(
        colors=", ".join(_commit_enum(COLOR_ENUM)),
        vehicle_types=", ".join(_commit_enum(VEHICLE_TYPE_ENUM)),
        military_enum="|".join(_commit_enum(MILITARY_ENUM)),
    )


def build_plr_messages(object_hint: str = "person") -> list[dict[str, Any]]:
    """Build chat messages for one PLR call (YAML wire format).

    object_hint: 'person' or 'vehicle' — selects the function file.
    The image is appended by gemma_backend.generate(pil, messages).
    """
    if object_hint == "vehicle":
        sys_text = _P_VEHICLE["system"].rstrip("\n")
        user_text = plr_yaml_user_prompt_vehicle()
    else:
        sys_text = _P_PERSON["system"].rstrip("\n")
        user_text = plr_yaml_user_prompt_person(with_reason=_plr_with_reason())
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
    """Build chat messages for a single yes/no VQA call (search re-rank —
    core/ir only). The system prompt comes from vqa.yaml; the user question
    is composed dynamically from the residue items.

    Residue item fields: subject / attribute_ko / attribute_en / is_negative
    (qp_v0.5 {subject, attribute} and legacy bare strings are upgraded).
    Output contract: max_tokens=4, temperature=0.0; anything but a clear
    "no" is treated as "yes" (recall-preferring).
    """
    pos_clauses: list[str] = []
    neg_clauses: list[str] = []

    def _clause(subj: str, attr_en: str, attr_ko: str, negate: bool) -> str:
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

    user_text = "Does this image satisfy ALL of the following: " + \
        " AND ".join(clauses) + "? Answer yes or no."
    return [
        {"role": "system", "content": _P_VQA["system"].rstrip("\n")},
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
    """Retry messages when the first PLR response fails parse/schema —
    template from retry.yaml, appended to the base messages."""
    retry_text = _P_RETRY["user"].rstrip("\n").format(
        error_reason=error_reason,
        original_response=original_response[:500],
    )
    base = build_plr_messages(object_hint)
    base.append({"role": "user", "content": [{"type": "text", "text": retry_text}]})
    return base


# =====================================================================
# Query parser prompt (core/ir search only)
# =====================================================================


def query_parser_system_prompt() -> str:
    return _P_QP["system"].rstrip("\n")


def query_parser_user_prompt(user_query: str) -> str:
    # qp_v0.4: no enum injection. Python normalizer handles it.
    return _P_QP["user"].rstrip("\n").format(
        user_query=user_query.replace('"', '\\"'),
    )


def build_query_parser_messages(user_query: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": query_parser_system_prompt()},
        {"role": "user", "content": query_parser_user_prompt(user_query)},
    ]


# =====================================================================
# Provider self-registration (pluggable-modules layer)
# =====================================================================

try:
    import providers.file_prompt_provider as _fpp  # noqa: F401  side-effect: registers providers
except Exception:  # pragma: no cover
    pass
