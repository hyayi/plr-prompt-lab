"""plr_parse — PLR 응답 파싱 + 정규화 (plr_prompts에서 분리된 "아웃풋 절반").

모델이 뱉은 자유 텍스트(주 포맷 YAML; JSON은 검색 쿼리파서 응답용)를
plr_schema 형태의 구조화 dict로 되돌린다. 모든 슬롯이 schema 어휘로 강제
정규화된다 — 그래서 이 파일이 인풋/아웃풋 parity 표면의 아웃풋 쪽이다
(인풋 쪽은 plr_prompts).

입력/출력 예)
  parse_plr_response("target: person\ngender: female\nupper:\n  color: crimson…")
  → {"object_type":"person","attributes":{
       "gender_scores":{"selected":"female","decision_margin":0.85,…},
       "upper_clothing":{"color_topk":[{"label":"gray",…}], …},  ← crimson→gray 강제
       …}}
"""

from __future__ import annotations

import json
import re
from typing import Any

from plr_schema import (
    AGE_GROUP_ENUM,
    GENDER_ENUM,
    MILITARY_ENUM,
    OUTFIT_TYPE_ENUM,
    PROMPT_VERSION,
    STATIC_ACTION_ENUM,
)


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
    """(다소 깨진) JSON 문자열 → dict. 마크다운 펜스/앞머리 프로즈/트레일링
    콤마를 관용 처리. 현재 소비자는 core/ir 검색 쿼리파서의 응답(JSON).
    최종 실패 시 ValueError (→ 호출부 재시도/폴백 트리거).
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
    """선택 라벨 1개 + margin → scoring이 기대하는 *_scores dict.
    선택값에 1.0, 나머지에 0.0을 부여 — 분포 자체는 플레이스홀더고
    저신뢰 완화는 decision_margin이 담당한다.

    입력/출력 예) _scores_dict(("male","female"), "male", 0.9, "female")
      → {"male":1.0, "female":0.0, "selected":"male", "decision_margin":0.9}
    enum 밖 selected는 default로 강제 (이것이 "확장은 config로 불가"의 근거).
    """
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
    """upper.sleeve 값을 {long|short|unknown}으로 강제.
    예) "LONG "→"long", "rolled-up"→"unknown" (미등록→unknown 방어)."""
    s = (value or "").strip().lower()
    return s if s in {"long", "short"} else "unknown"


def _norm_military(value: str) -> str:
    """military 값을 MILITARY_ENUM으로 강제. 없거나 enum 밖이면 'unknown'
    (방어적 — v1.5 프롬프트는 unknown을 요구하지 않지만 파서는 관용)."""
    s = (value or "").strip().lower()
    return s if s in {m.lower() for m in MILITARY_ENUM} else "unknown"


def parse_plr_yaml(raw: str, hint: str = "person") -> dict[str, Any]:
    """YAML 형식 PLR 응답 → PERSON/VEHICLE_SCHEMA 형태의 dict.

    관용 계층 3단: ① 정상 YAML 파싱 → ② PyYAML 실패 시 "key: value" 평면
    라인 파서(모델이 들여쓰기를 잃어도 회수) → ③ 필드 누락 시 default.
    내용 때문에 raise하지 않는다 — 빈 문자열만 예외.
    내부 g()/g_margin()이 중첩·평면·점표기("upper.color") 표기를 모두 수용.
    """
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
    """파서 진입점 — run_plr이 크롭마다 호출. 기본 YAML 파서로 보내고,
    fmt="json" 명시 시에만 JSON 파서 (레거시 응답 판독용)."""
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
    """topk 배열의 각 label을 enum 안으로 강제 — enum 밖이면 fallback으로 교체.

    Gemma가 shape은 맞는데 유사어를 넣는 일이 흔해서, 이 단계가 없으면
    주변 필드가 멀쩡한 행이 통째로 DLQ에 빠진다.

    입력/출력 예) [{"label":"crimson","score":0.9}], color_set, "gray"
      → [{"label":"gray","score":0.9}]   ← "config enum 확장 불가"의 실측 근거
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
    """파싱 직후·검증 직전의 최종 정규화 — Gemma의 흔한 일탈을 스키마 안으로.
    ① object_type 유사어(car→vehicle, pedestrian→person)
    ② *_scores.selected의 'unknown' → 슬롯별 폴백
    ③ 비거나 enum 밖인 *_topk → 플레이스홀더/강제 교체
    ④ military 3값 강제 ⑤ equipment type 밖 값 → other_equipment"""
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


