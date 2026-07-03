"""Parser-emit vs schema-declaration parity.

JSON Schema는 기본이 "미명시 키 허용"이라, 파서가 새 필드를 emit해도 검증이
터지지 않고 그 필드만 조용히 검증 사각지대가 된다 (실사례: v1.5의 sleeve와
gender reason이 스키마 선언 없이 저장되고 있었다). 이 테스트는 "파서가
emit하는 모든 키는 PERSON/VEHICLE_SCHEMA에 명시돼 있어야 한다"를 강제해서,
프롬프트 구조 개선 시 스키마 갱신 누락을 빨간불로 바꾼다.

No GPU, no DB, no Redis.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plr_parse import parse_plr_response  # noqa: E402
from plr_schema import PERSON_SCHEMA, VEHICLE_SCHEMA, validate_plr  # noqa: E402

# 모델이 낼 수 있는 최대치 응답 — 모든 답 슬롯을 채워서 파서의 emit 표면 전체를
# 드러낸다 (슬롯이 프롬프트에 추가되면 여기에도 추가해야 사각지대가 안 생긴다).
_MAX_PERSON_YAML = """\
target: person
gender: male
gender_reason: broad shoulders
age: adult
outfit: two_piece
upper:
  color: black
  type: jacket
  sleeve: long
lower:
  color: blue
  type: jeans
equipment:
  - type: backpack
    color: red
action: standing
military: civilian
riding: motorcycle
riding_color: red
margins: {gender: 0.9, age: 1.0, outfit: 0.8, upper: 0.7, lower: 0.7, sleeve: 0.6}
"""

_MAX_VEHICLE_YAML = """\
target: vehicle
vehicle_type: sedan
color: white
state: parked
military: civilian
margins: {vehicle_type: 0.9, color: 0.8}
"""

# _normalize_plr_json이 object_type 구분 없이 person 의복 블록을 채우는 알려진
# 동작 — vehicle 행에도 upper/lower_clothing 플레이스홀더가 붙는다. 스키마에
# 선언할 가치는 없는 잔재라 whitelist로 두고, 재인덱싱-후 정리 백로그에서
# person 한정으로 고칠 때 이 whitelist를 비워 회귀를 잡는다.
_VEHICLE_NORMALIZER_JUNK = {"upper_clothing", "lower_clothing"}


def _undeclared(emitted: dict[str, Any], declared: dict[str, Any]) -> list[str]:
    """emitted dict의 키 중 스키마 properties에 없는 경로 목록 (재귀)."""
    missing: list[str] = []
    for key, val in emitted.items():
        spec = declared.get(key)
        if spec is None:
            missing.append(key)
            continue
        if isinstance(val, dict) and isinstance(spec.get("properties"), dict):
            missing += [f"{key}.{m}" for m in _undeclared(val, spec["properties"])]
        elif isinstance(val, list) and isinstance(spec.get("items"), dict):
            item_props = spec["items"].get("properties")
            if isinstance(item_props, dict):
                for item in val:
                    if isinstance(item, dict):
                        missing += [f"{key}[].{m}" for m in _undeclared(item, item_props)]
    return sorted(set(missing))


def test_person_parser_emits_only_declared_keys() -> None:
    parsed = parse_plr_response(_MAX_PERSON_YAML, hint="person")
    validate_plr(parsed)  # 최대치 emit도 스키마 검증을 통과해야 한다
    declared = PERSON_SCHEMA["properties"]["attributes"]["properties"]
    missing = _undeclared(parsed["attributes"], declared)
    assert not missing, (
        f"파서가 emit하지만 PERSON_SCHEMA에 미명시(검증 사각지대): {missing} — "
        f"plr_schema.py의 스키마에 선언을 추가하라"
    )


def test_vehicle_parser_emits_only_declared_keys() -> None:
    parsed = parse_plr_response(_MAX_VEHICLE_YAML, hint="vehicle")
    validate_plr(parsed)
    declared = VEHICLE_SCHEMA["properties"]["attributes"]["properties"]
    emitted = {
        k: v for k, v in parsed["attributes"].items()
        if k not in _VEHICLE_NORMALIZER_JUNK
    }
    missing = _undeclared(emitted, declared)
    assert not missing, (
        f"파서가 emit하지만 VEHICLE_SCHEMA에 미명시(검증 사각지대): {missing} — "
        f"plr_schema.py의 스키마에 선언을 추가하라"
    )


def test_sleeve_and_reason_are_declared() -> None:
    """이 테스트가 태어난 이유였던 두 필드의 고정 회귀 가드."""
    person_attrs = PERSON_SCHEMA["properties"]["attributes"]["properties"]
    assert "reason" in person_attrs["gender_scores"]["properties"]
    sleeve = person_attrs["upper_clothing"]["properties"].get("sleeve")
    assert sleeve is not None
    assert set(sleeve["enum"]) >= {"long", "short", "unknown"}
