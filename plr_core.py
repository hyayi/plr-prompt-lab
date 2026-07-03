"""순수 single-view PLR 추론 코어 — 운영 인덱서와 lab이 공유하는 조립 지점.

`run_plr(pil, qreport, model, object_type_hint) -> plr_json` 이 스토리지-프리
파이프라인의 전부다:

  전처리(마커) → 프롬프트 조합(build_plr_messages) → model.generate 1회
  → 파싱·정규화(parse_plr_response) → 후처리(_attach_military_flags)

의존성은 plr_prompts / quality_gate / PIL / Model 프로토콜뿐 —
gemma_backend 직접 참조 0 (생성은 주입된 `model`을 통해서만). 그래서
`import plr_core`가 storage/psycopg2/redis를 절대 끌고 오지 않는다.

비공개 토글 두 개(동작 보존용): `_pre_marked`(마커를 호출부가 이미 그렸음/
그리지 않기로 함 — 운영 indexing이 객체당 1회만 그리려고, lab 실험 config의
marker:false도 이 경로 사용) / `_attach`(military 플래그 부착을 호출부로 미룸).
공개/lab 계약은 기본값(마커 그림 + 플래그 부착)이다.

입력/출력 예) run_plr(크롭PIL, SimpleNamespace(mode="normal_plr"), model, "person")
  → {"object_type":"person","attributes":{…}}  (plr_parse 모듈 docstring 예 참고)
"""

from __future__ import annotations

from typing import Any, Callable

from PIL import Image


# =====================================================================
# Target marker — canonical home is preprocess.py; re-exported here so
# indexing.py / older callers keep importing from plr_core.
# =====================================================================


# Moved to preprocess.py (image input surface — own module so pre-processing
# is a named, versionable component). The old name stays importable.
from preprocess import draw_target_marker as _draw_target_marker  # noqa: E402


# =====================================================================
# Military / color scalar enrichment (pure) — canonical home.
# =====================================================================

_MILITARY_COLOR = "military_olive"


def _top_color(color_topk: Any) -> str | None:
    """color_topk 배열에서 최고점 라벨 1개. 없거나 깨졌으면 None (무예외).

    입력/출력 예) [{"label":"black","score":0.9},{"label":"gray","score":0.1}]
      → "black"
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
    """plr_json에 스칼라 색/군인 플래그 필드를 제자리(in-place) 기록.

    기록 필드(스키마상 전부 optional):
      vehicle → attributes.primary_color(str) · is_military(bool)
      person  → upper/lower_clothing.primary_color(str) · is_soldier(bool)

    판정: 프롬프트-네이티브 military 판단 OR 국방색(military_olive) 규칙
    (후자는 구버전 행 recall 폴백). 무예외 방어적 — 소스가 깨져 있으면
    조용히 건너뜀. 파싱 후·저장 전에 호출되어 JSONB 포함-쿼리를 가능케 함.

    입력/출력 예) person + upper 최고색 black + military=="military"
      → attrs["is_soldier"]=True, upper_clothing["primary_color"]="black"
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
    """공유 최심부: 모델 1회 호출 → 원문 파싱. 검증은 하지 않는다 —
    스키마 검증·재시도·DLQ는 호출부(indexing) 책임으로 남겨 의미 불변.
    raw는 여기서 소비되고 버려진다 (lab은 _RawCapture 래퍼로 가로챔)."""
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
    """single-view PLR 추론 1회 (크롭 → plr_json).

    인자:
      pil: 객체 크롭(PIL). `_pre_marked=False`면 여기서 마커를 그린다.
      qreport: mode=="coarse_only"면 모델을 건너뛰고 최소 JSON 반환
        (v1.5 single-view 계약에서 게이팅은 제거 — 호출부가 normal_plr 고정).
      model: `generate(messages, image) -> str` 프로토콜 (진짜/Mock 교체점).
      object_type_hint: "person" | "vehicle" — 프롬프트 기능 파일 선택.
      build_messages: (lab 전용 divergence) 버전별 빌더 주입구. None(기본,
        core/ir의 유일 동작)이면 plr_prompts.build_plr_messages — 운영과
        바이트 동일. lab은 FilePromptProvider(version_override=…) 빌더를
        넣어 한 체크아웃에서 버전 비교를 가능하게 한다.
      _pre_marked/_attach: 비공개 토글 (모듈 docstring 참고).

    반환: PLR JSON dict.
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
