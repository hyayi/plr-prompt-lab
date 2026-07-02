"""User text query → structured search slots.

Two-stage parser:
  1. Korean / English synonym dictionary (cheap, deterministic) — handles the
     common cases instantly without calling Gemma.
  2. Gemma fallback for queries the dictionary doesn't fully cover.

The output of either stage matches the same QueryJSON shape so the search
pipeline doesn't care which path produced it.

The dictionary is intentionally small at launch — it can grow via Skill Loop 3
(docs §16) as we observe real user queries.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from plr_schema import COLOR_GROUP, lower_shape_of

from plr_prompts import (
    build_query_parser_messages,
    parse_plr_json,
)

log = logging.getLogger(__name__)


# =====================================================================
# Korean / English synonym dictionary
# =====================================================================

# Color: surface form (lowercase) → English enum. Includes Korean and English.
_COLOR_SYNONYMS: dict[str, str] = {
    # black
    "검정": "black", "검은": "black", "검은색": "black", "까만": "black",
    "흑색": "black", "black": "black",
    # dark gray
    "진회색": "dark_gray", "짙은회색": "dark_gray", "dark gray": "dark_gray",
    "darkgray": "dark_gray",
    # gray
    "회색": "gray", "gray": "gray", "grey": "gray",
    # light gray
    "연회색": "light_gray", "옅은회색": "light_gray", "light gray": "light_gray",
    # white
    "하얀": "white", "흰": "white", "흰색": "white", "하얀색": "white", "white": "white",
    # silver
    "은색": "silver", "silver": "silver",
    # red
    "빨강": "red", "빨간": "red", "빨간색": "red", "적색": "red", "red": "red",
    # orange
    "주황": "orange", "주황색": "orange", "오렌지": "orange", "orange": "orange",
    # yellow
    "노랑": "yellow", "노란": "yellow", "노란색": "yellow", "황색": "yellow",
    "yellow": "yellow",
    # brown
    "갈색": "brown", "brown": "brown",
    # dark brown
    "진갈색": "dark_brown", "짙은갈색": "dark_brown", "dark brown": "dark_brown",
    # beige
    "베이지": "beige", "beige": "beige",
    # cream
    "크림": "cream", "크림색": "cream", "cream": "cream",
    # gold
    "금색": "gold", "황금색": "gold", "gold": "gold",
    # blue
    "파랑": "blue", "파란": "blue", "파란색": "blue", "푸른": "blue", "blue": "blue",
    # navy
    "남색": "navy", "남청색": "navy", "navy": "navy",
    # light blue
    "하늘색": "light_blue", "하늘": "light_blue", "스카이블루": "light_blue",
    "light blue": "light_blue",
    # green
    "초록": "green", "초록색": "green", "녹색": "green", "green": "green",
    # dark green
    "진초록": "dark_green", "짙은초록": "dark_green", "dark green": "dark_green",
    # purple
    "보라": "purple", "보라색": "purple", "자주색": "purple", "purple": "purple",
    # pink
    "분홍": "pink", "핑크": "pink", "pink": "pink",
}

# Color groups (coarse) → list of canonical labels (for "어두운 옷" style queries)
_COLOR_GROUP_KEYWORDS: dict[str, str] = {
    "어두운": "dark", "어두은": "dark", "검은계열": "dark", "dark": "dark",
    "밝은": "light", "밝은색": "light", "light": "light",
    "원색": "vivid", "선명한": "vivid", "vivid": "vivid",
}

# Person upper types
_UPPER_TYPE_SYNONYMS: dict[str, str] = {
    "재킷": "jacket", "자켓": "jacket", "jacket": "jacket",
    "코트": "coat", "coat": "coat",
    "패딩": "padding", "다운": "padding", "padding": "padding",
    "바람막이": "windbreaker", "windbreaker": "windbreaker",
    "조끼": "vest", "베스트": "vest", "vest": "vest",
    "카디건": "cardigan", "cardigan": "cardigan",
    "티셔츠": "tshirt", "티": "tshirt", "tshirt": "tshirt", "t-shirt": "tshirt",
    # NOTE: "긴팔"(long-sleeve) alone is a SLEEVE attribute (-> _SLEEVE_SYNONYMS,
    # upper_sleeve), NOT a garment type. Mapping it to long_sleeve_tee here made
    # the upper_type group gate wrongly exclude long-sleeve OUTERWEAR (coat/
    # padding) from "긴팔" searches (review HIGH). Keep only the garment form.
    "긴팔티": "long_sleeve_tee",
    "셔츠": "shirt", "shirt": "shirt",
    "블라우스": "blouse", "blouse": "blouse",
    "후드티": "hoodie", "후드": "hoodie", "hoodie": "hoodie",
    "스웨터": "sweater", "sweater": "sweater",
    "니트": "knit", "knit": "knit",
    "유니폼": "uniform", "uniform": "uniform",
    "안전조끼": "safety_vest", "safety vest": "safety_vest",
    "작업복": "workwear", "workwear": "workwear",
    "원피스": "dress", "드레스": "dress", "dress": "dress",
    "점프수트": "jumpsuit", "jumpsuit": "jumpsuit",
}

# Person lower types
# Upper sleeve length (redesign 2026-06). Maps to the upper_sleeve hard axis.
# Sleeveless (민소매/나시) is intentionally omitted — it's neither long nor
# short in the {long|short|unknown} enum.
_SLEEVE_SYNONYMS: dict[str, str] = {
    "반팔": "short", "반소매": "short", "short sleeve": "short", "short-sleeve": "short",
    "긴팔": "long", "긴소매": "long", "long sleeve": "long", "long-sleeve": "long",
}

_LOWER_TYPE_SYNONYMS: dict[str, str] = {
    "바지": "pants", "pants": "pants",
    # `진` (one syllable) caused false positives in words like "쓰러진",
    # "다친", "지친". Keep the multi-syllable forms only.
    "청바지": "jeans", "데님": "jeans", "스키니진": "jeans", "jeans": "jeans",
    "트레이닝": "training_pants", "운동복바지": "training_pants",
    "training pants": "training_pants",
    "슬랙스": "slacks", "slacks": "slacks",
    "레깅스": "leggings", "leggings": "leggings",
    "반바지": "shorts", "shorts": "shorts",
    "치마": "skirt", "스커트": "skirt", "skirt": "skirt",
    "긴치마": "long_skirt", "long skirt": "long_skirt",
}

# Equipment
_EQUIPMENT_SYNONYMS: dict[str, str] = {
    # Bags
    "백팩": "backpack", "배낭": "backpack", "backpack": "backpack",
    "숄더백": "shoulder_bag", "shoulder bag": "shoulder_bag",
    "핸드백": "handbag", "handbag": "handbag",
    "크로스백": "cross_bag", "메신저백": "cross_bag", "cross bag": "cross_bag",
    # Weather / outdoor
    "카트": "cart", "수레": "cart", "cart": "cart",
    "우산": "umbrella", "umbrella": "umbrella",
    # Head — v0.9 collapsed beanie into hat in the schema because
    # Gemma E4B can't reliably tell soft headwear types apart in CCTV
    # crops. All non-helmet headwear surface forms map to "hat".
    "헬멧": "helmet", "helmet": "helmet",
    "안전모": "construction_helmet", "construction helmet": "construction_helmet",
    "모자": "hat", "캡": "hat", "hat": "hat", "cap": "hat",
    "비니": "hat", "beanie": "hat",
    "베레모": "hat", "베레": "hat", "beret": "hat",
    "중절모": "hat", "버킷햇": "hat", "버킷": "hat", "bucket hat": "hat",
    # Face / accessories
    "마스크": "mask", "mask": "mask",
    "안경": "glasses", "glasses": "glasses",
    "선글라스": "sunglasses", "sunglasses": "sunglasses",
    "헤드폰": "headphones", "headphones": "headphones",
    # Hand-held
    "휴대전화": "phone_in_hand", "휴대폰": "phone_in_hand",
    "스마트폰": "phone_in_hand", "phone": "phone_in_hand",
    "핸드폰": "phone_in_hand",
    "들고": "handheld_object", "손에": "handheld_object",
    "쥐고": "handheld_object", "들고있": "handheld_object",
    "병": "bottle", "물병": "bottle", "음료": "bottle", "bottle": "bottle",
    "서류": "clipboard", "클립보드": "clipboard", "clipboard": "clipboard",
    # Weapons / dangerous (security)
    "칼": "knife", "흉기": "knife", "나이프": "knife", "knife": "knife",
    "총": "firearm", "권총": "firearm", "총기": "firearm",
    "firearm": "firearm", "gun": "firearm", "pistol": "firearm",
    "몽둥이": "bat_stick", "방망이": "bat_stick", "야구방망이": "bat_stick",
    "막대": "bat_stick", "스틱": "bat_stick", "bat": "bat_stick",
    "stick": "bat_stick", "club": "bat_stick",
    "날카로운": "sharp_object", "뾰족한": "sharp_object",
    "sharp": "sharp_object", "blade": "sharp_object",
    "무기": "sharp_object",  # generic "weapon" routes to sharp_object as
                              # the broadest weapon enum we have; cascade
                              # group_overlap then OR-matches knife/firearm/bat.
    # Mobility
    "지팡이": "cane", "cane": "cane",
    "유모차": "stroller", "stroller": "stroller",
}

# Korean group keywords — let users say just "가방" or "무기" and have the
# search hit every member of the group.
_EQUIPMENT_GROUP_KEYWORDS: dict[str, str] = {
    "가방": "bag",        # any of backpack/shoulder_bag/handbag/cross_bag
    "흉기들고": "weapon", "무기들고": "weapon",
}

# Static actions (single-image only)
_ACTION_SYNONYMS: dict[str, str] = {
    "서있는": "standing", "서있": "standing", "standing": "standing",
    "걷는": "walking_like", "걸어가는": "walking_like", "walking": "walking_like",
    "뛰는": "running_like", "달리는": "running_like", "running": "running_like",
    "앉은": "sitting", "앉아있는": "sitting", "sitting": "sitting",
    "구부린": "bending", "bending": "bending",
    "쪼그린": "squatting", "쭈그린": "squatting", "squatting": "squatting",
    # Two-wheel riding — surface forms include both the vehicle word
    # together with "탄/타고" and the bare vehicle name (handled below in
    # the routing logic so a plain "오토바이" still targets the bike).
    "오토바이 탄": "riding_motorcycle", "오토바이 타고": "riding_motorcycle",
    "오토바이를 탄": "riding_motorcycle", "오토바이를 타고": "riding_motorcycle",
    "바이크 탄": "riding_motorcycle", "motorcycle rider": "riding_motorcycle",
    "자전거 탄": "riding_bicycle", "자전거 타고": "riding_bicycle",
    "자전거를 탄": "riding_bicycle", "자전거를 타고": "riding_bicycle",
    "bicycle rider": "riding_bicycle", "cyclist": "riding_bicycle",
    "스쿠터 탄": "riding_scooter", "스쿠터 타고": "riding_scooter",
    "스쿠터를 탄": "riding_scooter", "scooter rider": "riding_scooter",
    "킥보드 탄": "riding_kickboard", "킥보드 타고": "riding_kickboard",
    "킥보드를 탄": "riding_kickboard", "전동킥보드": "riding_kickboard",
    "kickboard rider": "riding_kickboard",
}

# Triggers for temporal event search (handled by a separate temporal module)
_TEMPORAL_KEYWORDS: tuple[str, ...] = (
    "쓰러진", "쓰러짐", "넘어진", "낙상",
    "fall", "fallen", "collapsed",
    "싸우", "폭행", "fighting", "assault",
    "배회", "맴도는", "loitering",
    "유기", "버린",
    "방화", "불", "연기", "fire", "smoke",
    "이상행동", "abnormal",
)

# Gender / age
_GENDER_SYNONYMS: dict[str, str] = {
    "남자": "male", "남성": "male", "남": "male", "남자아이": "male",
    "man": "male", "men": "male", "male": "male", "boy": "male",
    "여자": "female", "여성": "female", "여": "female", "여자아이": "female",
    "woman": "female", "women": "female", "female": "female", "girl": "female",
}
_AGE_SYNONYMS: dict[str, str] = {
    "어른": "adult", "성인": "adult", "adult": "adult",
    "아이": "child", "어린이": "child", "아동": "child", "child": "child",
    "kid": "child",
}

# Outfit type cues
_OUTFIT_TYPE_SYNONYMS: dict[str, str] = {
    "원피스": "one_piece",
    "드레스": "one_piece",
    "점프수트": "one_piece",
    "one piece": "one_piece",
    "dress": "one_piece",
    "two piece": "two_piece",
}

# Vehicle types
_VEHICLE_TYPE_SYNONYMS: dict[str, str] = {
    "세단": "sedan", "sedan": "sedan",
    "에스유브이": "suv", "suv": "suv", "SUV": "suv",
    "해치백": "hatchback", "hatchback": "hatchback",
    "밴": "van", "van": "van",
    "미니밴": "minivan", "minivan": "minivan",
    "픽업": "pickup_truck", "pickup": "pickup_truck",
    "트럭": "truck", "truck": "truck",
    "버스": "bus", "bus": "bus",
    "택시": "taxi", "taxi": "taxi",
    "구급차": "ambulance", "ambulance": "ambulance",
    "경찰차": "police_car", "police car": "police_car",
    "소방차": "fire_truck", "fire truck": "fire_truck",
    "오토바이": "motorcycle", "motorcycle": "motorcycle",
    "스쿠터": "scooter", "scooter": "scooter",
    "자전거": "bicycle", "bicycle": "bicycle",
    "킥보드": "kickboard", "전동킥보드": "kickboard",
    "kickboard": "kickboard", "e-scooter": "kickboard",
    # Korean 경차 (light car) — Morning / Casper / Ray / Spark sized.
    # v0.9 introduces this as a dedicated bucket so users can search
    # "경차" without it dissolving into the generic hatchback group.
    "경차": "light_car", "light car": "light_car", "light_car": "light_car",
    # NOTE: v0.8 had ~30 Korean/foreign model names (sonata/grandeur/
    # mercedes/bmw/...) but the v0.8_cot indexing run showed Gemma E4B
    # identified only 2/214 cars as "bmw" and 0 for everything else.
    # The model labels added prompt tokens without improving recall, so
    # v0.9 drops them. Users typing "소나타"/"벤츠" will fall through
    # to category match (sedan / vehicle_unknown) via embedding score.
}


# =====================================================================
# QueryJSON
# =====================================================================


@dataclass
class QueryJSON:
    """Structured form of a user query. All lists are English enums."""

    query_type: str = "person_search"
    target: str = "person"
    required: dict[str, list[str]] = field(default_factory=dict)
    optional: dict[str, list[str]] = field(default_factory=dict)
    excluded: dict[str, list[str]] = field(default_factory=dict)
    ambiguous: list[str] = field(default_factory=list)
    # Context-bound residue (qp_v0.5+): each item is
    #   {"subject": "<head noun>", "attribute": "<unknown word>"}
    # The Gemma parser emits this shape so a clause like
    # "쓰러져 있는 오토바이" stays bound to motorcycle and never
    # drifts onto a person crop in the VQA stage. Older dictionary-only
    # paths still produce bare strings; the search side coerces them to
    # the bound shape on the fly.
    free_form_residue: list[dict[str, str]] = field(default_factory=list)
    # Refine signals (redesign 2026-06): fine sub-type distinctions the HARD
    # gate intentionally does NOT enforce (jeans vs slacks, dress-shirt vs tee).
    # Each item: {"slot": "lower_type"|"upper_type"|..., "value": "<enum>",
    # "subject": "person"}. Resolved at the 2nd (refine) search stage via the
    # PLR fine label first, VQA only for low-confidence candidates.
    refine: list[dict[str, str]] = field(default_factory=list)
    needs_temporal_search: bool = False
    template_caption_en: str = ""
    template_caption_ko: str = ""
    parser_version: str = "qp_v0.4"
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "target": self.target,
            "required": dict(self.required),
            "optional": dict(self.optional),
            "excluded": dict(self.excluded),
            "ambiguous": list(self.ambiguous),
            "free_form_residue": list(self.free_form_residue),
            "refine": list(self.refine),
            "needs_temporal_search": self.needs_temporal_search,
            "template_caption_en": self.template_caption_en,
            "template_caption_ko": self.template_caption_ko,
            "parser_version": self.parser_version,
        }


# =====================================================================
# Dictionary parser
# =====================================================================


# Vehicle context phrases trigger vehicle_search even without explicit type.
# `차` (one syllable) was matching inside person words like `청자켓` and routing
# the whole query to vehicle_search. We keep only multi-character phrases that
# are unambiguously about vehicles, plus the compound `<color>차` pattern that
# Korean users commonly write ("흰차", "검정차", "파란차").
_VEHICLE_CONTEXT_RE = re.compile(
    r"(차량|자동차|승용차|화물차|버스|트럭|오토바이|자전거|밴|택시|"
    # `<color>차` with optional `색` suffix and optional whitespace:
    # matches 흰차 / 흰 차 / 흰색 차 / 하얀색차 / silver 차 …
    r"(?:흰|하얀|하양|검은|검정|회색|진회색|연회색|빨간|빨강|파란|파랑|노란|"
    r"노랑|초록|녹색|보라|주황|갈색|남색|연두|연한|짙은|어두운|밝은|silver|"
    r"검|흑)(?:색|색깔)?\s*차|"
    r"vehicle|car|truck|bus|van|motorcycle|bicycle|scooter|taxi)",
    re.IGNORECASE,
)

# Person markers — when present, vehicle hints by themselves shouldn't take
# over the whole query. "검은 옷 사람" + "흰차" should produce a mixed search,
# not a vehicle-only one.
_PERSON_CONTEXT_RE = re.compile(
    r"(사람|남자|남성|여자|여성|행인|보행자|소년|소녀|어른|어린이|아이|아동|"
    r"person|man|woman|male|female|pedestrian|boy|girl|adult|child)",
    re.IGNORECASE,
)


def _scan_dict(text_lower: str, table: dict[str, str]) -> list[str]:
    """Return English enums found in text via the synonym table.

    Iteration is longest-first AND consuming: after a surface form
    matches, we blank out its characters in our working copy of the
    text so shorter sub-strings cannot also match. This is the fix for
    "검은색 자동차" picking up both "검은색"→black AND "은색"→silver
    (the trailing "은색" of "검은색" used to match a separate
    silver synonym).
    """
    hits: list[str] = []
    remaining = text_lower
    for surface in sorted(table.keys(), key=len, reverse=True):
        canonical = table[surface]
        if surface and surface in remaining and canonical not in hits:
            hits.append(canonical)
            remaining = remaining.replace(surface, " ")
    return hits


# Korean particles and verb endings that survive the dict-scan removal
# pass but carry no search meaning. Treated as "consumed" so the Gemma
# re-rank stage doesn't try to match on them. Conservative set —
# anything semantically meaningful (color, type, gender) is already in
# one of the synonym dicts.
_PARTICLES_AND_FILLERS = frozenset({
    # particles
    "을", "를", "이", "가", "은", "는", "도", "만", "와", "과", "의", "로",
    "에", "에서", "에게", "께", "으로", "에서는", "에서도", "에게서",
    # very common verb endings that wrap clothing/equipment surface forms
    "입은", "입고", "쓴", "쓰고", "쓰는", "멘", "메고", "맨", "들고", "들은",
    "탄", "타고", "걸친", "걸치고", "차림", "차림의", "차림으로",
    # additional verb endings that appear in natural sentences but carry
    # no slot meaning. Without these "신고", "있는", "신은" etc would
    # land in the residue and falsely activate Gemma re-rank.
    "신은", "신고", "신었", "있는", "있다", "있고", "찼던",
    # generic context words already absorbed by *_CONTEXT_RE / synonym
    # dicts; including them here protects the residue from picking them
    # up when the regex paths miss the exact compound.
    "사람", "사람이", "사람을", "사람의",
    # generic head nouns for clothing — the colour/type slots already
    # capture the actual garment. "옷" by itself adds nothing the PLR
    # filter isn't already doing, and leaving it (with its 을/이/은
    # variants) in the residue spuriously triggers a Gemma VQA pass.
    "옷", "옷을", "옷이", "옷은", "옷도", "옷의",
})

# Suffix particles to strip from residue tokens before deciding whether
# the residue is empty. "옷을" → "옷" (then matched against the filler
# set), "신발이" → "신발" (still residue, but recognisable downstream).
_TRAILING_PARTICLE_SUFFIXES: tuple[str, ...] = (
    "으로", "에서", "에게", "에서는", "에서도", "들이", "들의", "들을", "들에",
    "을", "를", "이", "가", "은", "는", "도", "만", "와", "과", "의", "로", "에",
)


# ---------------------------------------------------------------------
# Negation post-processing for the dictionary path (Wave 3, FR-13.3)
# ---------------------------------------------------------------------
# The Gemma + normalizer path detects negation via _detect_negation in
# query_normalizer.py. The dictionary path scans Korean substrings, so
# "모자 안 쓴 사람" merrily places hat in `required.equipment` unless
# we scan for negation cues afterwards. Keep the pattern catalogue in
# sync with query_normalizer._NEG_VERB_RE / _NEG_SUFFIX_RE.
_DICT_NEG_VERBS = ("쓴", "쓰는", "쓰고", "쓰지", "입은", "입는", "입고", "신은",
                   "신는", "신고", "멘", "메는", "메고", "들고", "들은", "든",
                   "탄", "타고", "걸친", "걸치고")
# 단형 부정: "안 V" — "안 쓴", "안 입은"
_DICT_NEG_VERB_RE = re.compile(
    r"안\s*(?:" + "|".join(re.escape(v) for v in _DICT_NEG_VERBS) + r")"
)
# 장형 부정: "V-지 않-" — "입지 않은", "쓰지 않아요", "메지 않고", "걸치지 않은"
# Stems are the connective-form verb roots ("입지" not "입은"). 표기 변형
# ("입지 않-" 표준, "입지않-" 띄어쓰기 누락) 두 가지 모두 수용.
_DICT_LONG_NEG_STEMS = ("입지", "쓰지", "신지", "메지", "들지", "차지", "타지",
                       "걸치지", "하지")
_DICT_LONG_NEG_RE = re.compile(
    r"(?:" + "|".join(re.escape(v) for v in _DICT_LONG_NEG_STEMS) + r")"
    r"\s*않(?:은|는|아|아요|았|으면|고|지)?"
)
_DICT_NEG_SUFFIX_RE = re.compile(r"(없는|없이|없음|미착용|안한|아닌)")


def _dict_has_negation(text: str) -> bool:
    """Combine 단형 + 장형 + 명사형 negation patterns."""
    return bool(
        _DICT_NEG_VERB_RE.search(text)
        or _DICT_LONG_NEG_RE.search(text)
        or _DICT_NEG_SUFFIX_RE.search(text)
    )


def _apply_dict_negation(q: QueryJSON, user_text: str) -> None:
    """Move equipment / vehicle_type / clothing-type items into
    ``q.excluded`` when the original query negates them.

    Operates in-place. Scans the raw query for each Korean enum surface
    that landed in `required`, checking an 8-character window for a
    negation cue. Matching items are removed from `required` and added
    to `excluded` so the hard filter can reject candidates that DO
    have the negated attribute.

    Free-form residue items are tagged with is_negative=true when their
    attribute_ko sits next to a negation cue in the query.
    """
    if not _dict_has_negation(user_text):
        return

    def _neg_in_window(s: str) -> bool:
        return _dict_has_negation(s)

    # Per-table walk. Equipment includes the group keywords ("가방"/
    # "모자") so a negated group rejects every member. Colour slots
    # appear once for upper, once for lower, and once for vehicle —
    # we don't know which side the user meant ("검은 옷 안 입은" is
    # upper-leaning, "검은 차 안 탄" is vehicle-leaning) so we mirror
    # whatever side already has the colour in `required`. If neither
    # does, default to upper_color for "옷"-implied queries and
    # vehicle_color for vehicle queries — picked from q.target.
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
            if not _neg_in_window(window):
                continue
            mapped = noun_table[surface]
            if slot_name == "equipment_group":
                # Expand the group to its full member list and exclude
                # all of them under the canonical "equipment" slot.
                try:
                    from plr_schema import EQUIPMENT_TYPE_GROUP
                    members = EQUIPMENT_TYPE_GROUP.get(mapped, ())
                except Exception:
                    members = ()
                q.excluded.setdefault("equipment", [])
                for m in members:
                    if m not in q.excluded["equipment"]:
                        q.excluded["equipment"].append(m)
                    if "equipment" in q.required:
                        q.required["equipment"] = [
                            v for v in q.required["equipment"] if v != m
                        ]
                        if not q.required["equipment"]:
                            del q.required["equipment"]
                continue
            if slot_name == "color":
                enum_val = mapped
                # First decide which colour slot the negation belongs
                # to: the dictionary parser defaults every loose colour
                # to upper_color, so just mirroring `required` would
                # mis-route "빨간 차" into upper_color. Right-side
                # context wins: if a vehicle noun sits within 6 chars,
                # route to vehicle_color regardless of what `required`
                # currently holds.
                look_idx = user_text.find(surface)
                nxt = user_text[look_idx + len(surface):look_idx + len(surface) + 6] if look_idx >= 0 else ""
                is_vehicle_neighbour = any(
                    vw in nxt for vw in ("차량", "자동차", "차 ", "차를", "차에",
                                        "오토바이", "트럭", "버스", "세단",
                                        "SUV", "suv", "밴", "택시")
                ) or nxt.strip().startswith("차")
                if is_vehicle_neighbour:
                    target_slot = "vehicle_color"
                else:
                    # No vehicle hint — honour the slot currently
                    # holding the colour, else upper_color for person
                    # / vehicle_color for vehicle.
                    target_slot = None
                    for color_slot in ("upper_color", "lower_color", "vehicle_color"):
                        if color_slot in q.required and enum_val in q.required[color_slot]:
                            target_slot = color_slot
                            break
                    if target_slot is None:
                        target_slot = (
                            "vehicle_color" if q.target == "vehicle"
                            else "upper_color"
                        )
                # Remove the colour from every existing required slot
                # (the dict parser may have placed it in the wrong one)
                # and add to target_slot's excluded list.
                for color_slot in ("upper_color", "lower_color", "vehicle_color"):
                    if color_slot in q.required and enum_val in q.required[color_slot]:
                        q.required[color_slot] = [
                            v for v in q.required[color_slot] if v != enum_val
                        ]
                        if not q.required[color_slot]:
                            del q.required[color_slot]
                q.excluded.setdefault(target_slot, [])
                if enum_val not in q.excluded[target_slot]:
                    q.excluded[target_slot].append(enum_val)
                continue
            enum_val = mapped
            # Move out of required.
            if slot_name in q.required and enum_val in q.required[slot_name]:
                q.required[slot_name] = [
                    v for v in q.required[slot_name] if v != enum_val
                ]
                if not q.required[slot_name]:
                    del q.required[slot_name]
            # Add to excluded.
            q.excluded.setdefault(slot_name, [])
            if enum_val not in q.excluded[slot_name]:
                q.excluded[slot_name].append(enum_val)

    # Free-form residue: tag with is_negative=true when the residue's
    # attribute_ko sits next to a negation cue.
    for item in q.free_form_residue:
        if not isinstance(item, dict):
            continue
        attr_ko = (item.get("attribute_ko") or item.get("attribute") or "").strip()
        if not attr_ko or item.get("is_negative"):
            continue
        idx = user_text.find(attr_ko)
        if idx < 0:
            continue
        window = user_text[idx:idx + len(attr_ko) + 8]
        if _neg_in_window(window):
            item["is_negative"] = True


def _bind_residue_to_target(
    residue_tokens: list[str], target: str, required: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    """Wrap dictionary-path residue tokens as {subject, attribute} pairs.

    The qp_v0.5+ shape used by the Gemma normalizer path is a list of
    objects so VQA stays anchored to the right noun. The legacy
    dictionary path emits bare strings — coerce them here using
    ``target`` (and, for vehicles, the parsed type) as the subject.
    """
    if not residue_tokens:
        return []
    subject = target
    req = required or {}
    if target == "vehicle":
        vt = req.get("vehicle_type") or []
        subject = vt[0] if vt else "vehicle"
    elif target == "person":
        subject = "person"
    elif target == "mixed":
        # mixed = person+vehicle. Prefer the side that has a typed slot
        # so the VQA question lands on a concrete noun. "쓰러져 있는
        # 오토바이" → bind residue to motorcycle, not the generic
        # "mixed" placeholder.
        vt = req.get("vehicle_type") or []
        if vt:
            subject = vt[0]
        elif req.get("vehicle_color"):
            subject = "vehicle"
        else:
            subject = "person"
    else:
        subject = target or "object"
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for tok in residue_tokens:
        attr = (str(tok) or "").strip()
        if not attr:
            continue
        key = (subject, attr.lower())
        if key in seen:
            continue
        seen.add(key)
        # qp_v0.6 shape — attribute_en stays empty for the dict path,
        # the VQA composer falls back to attribute_ko then. Negation is
        # also left to the Gemma path; this dict route is a recall
        # fallback for short queries and doesn't try to parse negation.
        out.append({
            "subject": subject,
            "attribute_ko": attr,
            "attribute_en": "",
            "is_negative": False,
        })
    return out


def _extract_free_form_residue(text_lower: str) -> list[str]:
    """Tokens in ``text_lower`` that no synonym dict / context regex consumed.

    Used by the Gemma VQA re-rank stage (RFC §9). An empty list means the
    query was fully covered by the structured PLR slots and the search
    side can skip the Gemma re-rank altogether.

    Algorithm (deliberately simple, conservative):
      1. Strip every surface form known to the synonym dicts AND the
         context regexes from the lowercased text.
      2. Tokenise what's left on whitespace; drop one-char tokens,
         numbers, and known particles / verb endings.
      3. Return the remaining tokens. Order is preserved.

    Example: "호피무늬 가방 멘 남자" — 가방 / 멘 / 남자 are all known
    surfaces; "호피무늬" survives → ['호피무늬'].
    """
    cleaned = text_lower

    # Collect every surface form the dict-based parser actually checks.
    known: set[str] = set()
    for d in (
        _COLOR_SYNONYMS, _COLOR_GROUP_KEYWORDS,
        _UPPER_TYPE_SYNONYMS, _LOWER_TYPE_SYNONYMS,
        _EQUIPMENT_SYNONYMS, _EQUIPMENT_GROUP_KEYWORDS,
        _ACTION_SYNONYMS, _GENDER_SYNONYMS, _AGE_SYNONYMS,
        _OUTFIT_TYPE_SYNONYMS, _VEHICLE_TYPE_SYNONYMS,
    ):
        known.update(d.keys())

    # Context-regex literals — extract the alternation contents so they
    # too get stripped (the regex flips between car / man / etc., but
    # the residue extractor just wants the surface words).
    for pat in (_VEHICLE_CONTEXT_RE.pattern, _PERSON_CONTEXT_RE.pattern):
        for tok in re.findall(r"[가-힣a-zA-Z]+", pat):
            known.add(tok.lower())
    # Negation cue words — "없는", "없이", "안한", "아닌" etc. These are
    # consumed by the negation post-processor (_apply_dict_negation),
    # not by the synonym dicts, so without explicit stripping they leak
    # into residue and trigger a spurious VQA call.
    for cue in ("없는", "없이", "없음", "미착용", "안한", "아닌"):
        known.add(cue)
    for neg_verb in _DICT_NEG_VERBS:
        known.add(neg_verb)
        known.add("안 " + neg_verb)
    # 장형 부정 stem + 종결: "입지", "입지않", "않은", "않는" 등이 residue
    # 로 새지 않도록 known 에 모두 등록.
    for stem in _DICT_LONG_NEG_STEMS:
        known.add(stem)
        known.add(stem + "않")
    for tail in ("않은", "않는", "않아", "않아요", "않았", "않으면", "않고", "않지", "않"):
        known.add(tail)

    # Strip longest surfaces first so "오토바이" beats "바이" and "사람이"
    # is consumed before "사람".
    for surface in sorted(known, key=len, reverse=True):
        if surface and surface in cleaned:
            cleaned = cleaned.replace(surface, " ")

    residue: list[str] = []
    for tok in cleaned.split():
        tok = tok.strip()
        if len(tok) < 2:
            continue
        if tok in _PARTICLES_AND_FILLERS:
            continue
        if tok.isdigit():
            continue
        # Strip a single trailing Korean particle and re-check the
        # known/filler sets. This catches "옷을"/"옷이"/"사람의" /
        # "신발이" cases that the dict-scan above can't consume
        # because the synonym tables hold the bare noun.
        stripped = tok
        for suf in _TRAILING_PARTICLE_SUFFIXES:
            if len(stripped) > len(suf) + 1 and stripped.endswith(suf):
                stripped = stripped[: -len(suf)]
                break
        if stripped != tok:
            if stripped in _PARTICLES_AND_FILLERS:
                continue
            if stripped in known:
                continue
            tok = stripped
        residue.append(tok)
    return residue


def parse_with_dictionary(user_text: str) -> QueryJSON:
    """Quick deterministic parse. Doesn't call Gemma.

    For queries the dictionary can fully cover this is the entire parser; for
    others it returns a partial result that the Gemma parser can augment.
    """
    text = user_text.strip()
    t = text.lower()

    q = QueryJSON(raw_text=user_text)
    q.parser_version = "qp_v0.2-dict"

    # Temporal event detection
    if any(kw in t for kw in _TEMPORAL_KEYWORDS):
        q.needs_temporal_search = True
        q.query_type = "event_search"
        q.target = "event"
        # Still extract attributes for "쓰러진 빨간 옷 사람" style queries
        # so the event module can JOIN with the PLR index.

    # Vehicle vs person routing. Three signals:
    #   - vehicle context phrase or vehicle type word
    #   - person context phrase (gender, age, role) or any person clothing word
    has_vehicle_hint = bool(_VEHICLE_CONTEXT_RE.search(t)) or any(
        v in t for v in _VEHICLE_TYPE_SYNONYMS
    )
    # "차" 단독 — _VEHICLE_CONTEXT_RE intentionally excludes the bare
    # syllable to avoid matching it inside 청자켓 / 신차 / 찻집. But a
    # search query that is *only* "차" or "차 …" is unambiguously
    # asking about vehicles. Trigger the vehicle hint on the exact
    # surface forms we know the user means.
    _BARE_CAR_PATTERNS = (
        t == "차",
        t.startswith("차 "), t.startswith("차의 "), t.startswith("차를 "),
        t.endswith(" 차"), t.endswith(" 차를"), t.endswith(" 차도"),
    )
    if any(_BARE_CAR_PATTERNS):
        has_vehicle_hint = True
    has_person_hint = bool(_PERSON_CONTEXT_RE.search(t)) or any(
        u in t for u in _UPPER_TYPE_SYNONYMS
    ) or any(
        l in t for l in _LOWER_TYPE_SYNONYMS
    )
    if not q.needs_temporal_search:
        if has_vehicle_hint and has_person_hint:
            # "검은 옷 사람이 흰차 옆에" — multi-target. Score both object types
            # using their respective slot tables; co-occurrence ranking is a
            # follow-up (needs frame-level joining via the tube DB).
            q.query_type = "mixed_search"
            q.target = "mixed"
        elif has_vehicle_hint:
            q.query_type = "vehicle_search"
            q.target = "vehicle"
        # else: default person_search (already set in __init__)
    is_vehicle = q.target == "vehicle"

    # Color: scan once; we'll split into upper/lower based on neighboring words.
    colors_in_query = _scan_dict(t, _COLOR_SYNONYMS)
    color_groups = _scan_dict(t, _COLOR_GROUP_KEYWORDS)

    # Upper / lower type detection
    upper_types = _scan_dict(t, _UPPER_TYPE_SYNONYMS)
    lower_types = _scan_dict(t, _LOWER_TYPE_SYNONYMS)
    equipment = _scan_dict(t, _EQUIPMENT_SYNONYMS)
    # Pre-scan vehicle_types so the routing logic below can consult them;
    # the canonical scan a few lines down stays as-is for clarity.
    vehicle_types = _scan_dict(t, _VEHICLE_TYPE_SYNONYMS)
    # NOTE: actions are scanned again a few lines down; we pre-scan here
    # to decide how to handle the vehicle_type signal. Three branches:
    #
    # 1) "오토바이 탄 사람" — riding_* action surfaced. The user wants
    #    the rider crop, not the bike crop. Force target=person and
    #    drop the vehicle hint.
    #
    # 2) "오토바이" (bare two-wheel type, no rider verb, no person
    #    word) — analysis often labels the whole rider+bike crop as
    #    object_type=person with action=riding_*, while a separate
    #    vehicle row also exists. Routing this to vehicle alone misses
    #    most of the matches. Switch to target=mixed and inject the
    #    matching riding_* action so both sides get a fair shot.
    #
    # 3) Anything else with a vehicle hint stays as-is (target=vehicle
    #    for "흰차" or "트럭", target=mixed when a person word is also
    #    present like "흰차 옆 사람").
    _RIDING_ACTIONS = {"riding_motorcycle", "riding_bicycle",
                       "riding_scooter", "riding_kickboard"}
    _TWO_WHEEL_TO_RIDING = {
        "motorcycle": "riding_motorcycle",
        "bicycle":    "riding_bicycle",
        "scooter":    "riding_scooter",
        "kickboard":  "riding_kickboard",
    }
    _pre_actions = _scan_dict(t, _ACTION_SYNONYMS)
    _bare_two_wheel_riding: list[str] = []
    if any(a in _RIDING_ACTIONS for a in _pre_actions):
        # Case 1
        has_vehicle_hint = False
        if q.target in {"vehicle", "mixed"}:
            q.query_type = "person_search"
            q.target = "person"
    elif (q.target == "vehicle" and not has_person_hint
            and any(v in _TWO_WHEEL_TO_RIDING for v in vehicle_types)):
        # Case 2 — "오토바이" / "자전거" / "스쿠터" 단독
        _bare_two_wheel_riding = [
            _TWO_WHEEL_TO_RIDING[v] for v in vehicle_types
            if v in _TWO_WHEEL_TO_RIDING
        ]
        q.query_type = "mixed_search"
        q.target = "mixed"
    # Expand group keywords ("가방" → all bag-type enum values) so the user
    # can write a category and still hit each member during search.
    try:
        from plr_schema import EQUIPMENT_TYPE_GROUP
        for surface, grp_name in _EQUIPMENT_GROUP_KEYWORDS.items():
            if surface in t:
                for member in EQUIPMENT_TYPE_GROUP.get(grp_name, ()):
                    if member not in equipment:
                        equipment.append(member)
    except Exception:
        pass  # schema unavailable in unit-test contexts
    actions = _scan_dict(t, _ACTION_SYNONYMS)
    genders = _scan_dict(t, _GENDER_SYNONYMS)
    ages = _scan_dict(t, _AGE_SYNONYMS)
    outfits = _scan_dict(t, _OUTFIT_TYPE_SYNONYMS)
    sleeves = _scan_dict(t, _SLEEVE_SYNONYMS)
    vehicle_types = _scan_dict(t, _VEHICLE_TYPE_SYNONYMS)

    req = q.required

    if q.target == "vehicle":
        if colors_in_query:
            req["vehicle_color"] = colors_in_query
        if vehicle_types:
            req["vehicle_type"] = vehicle_types
        q.template_caption_en = _vehicle_template_en(colors_in_query, vehicle_types)
        q.template_caption_ko = text  # original Korean is good enough
        q.free_form_residue = _bind_residue_to_target(
            _extract_free_form_residue(t), q.target, q.required,
        )
        _apply_dict_negation(q, text)
        return q

    if q.target == "mixed":
        # Vehicle-side hints — only the unambiguous ones. We deliberately do
        # not push every color from the sentence into vehicle_color, because
        # the same color word often qualifies the person side ("검정 옷 사람이
        # 흰차"). Use vehicle_types here; vehicle_color stays empty unless the
        # user wrote a clear "<color> <vehicle>" pair, which the current
        # dictionary can't disambiguate alone. The search side will still rank
        # vehicles by their colors against the embedding.
        if vehicle_types:
            req["vehicle_type"] = vehicle_types
        # Bare two-wheel type ("오토바이" alone) wants both the vehicle row
        # AND any person caught riding one. Inject the matching riding_*
        # action so the person side of the mixed search hits those rows
        # via the action hard filter.
        if _bare_two_wheel_riding:
            req["action"] = _bare_two_wheel_riding
        # Fall through to the person slot extraction below so this dict also
        # carries upper_color/upper_type/gender/age/etc.

    # Person path (also runs for mixed target — extracts person slots).
    # Split colors into upper / lower by proximity to the surface form of the
    # type keyword in the original (lowercase) text.
    upper_colors, lower_colors, any_colors = _split_colors_upper_lower(
        t, colors_in_query, upper_types, lower_types
    )

    if upper_colors:
        req["upper_color"] = upper_colors
    if lower_colors:
        req["lower_color"] = lower_colors
    if any_colors:
        # garment-agnostic colour ("빨간 옷") → OR over upper/lower at the gate.
        req["any_color"] = any_colors
    if upper_types:
        req["upper_type"] = upper_types
    if sleeves:
        req["upper_sleeve"] = sleeves
    if lower_types:
        # Keep the fine label (caption / strict mode); the coarse SHAPE gate and
        # the fabric refine signal are derived by the shared post-processor
        # (_derive_groups_and_refine) near the end of this function.
        req["lower_type"] = lower_types
    if equipment:
        req["equipment"] = equipment
    if actions:
        req["action"] = actions
    if genders:
        req["gender"] = genders
    # Heuristic: a query that explicitly asks for skirt / long_skirt /
    # dress without naming a gender is almost always a female search.
    # We bias gender=female only when the user wrote nothing about it;
    # any explicit gender keyword wins (e.g. an off-distribution
    # "치마 입은 남자" still works because the keyword above already
    # populated req["gender"]).
    _FEMALE_BIASED_LOWER = {"skirt", "long_skirt"}
    _FEMALE_BIASED_OUTFIT = {"one_piece"}
    if "gender" not in req:
        bias_female = (
            any(lt in _FEMALE_BIASED_LOWER for lt in (lower_types or []))
            or any(o in _FEMALE_BIASED_OUTFIT for o in (outfits or []))
        )
        if bias_female:
            req["gender"] = ["female"]

    # Rider-vehicle attributes — only meaningful when the action is one
    # of the riding_* values. Lets a query like "빨간 오토바이 탄 남자"
    # filter on both the rider (gender=male, action=riding_motorcycle)
    # AND on the bike itself (rider_vehicle_type=motorcycle, color=red).
    # We deliberately don't try to disambiguate which of multiple colors
    # belongs to the bike vs. the clothing here — the cascade falls
    # through cleanly because each color slot is independently checked
    # against its own PLR field. So "빨간 오토바이 탄 검은 옷 남자"
    # still works: red lands on rider_vehicle_color, black on upper_color
    # via _split_colors_upper_lower above.
    if any(a in _RIDING_ACTIONS for a in actions):
        riding_two_wheel_types = [
            v for v in vehicle_types if v in _TWO_WHEEL_TO_RIDING
        ]
        if riding_two_wheel_types:
            req["rider_vehicle_type"] = riding_two_wheel_types
        if colors_in_query:
            req["rider_vehicle_color"] = colors_in_query[:]
            # "빨간 오토바이 탄 남자" — the 빨간 belongs to the bike, not
            # the rider's clothes. Strip the rider-vehicle colors out of
            # upper_color / lower_color so the user doesn't have to write
            # the same colour twice to mean two different things. If
            # they DID write a distinct clothing colour ("검은 옷 입고
            # 빨간 오토바이 탄") the upper/lower colour slot keeps its
            # other entries because we only drop the overlap.
            _bike_colors = set(colors_in_query)
            if req.get("upper_color"):
                req["upper_color"] = [
                    c for c in req["upper_color"] if c not in _bike_colors
                ] or None
                if req["upper_color"] is None:
                    del req["upper_color"]
            if req.get("lower_color"):
                req["lower_color"] = [
                    c for c in req["lower_color"] if c not in _bike_colors
                ] or None
                if req["lower_color"] is None:
                    del req["lower_color"]

    # Shared post-processors (also applied to the gemma path for parity):
    #  1) derive the coarse lower_shape gate + fabric refine from lower_type;
    #  2) surface-form domain intents (e.g. 청바지 = blue denim).
    _derive_groups_and_refine(q)
    _apply_lexical_overrides(q, text)
    if ages:
        req["age_group"] = ages
    if outfits:
        req["outfit_type"] = outfits

    # If color groups appear without specific colors, store as optional hint.
    if color_groups and not (upper_colors or lower_colors):
        q.optional["color_group"] = color_groups

    q.template_caption_en = _person_template_en(req)
    q.template_caption_ko = text
    q.free_form_residue = _bind_residue_to_target(
        _extract_free_form_residue(t), q.target, q.required,
    )
    _apply_dict_negation(q, text)
    return q


def _apply_lexical_overrides(q: QueryJSON, user_text: str) -> None:
    """Surface-form domain intents applied to BOTH parse paths (dictionary AND
    gemma) so they stay in parity. Mutates q.required in place.

    Currently handles `청바지` = "blue jeans" intent: most Korean speakers use
    the word as a near-synonym of blue denim. Unless the user named a different
    lower colour ("검정 청바지"), expand lower_color to the blue family and make
    sure the long_pants shape gate is present. Previously this lived only in the
    dictionary path, so the production gemma path never applied it (the bug that
    let "청바지" surface red trousers).
    """
    t = user_text.lower()
    if "청바지" in t:
        blue = list(COLOR_GROUP["blue"])  # ("blue", "navy", "light_blue")
        existing_lower = q.required.get("lower_color") or []
        explicit_non_blue = set(existing_lower) - set(blue)
        if not explicit_non_blue:
            q.required["lower_color"] = list(dict.fromkeys(existing_lower + blue))
        shape = q.required.get("lower_shape") or []
        if "long_pants" not in shape:
            q.required["lower_shape"] = list(dict.fromkeys(shape + ["long_pants"]))


def _derive_groups_and_refine(q: QueryJSON) -> None:
    """Shared post-processor (dictionary AND gemma paths):
    - LOWER: from fine `lower_type`, populate coarse `lower_shape` (HARD gate)
      and push the fabric distinction (jeans vs slacks) to q.refine.
    - UPPER: the fine `upper_type` is GROUP-gated in the hard filter (outerwear
      vs top, scoring.passes_hard_filter); push the fine distinction (coat vs
      jacket) to q.refine too.
    Fine labels stay in `required` for caption/strict; the hard gate reads
    lower_shape + upper_type-group. Idempotent (shape set only if absent, refine
    deduped by (slot,value)).
    """
    seen = {(r.get("slot"), r.get("value")) for r in q.refine}

    lts = q.required.get("lower_type") or []
    if lts:
        shapes = [s for s in dict.fromkeys(lower_shape_of(lt) for lt in lts)
                  if s != "lower_unknown"]
        if shapes and "lower_shape" not in q.required:
            q.required["lower_shape"] = shapes
        for lt in lts:
            if ("lower_type", lt) not in seen:
                q.refine.append({"slot": "lower_type", "value": lt, "subject": "person"})
                seen.add(("lower_type", lt))

    for ut in (q.required.get("upper_type") or []):
        if ("upper_type", ut) not in seen:
            q.refine.append({"slot": "upper_type", "value": ut, "subject": "person"})
            seen.add(("upper_type", ut))


def _split_colors_upper_lower(
    text_lower: str,
    colors: list[str],
    upper_types: list[str],
    lower_types: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Disambiguate which colors modify upper vs lower clothing.

    Heuristic: for each color, look at the nearest type keyword in the original
    text. Upper-type proximity → upper_color; lower-type proximity → lower_color.
    Colors with NO nearby garment keyword go to `any_colors` (garment-agnostic
    "red anywhere", e.g. "빨간 옷") — they are routed to required.any_color and
    matched as an OR over upper/lower at the gate (redesign 2026-06). Previously
    such colors silently defaulted to upper, which dropped red-bottom-only
    matches.

    Returns (upper_colors, lower_colors, any_colors).
    """
    if not colors:
        return [], [], []
    # Build position list of upper/lower surface forms in the text
    upper_positions = _positions_of(text_lower, _UPPER_TYPE_SYNONYMS, upper_types)
    lower_positions = _positions_of(text_lower, _LOWER_TYPE_SYNONYMS, lower_types)

    # Inverse map color enum → list of surface positions
    upper_colors: list[str] = []
    lower_colors: list[str] = []
    any_colors: list[str] = []
    for color in colors:
        # Find all surface forms of this color in the text and pick the
        # earliest one for distance measurement.
        color_positions = _positions_of_color(text_lower, color)
        if not color_positions:
            continue
        pos = color_positions[0]
        nearest_upper = _nearest(pos, upper_positions)
        nearest_lower = _nearest(pos, lower_positions)

        if nearest_upper is None and nearest_lower is None:
            any_colors.append(color)
        elif nearest_lower is None:
            upper_colors.append(color)
        elif nearest_upper is None:
            lower_colors.append(color)
        elif nearest_lower < nearest_upper:
            lower_colors.append(color)
        else:
            upper_colors.append(color)
    return upper_colors, lower_colors, any_colors


def _positions_of(
    text: str, table: dict[str, str], canonicals: list[str]
) -> list[int]:
    """Return character positions in text where any surface form of these
    canonical labels appears."""
    positions: list[int] = []
    canonical_set = set(canonicals)
    for surface, canonical in table.items():
        if canonical not in canonical_set:
            continue
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(surface)
    return sorted(positions)


def _positions_of_color(text: str, canonical: str) -> list[int]:
    """Same as _positions_of but for a single canonical color label."""
    positions: list[int] = []
    for surface, c in _COLOR_SYNONYMS.items():
        if c != canonical:
            continue
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(surface)
    return sorted(positions)


def _nearest(pos: int, candidates: list[int]) -> int | None:
    """Return min distance from pos to any candidate. None if no candidates."""
    if not candidates:
        return None
    return min(abs(c - pos) for c in candidates)


def _person_template_en(required: dict[str, list[str]]) -> str:
    """Build a short comma-separated English caption from required slots."""
    parts: list[str] = []
    ages = required.get("age_group", [])
    gen = required.get("gender", [])
    if ages or gen:
        head = " ".join((ages[0] if ages else "person", gen[0] if gen else "")).strip()
        parts.append(head)
    elif "person" not in required:
        parts.append("person")

    up_col = required.get("upper_color", [])
    up_type = required.get("upper_type", [])
    if up_col or up_type:
        parts.append(
            " ".join([up_col[0] if up_col else "", up_type[0] if up_type else "upper clothing"]).strip()
        )

    lo_col = required.get("lower_color", [])
    lo_type = required.get("lower_type", [])
    if lo_col or lo_type:
        parts.append(
            " ".join([lo_col[0] if lo_col else "", lo_type[0] if lo_type else "lower clothing"]).strip()
        )

    eq = required.get("equipment", [])
    if eq:
        parts.append("carrying " + " and ".join(eq))

    act = required.get("action", [])
    if act:
        parts.append(f"{act[0]} posture")

    return ", ".join(p for p in parts if p)


def _vehicle_template_en(colors: list[str], types_: list[str]) -> str:
    parts: list[str] = []
    if colors and types_:
        parts.append(f"{colors[0]} {types_[0]}")
    elif types_:
        parts.append(types_[0])
    elif colors:
        parts.append(f"{colors[0]} vehicle")
    else:
        parts.append("vehicle")
    return ", ".join(parts)


# =====================================================================
# Gemma fallback parser
# =====================================================================


def parse_with_gemma(user_text: str, backend, build_messages=None) -> QueryJSON:
    """Use the Gemma backend to parse the query. Falls back to dictionary
    output if Gemma's response is unusable.

    ``build_messages`` (lab-only injection seam, mirrors plr_core.run_plr's
    parameter): optional callable ``(user_text) -> messages`` that replaces the
    module-level ``build_query_parser_messages`` constants — this is how
    ``lab run --version <V>`` sends the query-parser prompt from
    ``prompts/<V>.yaml`` instead of the hardcoded constants. ``None`` keeps the
    constants path byte-identical to the live core/ir behaviour.
    """
    builder = build_messages or build_query_parser_messages
    msgs = builder(user_text)
    try:
        gen = backend.generate(None, msgs, max_tokens=512, temperature=0.0)
    except Exception as e:
        log.warning("Gemma query parser failed (%s) — using dictionary only", e)
        return parse_with_dictionary(user_text)

    try:
        data = parse_plr_json(gen.raw)
    except Exception as e:
        log.warning("Gemma query JSON malformed (%s) — using dictionary only", e)
        return parse_with_dictionary(user_text)

    return _json_to_query(data, user_text)


def _json_to_query(data: dict[str, Any], user_text: str) -> QueryJSON:
    # qp_v0.4: Gemma now emits raw entities ({attributes: {upper_color:
    # "검정", ...}, free_form_residue: [...]}). Run the Python
    # normaliser to map them onto the enum-shaped QueryJSON required
    # slots. Fall through to the legacy reader if the response is
    # already in the old shape (covers the case where someone toggles
    # back to qp_v0.3 prompts without reloading the parser module).
    if "attributes" in data and "required" not in data:
        from query_normalizer import normalize_query_entities
        data = normalize_query_entities(data, user_text)

    q = QueryJSON(raw_text=user_text)
    q.query_type = data.get("query_type") or "person_search"
    q.target = data.get("target") or _target_from_query_type(q.query_type)
    q.required = {k: list(v) for k, v in (data.get("required") or {}).items() if v}
    q.optional = {k: list(v) for k, v in (data.get("optional") or {}).items() if v}
    q.excluded = {k: list(v) for k, v in (data.get("excluded") or {}).items() if v}
    q.ambiguous = list(data.get("ambiguous") or [])
    # Equipment group expansion. The dictionary parser already does this
    # via _EQUIPMENT_GROUP_KEYWORDS ("가방" → all bag-group members),
    # but Gemma typically picks a single equipment value ("handbag")
    # which then rejects every other bag-shaped match. Re-apply the
    # group lookup to both required and optional buckets here so the
    # two paths produce the same recall.
    try:
        from plr_schema import EQUIPMENT_TYPE_GROUP
        # Inverse: equipment label → its containing group(s).
        # If the user wrote "가방" and Gemma reduced it to "handbag",
        # we expand it back to the full bag group. We do NOT expand
        # when the user query *itself* contained a specific subtype
        # (e.g. "백팩" explicit) — heuristic: only expand when the
        # raw text contains the group keyword.
        for slot in ("equipment",):
            vals = q.required.get(slot) or q.optional.get(slot) or []
            if not vals:
                continue
            for grp_kw, grp_name in _EQUIPMENT_GROUP_KEYWORDS.items():
                if grp_kw not in user_text.lower():
                    continue
                members = EQUIPMENT_TYPE_GROUP.get(grp_name, ())
                if not members:
                    continue
                expanded = list(dict.fromkeys(list(vals) + list(members)))
                if slot in q.required:
                    q.required[slot] = expanded
                else:
                    q.optional[slot] = expanded
    except Exception:
        pass
    # Gemma + normalizer produce qp_v0.6 context-bound residue with
    # four fields: subject, attribute_ko, attribute_en, is_negative.
    # Older shapes are upgraded here: qp_v0.5 used `attribute` instead
    # of attribute_ko, and legacy paths emit bare strings. Anything
    # missing a subject falls back to the parsed target.
    raw_residue = data.get("free_form_residue") or []
    bound: list[dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()
    default_subj = q.target if q.target in {"person", "vehicle"} else (q.target or "object")
    for item in raw_residue:
        if isinstance(item, dict):
            subj = (str(item.get("subject") or "")).strip() or default_subj
            attr_ko = (str(item.get("attribute_ko")
                          or item.get("attribute")
                          or "")).strip()
            attr_en = (str(item.get("attribute_en") or "")).strip()
            is_neg = bool(item.get("is_negative", False))
        else:
            attr_ko = (str(item) or "").strip()
            attr_en = ""
            subj = default_subj
            is_neg = False
        if not (attr_ko or attr_en):
            continue
        key = (subj, (attr_ko or attr_en).lower(), is_neg)
        if key in seen:
            continue
        seen.add(key)
        bound.append({
            "subject": subj,
            "attribute_ko": attr_ko,
            "attribute_en": attr_en,
            "is_negative": is_neg,
        })
    q.free_form_residue = bound
    q.needs_temporal_search = bool(data.get("needs_temporal_search", False))
    q.template_caption_en = data.get("template_caption_en") or ""
    q.template_caption_ko = data.get("template_caption_ko") or user_text
    # Parity with the dictionary path: derive the coarse lower_shape hard gate +
    # fabric refine from whatever fine lower_type Gemma emitted. (any_color
    # routing on the gemma path is deferred to Step 2's prompt support.)
    _derive_groups_and_refine(q)
    return q


def _target_from_query_type(qt: str) -> str:
    return {
        "person_search": "person",
        "vehicle_search": "vehicle",
        "event_search": "event",
        "mixed_search": "mixed",
    }.get(qt, "person")


# =====================================================================
# Public entry point (chooses dictionary vs Gemma)
# =====================================================================


def parse_query(
    user_text: str, *, backend=None, force_gemma: bool = False, build_messages=None,
) -> QueryJSON:
    """Parse a user query into structured slots.

    qp_v0.3 (2026-05-26): Gemma is the primary parser. The dictionary
    parser used to be primary, which forced us to chase Korean particle
    edge cases ("옷을" vs "옷이" vs "옷" → all need to be treated the
    same) by hand-listing every form in `_PARTICLES_AND_FILLERS`. That
    is an open-ended whack-a-mole. Letting Gemma read the whole query
    plus the enum tables removes the issue entirely: the model decides
    what is a slot value and what is residue.

    Fallback path: if the Gemma backend isn't available or its response
    can't be parsed, we keep the old dictionary route so the system
    degrades to "still works, just narrower" instead of "nothing works".
    `force_gemma=True` skips even that fallback (used in tests).

    `build_messages` (lab-only, keyword): forwarded to parse_with_gemma so a
    per-version query-parser prompt (prompts/<V>.yaml) can replace the
    constants. Ignored on the dictionary path (no prompt is sent there).
    """
    if backend is not None:
        try:
            q = parse_with_gemma(user_text, backend, build_messages=build_messages)
            # Parity: apply the same surface-form intents the dictionary path
            # applies (e.g. 청바지 = blue denim). This closes the long-standing
            # gap where the production gemma path never narrowed "청바지".
            _apply_lexical_overrides(q, user_text)
            return q
        except Exception as e:
            log.warning("Gemma parser failed (%s), falling back to dictionary", e)
            if force_gemma:
                raise
    # No backend → dictionary parser is the best we can do (applies overrides
    # internally).
    return parse_with_dictionary(user_text)


# =====================================================================
# Pluggable-modules shim (A1 registry integration)
# =====================================================================
# Import the YamlParser provider so it self-registers with the registry
# at query_parser import time.  This preserves full backward-compat:
# callers that do ``import query_parser; query_parser.parse_query(...)``
# continue to work unchanged, while ``get_provider("parser")`` also
# resolves to the same YamlParser instance.
#
# The import is deferred + guarded so that circular-import scenarios and
# environments without the parser/ package (e.g. legacy containers) do
# not break the module.
try:
    import parser.yaml_parser as _yaml_parser_module  # noqa: F401
except Exception as _e:  # pragma: no cover
    log.debug("query_parser: could not import parser.yaml_parser (%s)", _e)
