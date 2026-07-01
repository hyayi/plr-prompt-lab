"""PLR (Person/Vehicle attribute extraction) JSON schema + enum catalog.

The Gemma backend produces JSON conforming to these schemas. The same enums
are used by:
  - plr_prompts.py  (prompt construction)
  - template_caption.py  (deterministic caption generation)
  - query_parser.py  (Korean → English enum normalization)
  - scoring.py  (slot matching + coarse-group bonus)

Everything is in English internally (per the design's "internal language unified
to English" decision); Korean enters/exits only at the query parser and the
caption_ko/caption_en outputs.
"""

from __future__ import annotations

from typing import Any, Final


# =====================================================================
# Color enum (shared by upper/lower clothing, equipment, vehicle)
# =====================================================================

COLOR_ENUM: Final[tuple[str, ...]] = (
    # achromatic
    "black", "dark_gray", "gray", "light_gray", "white", "silver",
    # warm
    "red", "orange", "yellow", "brown", "dark_brown", "beige", "cream", "gold",
    # cool
    "blue", "navy", "light_blue", "green", "dark_green",
    # military / tactical
    "military_olive",  # Korean 국방색: dark olive-drab used in military uniforms/vehicles
    # other
    "purple", "pink",
    "mixed_pattern",   # stripe/check/floral — single color undecidable
    "multi_color",     # two+ colors in roughly equal proportion
)

COLOR_GROUP: Final[dict[str, tuple[str, ...]]] = {
    "dark": ("black", "dark_gray", "dark_brown", "dark_green", "military_olive"),
    # The old `light` group lumped together five colours that look
    # quite different on a CCTV crop — white, cream, beige (warm
    # whites) on one side and silver, light_gray (cool metallics)
    # on the other. The cascade hard-filter only checks group
    # overlap, so a "흰색 세단" query was pulling in every silver
    # car too (e.g. 85 white + 30 silver on the production video).
    # Splitting the two families keeps recall high inside each
    # family while preventing the cross-family false positives the
    # user reported on 2026-05-26.
    "light_white": ("white", "cream", "beige"),
    "metallic_light": ("silver", "light_gray"),
    "vivid": ("red", "orange", "yellow", "green", "purple", "pink", "gold"),
    # `blue` is its own group: navy / light_blue used to be split across
    # dark / light and `blue` itself lived under vivid. The result was
    # that the cascade hard-filter's group-overlap match passed almost
    # everything for "청바지" (which expands to lower_color=[blue, navy,
    # light_blue]) — blue ∈ vivid, navy ∈ dark, light_blue ∈ light, so a
    # candidate of any colour would match at least one group. Carving out
    # a single `blue` group keeps the "denim intent" precise: navy and
    # light_blue belong to the blue family, not to dark or light.
    "blue": ("blue", "navy", "light_blue"),
    "neutral": ("gray", "brown"),
}


def color_group(label: str) -> str:
    """Map a color label to its coarse group: dark | light | vivid | neutral | unknown."""
    for grp, members in COLOR_GROUP.items():
        if label in members:
            return grp
    return "unknown"


# Fine perceptual grouping for EXACT-intent colour HARD gates ("검은색 차",
# "검은 옷"). Deliberately tighter than the coarse COLOR_GROUP, which lumps every
# dark hue (black + dark_gray + dark_brown + dark_green + military_olive) into one
# "dark" band. That coarse band is correct for brightness queries ("어두운 차")
# and soft-score bonuses, but wrong for a specific colour: a user asking for a
# "검은색 차" does not want brown / green / 국방색 cars. Here black groups only
# with dark_gray (achromatic dark); each chromatic dark stays with its own hue.
# Used ONLY by passes_hard_filter's colour slots — color_group() is unchanged.
COLOR_HARD_GROUP: Final[dict[str, tuple[str, ...]]] = {
    "black": ("black", "dark_gray"),        # achromatic dark
    "gray": ("gray",),
    "white": ("white", "cream", "beige"),
    "silver": ("silver", "light_gray"),
    "red": ("red",),
    "orange": ("orange",),
    "yellow": ("yellow", "gold"),
    "brown": ("brown", "dark_brown"),
    "green": ("green", "dark_green"),
    "blue": ("blue", "navy", "light_blue"),
    "purple": ("purple",),
    "pink": ("pink",),
    "military_olive": ("military_olive",),   # 국방색 — never folds into black
}


def color_hard_group(label: str) -> str:
    """Fine colour group for hard gating. Falls back to the label itself so an
    unlisted colour only ever matches itself (never a broad band)."""
    for grp, members in COLOR_HARD_GROUP.items():
        if label in members:
            return grp
    return label or "unknown"


# =====================================================================
# Person attributes
# =====================================================================

GENDER_ENUM: Final[tuple[str, ...]] = ("male", "female")
AGE_GROUP_ENUM: Final[tuple[str, ...]] = ("adult", "child")

OUTFIT_TYPE_ENUM: Final[tuple[str, ...]] = (
    "two_piece",   # upper + lower separately
    "one_piece",   # dress / jumpsuit / one-piece uniform
    "layered",     # heavy coat hides lower body split
    "obscured",    # occlusion/lighting — undecidable
)

UPPER_TYPE_ENUM: Final[tuple[str, ...]] = (
    # outerwear
    "jacket", "coat", "padding", "windbreaker", "vest", "cardigan",
    # tops
    "tshirt", "long_sleeve_tee", "shirt", "blouse", "hoodie", "sweater", "knit",
    # uniform/work
    "uniform", "safety_vest", "workwear",
    # one-piece (only when outfit_type=one_piece)
    "dress", "one_piece_uniform", "jumpsuit",
    # unknown
    "upper_unknown",
)

UPPER_TYPE_GROUP: Final[dict[str, tuple[str, ...]]] = {
    "upper_outerwear": ("jacket", "coat", "padding", "windbreaker", "vest", "cardigan"),
    "upper_top": ("tshirt", "long_sleeve_tee", "shirt", "blouse", "hoodie", "sweater", "knit"),
    "uniform_like": ("uniform", "safety_vest", "workwear"),
    "one_piece_garment": ("dress", "one_piece_uniform", "jumpsuit"),
}


def upper_type_group(label: str) -> str:
    for grp, members in UPPER_TYPE_GROUP.items():
        if label in members:
            return grp
    return "unknown"


LOWER_TYPE_ENUM: Final[tuple[str, ...]] = (
    "pants", "jeans", "training_pants", "slacks", "leggings",
    "shorts",
    "skirt", "long_skirt",
    "none",            # outfit_type=one_piece (no separate lower garment)
    "lower_unknown",
)

LOWER_TYPE_GROUP: Final[dict[str, tuple[str, ...]]] = {
    "pants_like": ("pants", "jeans", "training_pants", "slacks", "leggings"),
    "shorts_like": ("shorts",),
    "skirt_like": ("skirt", "long_skirt"),
    "no_separate": ("none",),
}


def lower_type_group(label: str) -> str:
    for grp, members in LOWER_TYPE_GROUP.items():
        if label in members:
            return grp
    return "unknown"


# ---------------------------------------------------------------------
# Shape axis — CCTV-reliable HARD-FILTER level (redesign 2026-06).
# `lower_type` (jeans/slacks/...) stays a fine REFINE-only label; the hard
# gate runs on coarse SHAPE, which a small CCTV crop can actually tell apart
# (long_pants vs shorts vs skirt length). For rows indexed before the
# dedicated `lower_shape` field exists, shape is DERIVED from the stored
# fine `lower_type` label at read time (no reindex needed for this axis).
# ---------------------------------------------------------------------

LOWER_SHAPE_ENUM: Final[tuple[str, ...]] = (
    "long_pants", "shorts", "long_skirt", "short_skirt",
    "none", "lower_unknown",
)

# Fine lower_type label -> coarse shape (read-time derivation for old rows).
LOWER_TYPE_TO_SHAPE: Final[dict[str, str]] = {
    "pants": "long_pants", "jeans": "long_pants", "slacks": "long_pants",
    "training_pants": "long_pants", "leggings": "long_pants",
    "shorts": "shorts",
    "skirt": "short_skirt", "long_skirt": "long_skirt",
    "none": "none",
    "lower_unknown": "lower_unknown",
}


def lower_shape_of(lower_type_label: str | None) -> str:
    """Coarse lower shape for the hard gate, derived from a fine lower_type
    label. Returns 'lower_unknown' for missing/unrecognised labels."""
    if not lower_type_label:
        return "lower_unknown"
    return LOWER_TYPE_TO_SHAPE.get(lower_type_label, "lower_unknown")


# Upper sleeve axis — HARD-FILTER level. Newly EXTRACTED field; absent on
# pre-redesign rows -> the gate must wildcard-pass (handled in scoring).
UPPER_SLEEVE_ENUM: Final[tuple[str, ...]] = ("long", "short", "unknown")


def upper_outer_of(upper_type_label: str | None) -> str:
    """Read-time derivation of the outer-layer group from a fine upper_type
    label: returns 'upper_outerwear' when the visible top IS outerwear, else
    'none'. (Pre-redesign rows carry a single upper_type; the dedicated
    `upper_outer` extracted field supersedes this once reindexed.)"""
    if not upper_type_label:
        return "none"
    return "upper_outerwear" if upper_type_group(upper_type_label) == "upper_outerwear" else "none"


# Equipment body-location — HARD-FILTER level (presence@location).
EQUIP_LOCATION_ENUM: Final[tuple[str, ...]] = (
    "head", "eye", "face", "back", "hand",
)


EQUIPMENT_TYPE_ENUM: Final[tuple[str, ...]] = (
    # Bags
    "backpack", "shoulder_bag", "handbag", "cross_bag", "cart",
    # Outdoor / weather
    "umbrella",
    # Head wear — v0.9 collapsed beanie into hat (Gemma E4B can't reliably
    # tell a knit beanie from a baseball cap from a brimmed hat in CCTV
    # crops, so we route every soft headwear to "hat" and rely on the
    # query_parser to map 캡/비니/베레모 → hat for searches).
    "helmet", "construction_helmet", "hat",
    # Face / accessories
    "mask", "glasses", "sunglasses", "headphones",
    # Hand-held objects (everyday)
    "phone_in_hand",
    "handheld_object",      # generic "something in hand", uncategorisable
    "bottle", "clipboard",  # common workplace / public objects
    # Weapons & dangerous items (security search)
    "knife", "firearm", "bat_stick", "sharp_object",
    # Mobility / other
    "cane", "stroller", "other_equipment",
)

# Coarse-group for equipment so query_parser's hard filter can pass any bag
# when the user just says "가방", or any weapon when they say "무기".
EQUIPMENT_TYPE_GROUP: Final[dict[str, tuple[str, ...]]] = {
    "bag": ("backpack", "shoulder_bag", "handbag", "cross_bag"),
    "headwear": ("helmet", "construction_helmet", "hat"),
    "face_accessory": ("mask", "glasses", "sunglasses", "headphones"),
    "weapon": ("knife", "firearm", "bat_stick", "sharp_object"),
    "handheld": ("phone_in_hand", "handheld_object", "bottle", "clipboard"),
    "mobility": ("cart", "cane", "stroller"),
    "weather": ("umbrella",),
}


def equipment_type_group(label: str) -> str:
    for grp, members in EQUIPMENT_TYPE_GROUP.items():
        if label in members:
            return grp
    return "unknown"

STATIC_ACTION_ENUM: Final[tuple[str, ...]] = (
    "standing", "walking_like", "running_like", "sitting", "bending", "squatting",
    "lying_or_fallen_candidate",   # confirmation is temporal-module's job
    "using_phone", "carrying_object", "pushing_or_pulling",
    # Riding a two-wheel vehicle. PLR will set object_type=person on the
    # rider and put the vehicle kind into action so a single query like
    # "오토바이 탄 사람" reaches the rider, not the bike crop.
    "riding_motorcycle", "riding_bicycle", "riding_scooter", "riding_kickboard",
    "posture_unknown",
)


# =====================================================================
# Vehicle attributes
# =====================================================================

VEHICLE_TYPE_ENUM: Final[tuple[str, ...]] = (
    # Passenger — light_car is the Korean "경차" bracket (Morning /
    # Casper / Ray / Spark sized), which the v0.8 enum lumped into
    # hatchback. Adding it back as a top-level category gives the
    # search side a single bucket for "경차" queries without
    # re-introducing model names.
    "sedan", "suv", "hatchback", "light_car", "van", "minivan",
    # Commercial / utility
    "pickup_truck", "truck", "bus", "taxi",
    # Emergency
    "ambulance", "police_car", "fire_truck", "emergency_vehicle",
    # Two-wheel
    "motorcycle", "scooter", "bicycle", "kickboard",
    # Special
    "construction_vehicle",
    # Unknown
    "vehicle_unknown",
)
# v0.8 had ~30 Korean/foreign model names (sonata/grandeur/bmw/...).
# Production re-index showed Gemma E4B identified only 2/214 cars as
# "bmw" and 0 for everything else, so the model labels added prompt
# tokens without improving recall. Dropped for v0.9.

VEHICLE_TYPE_GROUP: Final[dict[str, tuple[str, ...]]] = {
    "passenger": ("sedan", "suv", "hatchback", "light_car", "van", "minivan"),
    "commercial": ("pickup_truck", "truck", "bus", "taxi"),
    "emergency": ("ambulance", "police_car", "fire_truck", "emergency_vehicle"),
    "two_wheel": ("motorcycle", "scooter", "bicycle", "kickboard"),
    "special": ("construction_vehicle",),
}


def vehicle_type_group(label: str) -> str:
    for grp, members in VEHICLE_TYPE_GROUP.items():
        if label in members:
            return grp
    return "unknown"


# =====================================================================
# Military judgment (prompt-native, plr_v1.4_cot)
# =====================================================================

# Gemma judges military/civilian directly from camouflage / field uniform /
# load-bearing gear cues (person) or camo paint / military body type / insignia
# (vehicle), instead of inferring it post-hoc from a single olive colour. The
# parser pins the output to these three values; indexing._attach_military_flags
# turns "military" into is_soldier / is_military (with olive kept as a fallback).
MILITARY_ENUM: Final[tuple[str, ...]] = ("military", "civilian", "unknown")


# =====================================================================
# JSON Schema (jsonschema-compatible) for Person/Vehicle PLR output
# =====================================================================

PROMPT_VERSION: Final[str] = "plr_v0.4"


def _topk_array_schema(label_enum: tuple[str, ...]) -> dict[str, Any]:
    """Helper: array of {label: enum, score: 0..1}."""
    return {
        "type": "array",
        # minItems=0 — production data has frequent "color unknown" cases (low-res
        # crops, motion blur, occlusion). normalize_plr_json() fills in a single
        # placeholder when an array is missing/empty; if the model legitimately
        # returns nothing, indexing should not block.
        "minItems": 0,
        "maxItems": 5,
        "items": {
            "type": "object",
            "required": ["label", "score"],
            "properties": {
                "label": {"type": "string", "enum": list(label_enum)},
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
    }


PERSON_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["object_type", "attributes"],
    "properties": {
        "object_type": {"const": "person"},
        "track_id": {"type": "string"},
        "visibility": {
            "type": "object",
            "properties": {
                "body_visibility": {
                    "type": "string",
                    "enum": ["full_body", "upper_only", "lower_only", "partial"],
                },
                "occlusion": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high"],
                },
                "image_quality": {
                    "type": "string",
                    "enum": ["good", "fair", "poor"],
                },
                "quality_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "usable_for_attribute": {"type": "boolean"},
                "quality_warnings": {"type": "array", "items": {"type": "string"}},
            },
        },
        "attributes": {
            "type": "object",
            "required": [
                "gender_scores", "age_group_scores", "outfit_type_scores",
                "upper_clothing", "lower_clothing",
                "static_action_state_scores",
            ],
            "properties": {
                "gender_scores": {
                    "type": "object",
                    "required": ["male", "female", "selected"],
                    "properties": {
                        "male": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "female": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "selected": {"type": "string", "enum": list(GENDER_ENUM)},
                        "decision_margin": {"type": "number"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "caution": {"type": "string"},
                    },
                },
                "age_group_scores": {
                    "type": "object",
                    "required": ["adult", "child", "selected"],
                    "properties": {
                        "adult": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "child": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "selected": {"type": "string", "enum": list(AGE_GROUP_ENUM)},
                        "decision_margin": {"type": "number"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "outfit_type_scores": {
                    "type": "object",
                    "required": ["selected"],
                    "properties": {
                        "two_piece": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "one_piece": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "layered": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "obscured": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "selected": {"type": "string", "enum": list(OUTFIT_TYPE_ENUM)},
                        "decision_margin": {"type": "number"},
                    },
                },
                "upper_clothing": {
                    "type": "object",
                    "required": ["color_topk", "type_topk"],
                    "properties": {
                        "color_topk": _topk_array_schema(COLOR_ENUM),
                        "type_topk": _topk_array_schema(UPPER_TYPE_ENUM),
                        "evidence": {"type": "string"},
                        # Scalar derived field for JSONB-containment queries (set at index time).
                        "primary_color": {"type": "string", "enum": list(COLOR_ENUM)},
                    },
                },
                "lower_clothing": {
                    "type": "object",
                    "required": ["color_topk", "type_topk"],
                    "properties": {
                        "color_topk": _topk_array_schema(COLOR_ENUM),
                        "type_topk": _topk_array_schema(LOWER_TYPE_ENUM),
                        "evidence": {"type": "string"},
                        # Scalar derived field for JSONB-containment queries (set at index time).
                        "primary_color": {"type": "string", "enum": list(COLOR_ENUM)},
                    },
                },
                "equipment": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "score"],
                        "properties": {
                            "type": {"type": "string", "enum": list(EQUIPMENT_TYPE_ENUM)},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "color_topk": _topk_array_schema(COLOR_ENUM),
                            "evidence": {"type": "string"},
                        },
                    },
                },
                "static_action_state_scores": {
                    "type": "object",
                    "required": ["selected"],
                    "properties": {
                        "selected": {"type": "string", "enum": list(STATIC_ACTION_ENUM)},
                        "evidence": {"type": "string"},
                        # All actions in STATIC_ACTION_ENUM are optional numeric scores.
                        # Validator is permissive — schema rejects only invalid enum
                        # values, not missing scores.
                    },
                    "additionalProperties": True,
                },
                # Scalar derived field for JSONB-containment queries (B4 populates).
                "is_soldier": {"type": "boolean"},
                # Prompt-native military judgment (plr_v1.4_cot). Gemma emits this
                # directly from camouflage / field-uniform / load-bearing cues.
                "military": {"type": "string", "enum": list(MILITARY_ENUM)},
                # Optional: details about the vehicle the person is riding.
                # Populated only when action is riding_*. Lets queries like
                # "빨간 오토바이 탄 남자" match on rider_vehicle_color/type
                # in addition to action=riding_motorcycle and gender=male.
                "rider_vehicle": {
                    "type": "object",
                    "required": [],
                    "properties": {
                        "color_topk": _topk_array_schema(COLOR_ENUM),
                        "type": {
                            "type": "string",
                            "enum": ["motorcycle", "bicycle", "scooter", "kickboard", "unknown"],
                        },
                    },
                    "additionalProperties": True,
                },
            },
        },
        "prompt_version": {"type": "string"},
    },
}


VEHICLE_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["object_type", "attributes"],
    "properties": {
        "object_type": {"const": "vehicle"},
        "track_id": {"type": "string"},
        "visibility": {
            "type": "object",
            "properties": {
                "vehicle_visibility": {"type": "string", "enum": ["full", "partial"]},
                "occlusion": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high"],
                },
                "image_quality": {
                    "type": "string",
                    "enum": ["good", "fair", "poor"],
                },
                "quality_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "quality_warnings": {"type": "array", "items": {"type": "string"}},
            },
        },
        "attributes": {
            "type": "object",
            "required": ["color_topk", "type_topk"],
            "properties": {
                "color_topk": _topk_array_schema(COLOR_ENUM),
                "type_topk": _topk_array_schema(VEHICLE_TYPE_ENUM),
                "evidence": {"type": "string"},
                # Scalar derived fields for JSONB-containment queries (set at index time).
                "primary_color": {"type": "string", "enum": list(COLOR_ENUM)},
                "is_military": {"type": "boolean"},
                # Prompt-native military judgment (plr_v1.4_cot). Gemma emits this
                # directly from camo paint / military body type / insignia cues.
                "military": {"type": "string", "enum": list(MILITARY_ENUM)},
            },
        },
        "prompt_version": {"type": "string"},
    },
}


# =====================================================================
# Validation
# =====================================================================


class SchemaValidationError(ValueError):
    """Raised when a PLR JSON does not match the expected schema."""


def validate_plr(data: dict[str, Any]) -> None:
    """Validate a PLR JSON dict. Raises SchemaValidationError on failure.

    Uses jsonschema if available, falls back to a minimal structural check.
    """
    obj_type = data.get("object_type")
    if obj_type == "person":
        schema = PERSON_SCHEMA
    elif obj_type == "vehicle":
        schema = VEHICLE_SCHEMA
    else:
        raise SchemaValidationError(
            f"object_type must be 'person' or 'vehicle', got {obj_type!r}"
        )

    try:
        import jsonschema  # type: ignore
    except ImportError:
        # Fallback minimal check
        attrs = data.get("attributes")
        if not isinstance(attrs, dict):
            raise SchemaValidationError("attributes must be a dict")
        return

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise SchemaValidationError(f"PLR JSON invalid: {e.message}") from e


def is_valid_plr(data: dict[str, Any]) -> bool:
    """Return True if the dict validates as a PLR JSON (person or vehicle)."""
    try:
        validate_plr(data)
        return True
    except SchemaValidationError:
        return False
