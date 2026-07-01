"""Quality scoring + PLR mode decision for incoming crops.

The Gemma backend is expensive enough that we don't want to run it on crops
where the answer would be unreliable anyway. The quality gate produces a
single score in [0,1] and a mode:

  - normal_plr     : full PLR (no SR), trust output as-is
  - sr_candidate   : run SR alongside original, dual-view merge (sr_policy.merge_dual_view)
  - coarse_only    : skip Gemma; produce object_type + coarse color group only

Calibrated thresholds (CALIBRATED_THRESHOLDS) come from the Phase 4 experiment
documented in docs §7.4. The defaults below are placeholders — they MUST be
replaced with experiment-derived values before production.

Inputs:
  - PIL image (the crop itself)
  - Optional tracker metadata (bbox_stability hint from upstream)

Outputs:
  QualityReport with:
    score, mode, sub-scores (for logging / failure analysis)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# =====================================================================
# Calibrated thresholds (placeholders — replace via Phase 4 experiment)
# =====================================================================


@dataclass(frozen=True)
class CalibratedThresholds:
    """Bucket thresholds that decide normal_plr / sr_candidate / coarse_only.

    All fields are inclusive lower-bounds for the "better" mode.
    See docs §7.4 for the calibration procedure.
    """

    # Minimum crop height (pixels) to be eligible for each mode.
    min_height_normal: int = 96
    min_height_sr_candidate: int = 48
    min_height_coarse_only: int = 28

    # Quality score thresholds.
    normal_quality: float = 0.60
    sr_quality: float = 0.35
    coarse_only_quality: float = 0.20


CALIBRATED_THRESHOLDS = CalibratedThresholds()


def set_calibrated_thresholds(thresholds: CalibratedThresholds) -> None:
    """Override at process startup once the calibration experiment is done."""
    global CALIBRATED_THRESHOLDS
    CALIBRATED_THRESHOLDS = thresholds


# =====================================================================
# Sub-score weights (docs §7.1 formula)
# =====================================================================

W_RESOLUTION = 0.25
W_SHARPNESS = 0.20
W_BRIGHTNESS = 0.15
W_BODY_VISIBILITY = 0.15
W_OCCLUSION = 0.15
W_BBOX_STABILITY = 0.10

assert abs(
    W_RESOLUTION + W_SHARPNESS + W_BRIGHTNESS
    + W_BODY_VISIBILITY + W_OCCLUSION + W_BBOX_STABILITY - 1.0
) < 1e-9


# =====================================================================
# Data classes
# =====================================================================


@dataclass
class TrackerHint:
    """Optional upstream tracker info. None of these fields are required."""

    bbox_stability: float | None = None  # 0..1, higher = more stable bbox over track
    body_visibility: str | None = None   # 'full_body' | 'upper_only' | 'lower_only' | 'partial'
    occlusion: str | None = None         # 'none' | 'low' | 'medium' | 'high'


@dataclass
class QualityReport:
    """Aggregate quality report for one crop."""

    score: float                    # 0..1
    mode: str                       # 'normal_plr' | 'sr_candidate' | 'coarse_only'
    crop_width: int
    crop_height: int
    sub_scores: dict[str, float]    # individual component scores
    warnings: list[str]

    def to_visibility_block(self) -> dict[str, Any]:
        """Project quality info into the PLR visibility schema."""
        return {
            "image_quality": self.coarse_label(),
            "quality_score": round(self.score, 3),
            "quality_warnings": list(self.warnings),
        }

    def coarse_label(self) -> str:
        if self.score >= 0.7:
            return "good"
        if self.score >= 0.4:
            return "fair"
        return "poor"


# =====================================================================
# Sub-score calculators (each returns a value in [0, 1])
# =====================================================================


def resolution_score(height: int) -> float:
    """Higher crop heights → higher score. Saturates at 160 px."""
    if height <= 32:
        return 0.0
    if height >= 160:
        return 1.0
    return (height - 32) / (160 - 32)


def sharpness_score(arr_gray: np.ndarray) -> float:
    """Variance of Laplacian, mapped to [0, 1].

    Empirically, var(Laplacian) < 30 = blurry, > 300 = sharp.
    """
    # Discrete Laplacian via convolution. We use a 3x3 kernel approximation
    # implemented with numpy operations (avoid SciPy dependency).
    if arr_gray.size == 0:
        return 0.0
    g = arr_gray.astype(np.float32)
    # Pad so the boundary doesn't dominate
    pad = np.pad(g, 1, mode="edge")
    lap = (
        pad[0:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, 0:-2] + pad[1:-1, 2:]
        - 4.0 * g
    )
    v = float(lap.var())
    if v <= 30:
        return 0.0
    if v >= 300:
        return 1.0
    return (v - 30) / (300 - 30)


def brightness_score(arr_gray: np.ndarray) -> float:
    """Favors mid-range luminance, penalizes very dark or saturated bright."""
    if arr_gray.size == 0:
        return 0.0
    mean = float(arr_gray.mean())  # 0..255
    # Triangular function peaking at 128
    distance = abs(mean - 128) / 128.0  # 0 (best) .. 1 (worst)
    return max(0.0, 1.0 - distance)


_BODY_VIS_MAP = {
    "full_body": 1.0,
    "partial": 0.6,
    "upper_only": 0.5,
    "lower_only": 0.5,
    None: 1.0,
}

_OCCLUSION_MAP = {
    "none": 1.0,
    "low": 0.8,
    "medium": 0.5,
    "high": 0.2,
    None: 0.8,  # unknown — be mildly optimistic
}


def body_visibility_score(hint: TrackerHint | None) -> float:
    if hint is None:
        return 1.0
    return _BODY_VIS_MAP.get(hint.body_visibility, 0.7)


def occlusion_score(hint: TrackerHint | None) -> float:
    if hint is None:
        return _OCCLUSION_MAP[None]
    return _OCCLUSION_MAP.get(hint.occlusion, _OCCLUSION_MAP[None])


def bbox_stability_score(hint: TrackerHint | None) -> float:
    if hint is None or hint.bbox_stability is None:
        return 1.0
    return max(0.0, min(1.0, float(hint.bbox_stability)))


# =====================================================================
# Main entry point
# =====================================================================


def evaluate(
    pil_image: Image.Image, tracker_hint: TrackerHint | None = None
) -> QualityReport:
    """Run all sub-scores and pick a PLR mode."""
    w, h = pil_image.size
    warnings: list[str] = []

    # Greyscale once for sharpness + brightness
    gray = np.asarray(pil_image.convert("L"), dtype=np.uint8)

    sub = {
        "resolution": resolution_score(h),
        "sharpness": sharpness_score(gray),
        "brightness": brightness_score(gray),
        "body_visibility": body_visibility_score(tracker_hint),
        "occlusion": occlusion_score(tracker_hint),
        "bbox_stability": bbox_stability_score(tracker_hint),
    }
    score = (
        W_RESOLUTION * sub["resolution"]
        + W_SHARPNESS * sub["sharpness"]
        + W_BRIGHTNESS * sub["brightness"]
        + W_BODY_VISIBILITY * sub["body_visibility"]
        + W_OCCLUSION * sub["occlusion"]
        + W_BBOX_STABILITY * sub["bbox_stability"]
    )
    score = float(max(0.0, min(1.0, score)))

    # Mode decision
    t = CALIBRATED_THRESHOLDS
    if h < t.min_height_coarse_only or score < t.coarse_only_quality:
        mode = "coarse_only"
        warnings.append(f"low quality (h={h} score={score:.2f})")
    elif h < t.min_height_normal or score < t.normal_quality:
        if h < t.min_height_sr_candidate or score < t.sr_quality:
            mode = "coarse_only"
            warnings.append(f"below SR threshold (h={h} score={score:.2f})")
        else:
            mode = "sr_candidate"
            warnings.append("SR-eligible due to size/quality")
    else:
        mode = "normal_plr"

    return QualityReport(
        score=score,
        mode=mode,
        crop_width=w,
        crop_height=h,
        sub_scores={k: round(v, 3) for k, v in sub.items()},
        warnings=warnings,
    )


def coarse_only_plr_json(
    obj_type_hint: str,
    report: QualityReport,
    dominant_color: str | None = None,
) -> dict[str, Any]:
    """Build a minimal PLR JSON for crops that fail the quality gate.

    The structured field is intentionally sparse — only object_type and a
    coarse color group, both of which are still useful for search filtering.
    """
    from plr_schema import PROMPT_VERSION, color_group  # local import to avoid cycle

    base = {
        "object_type": obj_type_hint if obj_type_hint in ("person", "vehicle") else "person",
        "visibility": report.to_visibility_block(),
        "attributes": {},
        "prompt_version": PROMPT_VERSION,
        "_coarse_only": True,
    }
    if dominant_color:
        base["attributes"]["coarse_color_group"] = color_group(dominant_color)
    return base


def report_as_dict(r: QualityReport) -> dict[str, Any]:
    """Serializable representation for logging/storage."""
    d = asdict(r)
    return d
