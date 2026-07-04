"""demo — GPU-free onboarding module for `lab demo`.

Builds a tiny synthetic dataset in ./datasets/demo/, runs re_score with a
MockModel to produce attributes.jsonl/predictions.jsonl, and prints a
walkthrough of what happened. No GPU, no DB, no Redis required.

Scoring is NOT done locally: since RE-004 (2026-07) the lab hands off to the
eval server — run `lab submit` (with `--pull` to fetch the rendered
metrics/report/gallery). This demo therefore stops at the run step and points
the user at `lab submit`.

Design mirrors tests/test_cycle_e2e.py's MockModel + synthetic-fixture pattern.
"""
from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

# =====================================================================
# MockModel — the ONE canonical stub now lives in gemma_model (shared by
# demo, registry and the tests).  Import it here and alias the YAML
# constants so the rest of this module is unchanged.
# =====================================================================

from gemma_model import (  # noqa: E402
    MockModel,
    MOCK_YAML_V1_ALL_FEMALE as _MOCK_YAML_V1,
)


# =====================================================================
# Synthetic dataset builder
# =====================================================================

_OBJ_IDS = ["demo_001", "demo_002", "demo_003", "demo_004", "demo_005"]
# Ground truth: all five crops are female
_TRUE_LABELS = {oid: "female" for oid in _OBJ_IDS}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_tiny_jpg(path: Path, rgb: tuple[int, int, int] = (128, 100, 90)) -> None:
    """Write a minimal 100x150 JPEG — large enough for quality_gate normal_plr mode."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (100, 150), rgb)
    img.save(str(path), format="JPEG")


def build_synthetic_dataset(demo_dir: Path) -> Path:
    """Create a minimal valid dataset in demo_dir and return its path."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = demo_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    # 1. Crop images (5 tiny JPEGs, each a slightly different shade)
    for i, oid in enumerate(_OBJ_IDS):
        shade = 100 + i * 10
        _write_tiny_jpg(crops_dir / f"{oid}.jpg", rgb=(shade, shade - 10, shade - 20))

    # 2. predictions.jsonl — bootstrap (re_score will overwrite)
    _write_jsonl(
        demo_dir / "predictions.jsonl",
        [{"obj_id": oid, "pred": "unknown", "reason": ""} for oid in _OBJ_IDS],
    )

    # 3. labels.jsonl — ground truth (all female)
    _write_jsonl(
        demo_dir / "labels.jsonl",
        [{"obj_id": oid, "label": label} for oid, label in _TRUE_LABELS.items()],
    )

    # 4. manifest.yaml
    manifest_text = textwrap.dedent(f"""\
        attribute: gender
        n: {len(_OBJ_IDS)}
        created: "2026-07-02"
        source_note: "synthetic demo dataset — {len(_OBJ_IDS)} dummy person crops"
    """)
    (demo_dir / "manifest.yaml").write_text(manifest_text, encoding="utf-8")

    return demo_dir


# =====================================================================
# Main demo runner
# =====================================================================

def run_demo(lab_root: Path, keep_dir: bool = False) -> int:
    """Run the GPU-free demo run step. Returns exit code (0 on success).

    Scoring is intentionally NOT done here — it moved to the eval server
    (RE-004). The demo builds a synthetic dataset and re-scores it with a
    MockModel to produce attributes.jsonl/predictions.jsonl, then points the
    user at `lab submit` for the server-side metrics/report/gallery.
    """
    from runners import re_score as rs

    demo_dir = lab_root / "datasets" / "demo"

    print("=" * 65)
    print("  lab demo — GPU-free PLR prompt-lab onboarding")
    print("=" * 65)
    print()
    print("Step 1/2  Build synthetic dataset")
    print(f"          → {demo_dir}")

    # Clean slate if re-running
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    build_synthetic_dataset(demo_dir)

    n = len(_OBJ_IDS)
    print(f"          {n} dummy crops written (100x150 JPEG each)")
    print(f"          labels.jsonl: all {n} crops are 'female' (ground truth)")
    print(f"          manifest.yaml: attribute=gender, n={n}")
    print()

    # ---- MockModel re-score (predicts female) — produces run artifacts ----
    print("Step 2/2  Re-score with MockModel (predicts: female)")
    print("          [no GPU — MockModel returns canned PLR YAML]")
    meta = rs.re_score("gender", MockModel(_MOCK_YAML_V1), golden_dir=str(demo_dir), model_name="mock")
    preds = _read_jsonl(demo_dir / "predictions.jsonl")
    print(f"          re_score wrote {meta['n']} predictions")
    print(f"          sample: {preds[0]}")
    print(f"          artifacts: attributes.jsonl + predictions.jsonl in {demo_dir.name}/")
    print()

    print("=" * 65)
    print("  Demo complete — what just happened:")
    print()
    print("  1. A synthetic dataset was built in datasets/demo/")
    print("     (crops + labels.jsonl + predictions.jsonl + manifest.yaml)")
    print("  2. re_score() was called with a MockModel (no GPU), producing")
    print("     attributes.jsonl + predictions.jsonl.")
    print("     In real use: 'lab run --attribute gender --version X'")
    print("     calls LabGemmaModel on real CCTV crops (GPU required).")
    print()
    print("  채점은 로컬이 아니라 평가 서버에서 이뤄집니다:")
    print("  - 'lab submit'으로 서버에 제출하세요 (--pull로 지표/리포트/갤러리 회수).")
    print()
    print("  Next steps:")
    print("  - Read HANDOFF.md for the full prompt-engineer workflow.")
    print("  - See DATASET_SPEC.md for the dataset directory format.")
    print("  - For a real run: see INSTALL.md (GPU + model required).")
    print("=" * 65)

    if not keep_dir:
        shutil.rmtree(demo_dir, ignore_errors=True)

    return 0
