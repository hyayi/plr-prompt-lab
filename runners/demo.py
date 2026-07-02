"""demo — GPU-free onboarding module for `lab demo`.

Builds a tiny synthetic dataset in ./demo_dataset/, runs re_score with a
MockModel, evaluates accuracy, and prints a walkthrough of what happened.
No GPU, no DB, no Redis required.

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
    MOCK_YAML_V2_ALL_MALE as _MOCK_YAML_V2,
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
        model: "mock_v1"
        prompt: "prompts/plr_v1.4_cot.yaml"
    """)
    (demo_dir / "manifest.yaml").write_text(manifest_text, encoding="utf-8")

    return demo_dir


# =====================================================================
# Eval helper (mirrors _run_eval_for in test_cycle_e2e.py)
# =====================================================================

def _run_eval(gdir: Path, version: str, ledger_path: Path, lab_root: Path) -> tuple[dict, str]:
    """Run run_eval.main() and return (ledger_record, stdout_text)."""
    import importlib.util
    import io
    from contextlib import redirect_stdout

    spec = importlib.util.spec_from_file_location(
        "run_eval_demo",
        str(lab_root / "eval" / "run_eval.py"),
    )
    run_eval = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(run_eval)  # type: ignore[union-attr]

    orig_argv = sys.argv
    sys.argv = [
        "run_eval",
        "--attribute", "gender",
        "--golden", str(gdir),
        "--version", version,
        "--ledger", str(ledger_path),
        "--date", "2026-07-02T00:00:00",
    ]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_eval.main()
    except SystemExit:
        pass
    finally:
        sys.argv = orig_argv

    # Read back the ledger record just written (the last one)
    records = _read_jsonl(ledger_path)
    last = records[-1] if records else {}
    return last, buf.getvalue()


# =====================================================================
# Main demo runner
# =====================================================================

def run_demo(lab_root: Path, keep_dir: bool = False) -> int:
    """Run the full GPU-free demo cycle. Returns exit code (0 on success)."""
    from runners import re_score as rs

    demo_dir = lab_root / "demo_dataset"

    print("=" * 65)
    print("  lab demo — GPU-free PLR prompt-lab onboarding")
    print("=" * 65)
    print()
    print("Step 1/5  Build synthetic dataset")
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

    # ---- Version 1: MockModel predicts female (should score 1.0) ----
    print("Step 2/5  Re-score with MockModel v1 (predicts: female)")
    print("          [no GPU — MockModel returns canned PLR YAML]")
    meta_v1 = rs.re_score("gender", MockModel(_MOCK_YAML_V1), golden_dir=str(demo_dir))
    preds_v1 = _read_jsonl(demo_dir / "predictions.jsonl")
    print(f"          re_score wrote {meta_v1['n']} predictions")
    print(f"          sample: {preds_v1[0]}")
    print()

    ledger_path = demo_dir / "ledger.jsonl"

    print("Step 3/5  Evaluate mock_v1 predictions vs ground truth")
    rec_v1, out_v1 = _run_eval(demo_dir, "mock_v1", ledger_path, lab_root)
    acc_v1 = rec_v1.get("accuracy", 0.0)
    print(f"          accuracy: {acc_v1:.3f}  (all female predicted = all correct)")
    print(f"          ledger record appended: {ledger_path.name}")
    print()

    # ---- Version 2: MockModel predicts male (should score 0.0) ----
    print("Step 4/5  Re-score with MockModel v2 (predicts: male) — to show a Δ")
    meta_v2 = rs.re_score("gender", MockModel(_MOCK_YAML_V2), golden_dir=str(demo_dir))
    preds_v2 = _read_jsonl(demo_dir / "predictions.jsonl")
    print(f"          re_score wrote {meta_v2['n']} predictions")
    print(f"          sample: {preds_v2[0]}")
    print()

    print("Step 5/5  Evaluate mock_v2 predictions vs ground truth")
    rec_v2, out_v2 = _run_eval(demo_dir, "mock_v2", ledger_path, lab_root)
    acc_v2 = rec_v2.get("accuracy", 0.0)
    delta = acc_v2 - acc_v1
    print(f"          accuracy: {acc_v2:.3f}  (all male predicted vs female labels)")
    print(f"          Δ vs mock_v1: {delta:+.3f}")
    print()

    # Verify ledger
    ledger_records = _read_jsonl(ledger_path)
    print(f"          ledger now has {len(ledger_records)} records:")
    for lr in ledger_records:
        print(f"            version={lr['version']}  accuracy={lr['accuracy']:.3f}")
    print()

    print("=" * 65)
    print("  Demo complete — what just happened:")
    print()
    print("  1. A synthetic dataset was built in demo_dataset/")
    print("     (crops + labels.jsonl + predictions.jsonl + manifest.yaml)")
    print("  2. re_score() was called twice with a MockModel (no GPU).")
    print("     In real use: 'lab run --attribute gender --version X'")
    print("     calls LabGemmaModel on real CCTV crops (GPU required).")
    print("  3. run_eval() scored each version against ground truth.")
    print("     In real use: 'lab eval --attribute gender --version X'")
    print("  4. Two ledger records show a Δ between versions —")
    print("     this is the signal you iterate on when editing prompts.")
    print()
    print("  Next steps:")
    print("  - Read HANDOFF.md for the full prompt-engineer workflow.")
    print("  - See DATASET_SPEC.md for the dataset directory format.")
    print("  - For a real run: see INSTALL.md (GPU + model required).")
    print("=" * 65)

    if not keep_dir:
        shutil.rmtree(demo_dir, ignore_errors=True)

    return 0
