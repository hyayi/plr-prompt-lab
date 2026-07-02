#!/usr/bin/env python3
"""Golden-set eval for any PLR attribute (loop-engineering C2/C3).

Joins stored model predictions (predictions.jsonl) with human ground-truth
(labels.jsonl) for one attribute (gender / vehicle_type / military / ...),
reports accuracy + confusion, DIFFS against the previous ledger entry for the
same (attribute, version), and appends a version-keyed record to ledger.jsonl.

The current prompt version is scored from the DB snapshot (no Gemma). Scoring a
*changed* prompt on the same crops without a full reindex is a separate GPU step
(re_score.py, future) that re-writes predictions.jsonl for the new version.

Usage:
    python3 run_eval.py --attribute gender \
        --golden golden/gender --version plr_v1.4_cot

predictions.jsonl: {"obj_id": "1003", "pred": "male", "reason": "...", "margin": "0.9"}
labels.jsonl:      {"obj_id": "1003", "label": "female"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

# Lab root (one level above eval/) must be importable for the shared provenance
# helper whether this runs standalone or loaded via importlib from lab.py.
_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LAB_ROOT not in sys.path:
    sys.path.insert(0, _LAB_ROOT)


def _prompt_hash() -> str:
    """Short stable hash of the active prompt surface (shared helper)."""
    from provenance import prompt_hash

    return prompt_hash(_LAB_ROOT)

# Per-attribute "bias" metric: (true_class -> mistaken_as) whose rate we headline.
# For gender the user's concern is women predicted male, so ("female","male").
BIAS_PAIR = {
    "gender": ("female", "male"),
}


def _jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def _last_ledger(ledger: str, attribute: str, version: str) -> dict | None:
    if not os.path.exists(ledger):
        return None
    prev = None
    for r in _jsonl(ledger):
        if r.get("attribute") == attribute and r.get("version") != version:
            prev = r  # last one wins → the most recent OTHER version
    return prev


def _read_seed_hash(eval_dir: str) -> str | None:
    """Read the core/ir HEAD from SEED.md one level above eval/."""
    lab_root = os.path.dirname(eval_dir)
    seed_md = os.path.join(lab_root, "SEED.md")
    if not os.path.exists(seed_md):
        return None
    with open(seed_md) as f:
        for line in f:
            if "core/ir HEAD" in line and "`" in line:
                parts = line.split("`")
                if len(parts) >= 3:
                    candidate = parts[-2].strip()
                    if len(candidate) >= 7:
                        return candidate
    return None


def _warn_stale_seed(seed_hash: str | None, core_ir_path: str | None) -> None:
    """Warn to stderr if live core/ir HEAD != SEED.md hash."""
    if seed_hash is None:
        return
    ir_path = core_ir_path or os.environ.get("CORE_IR_PATH")
    if not ir_path or not os.path.isdir(os.path.join(ir_path, ".git")):
        return
    try:
        import subprocess
        result = subprocess.run(
            ["git", "-C", ir_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        live_head = result.stdout.strip()
        if live_head and live_head != seed_hash:
            import sys
            print(
                f"WARNING: core/ir HEAD ({live_head[:12]}) != SEED.md hash "
                f"({seed_hash[:12]}) — Δ may not be comparable (stale seed).",
                file=sys.stderr,
            )
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--golden", default=None, help="golden dir (defaults to golden/<attribute>)")
    ap.add_argument("--version", default="plr_v1.4_cot")
    ap.add_argument("--ledger", default=os.path.join(here, "ledger.jsonl"))
    ap.add_argument("--date", default=None)
    ap.add_argument("--core-ir", default=None, dest="core_ir",
                    help="path to core/ir repo (for stale-seed warning)")
    # Experiment-combination keys (P2-1). Optional/back-compat: default to the
    # historical (gemma + plr) combination; --dataset defaults to --golden.
    ap.add_argument("--model", default="gemma",
                    help="registry model name that produced predictions (default: gemma)")
    ap.add_argument("--pipeline", default="plr",
                    help="pipeline name (default: plr)")
    ap.add_argument("--dataset", default=None,
                    help="dataset path/name for the ledger (default: --golden path)")
    args = ap.parse_args()

    seed_hash = _read_seed_hash(here)
    _warn_stale_seed(seed_hash, args.core_ir)

    gdir = args.golden or os.path.join(here, "golden", args.attribute)
    preds = {r["obj_id"]: (r.get("pred") or "unknown") for r in _jsonl(os.path.join(gdir, "predictions.jsonl"))}
    labels = {r["obj_id"]: (r.get("label") or r.get("true") or "unknown") for r in _jsonl(os.path.join(gdir, "labels.jsonl"))}

    ids = [i for i in preds if i in labels]
    if not ids:
        raise SystemExit("No overlap between predictions.jsonl and labels.jsonl")

    classes = sorted({*(labels[i] for i in ids), *(preds[i] for i in ids)})
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for i in ids:
        confusion[labels[i]][preds[i]] += 1
        correct += labels[i] == preds[i]
    n = len(ids)
    acc = correct / n

    recall = {}
    for c in classes:
        cid = [i for i in ids if labels[i] == c]
        if cid:
            recall[c] = round(sum(preds[i] == c for i in cid) / len(cid), 4)

    bias = None
    if args.attribute in BIAS_PAIR:
        t_cls, as_cls = BIAS_PAIR[args.attribute]
        tid = [i for i in ids if labels[i] == t_cls]
        if tid:
            bias = {"pair": f"{t_cls}->{as_cls}",
                    "rate": round(sum(preds[i] == as_cls for i in tid) / len(tid), 4),
                    "count": f"{sum(preds[i] == as_cls for i in tid)}/{len(tid)}"}

    prev = _last_ledger(args.ledger, args.attribute, args.version)

    print(f"=== {args.attribute} eval: {args.version} (n={n}) ===")
    print(f"accuracy: {acc:.3f} ({correct}/{n})", end="")
    if prev:
        d = acc - prev["accuracy"]
        print(f"   Δ vs {prev['version']}: {d:+.3f} ({prev['accuracy']:.3f} → {acc:.3f})")
    else:
        print("   (no prior version to diff)")
    if bias:
        print(f"bias {bias['pair']}: {bias['rate']:.3f} ({bias['count']})", end="")
        if prev and prev.get("bias"):
            print(f"   Δ: {bias['rate'] - prev['bias']['rate']:+.3f}")
        else:
            print()
    print("recall: " + ", ".join(f"{k}={v}" for k, v in recall.items()))
    print("confusion (rows=true, cols=pred):")
    print("        " + "".join(f"{c:>10}" for c in classes))
    for t in classes:
        print(f"{t:>8}" + "".join(f"{confusion[t][p]:>10}" for p in classes))

    gemma_repo = os.environ.get("IR_GEMMA_REPO", "")
    record = {
        "attribute": args.attribute, "version": args.version,
        "date": args.date or datetime.now().isoformat(timespec="seconds"),
        "n": n, "accuracy": round(acc, 4), "recall": recall, "bias": bias,
        "confusion": {t: dict(confusion[t]) for t in classes},
        "seed_hash": seed_hash or "",
        "gemma_repo": gemma_repo,
        # ---- Experiment-combination keys (P2-1) ----
        "dataset": args.dataset or gdir,
        "model": args.model,
        "pipeline": args.pipeline,
        "prompt_hash": _prompt_hash(),
    }
    with open(args.ledger, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nledger += {args.ledger}")


if __name__ == "__main__":
    main()
