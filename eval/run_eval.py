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
    from evalkit.provenance import prompt_hash

    return prompt_hash(_LAB_ROOT)

def _signal_stats(
    vals: dict, eval_ids: list, is_correct: dict, threshold: float,
) -> dict | None:
    """Accuracy split by a per-crop signal (model margin or crop quality).

    plr_v1.5_cot replaced the unknown escape hatch with "commit + margin",
    so eval must verify the signal is informative: if low-margin /
    low-quality crops concentrate the errors, the signal is usable
    downstream; if high/low accuracy are equal, it is noise.
    Returns None when the signal is absent from predictions (old files,
    or attributes whose prompt emits no margin)."""
    ids = [i for i in eval_ids if vals.get(i) is not None]
    if not ids:
        return None

    def _acc(group):
        return round(sum(is_correct[i] for i in group) / len(group), 4) if group else None

    def _mean(group):
        return round(sum(vals[i] for i in group) / len(group), 4) if group else None

    hi = [i for i in ids if vals[i] >= threshold]
    lo = [i for i in ids if vals[i] < threshold]
    return {
        "threshold": threshold,
        "n": len(ids),
        "high": {"n": len(hi), "accuracy": _acc(hi)},
        "low": {"n": len(lo), "accuracy": _acc(lo)},
        "mean_correct": _mean([i for i in ids if is_correct[i]]),
        "mean_wrong": _mean([i for i in ids if not is_correct[i]]),
    }


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
    ap.add_argument("--margin-threshold", type=float, default=0.7,
                    dest="margin_threshold",
                    help="confidence split point for margin_stats (default: 0.7)")
    ap.add_argument("--quality-threshold", type=float, default=0.4,
                    dest="quality_threshold",
                    help="crop-quality split point for quality_stats (default: 0.4)")
    args = ap.parse_args()

    from evalkit.provenance import read_seed_hash, warn_stale_seed
    seed_hash = read_seed_hash(_LAB_ROOT)
    warn_stale_seed(_LAB_ROOT, seed_hash, args.core_ir)

    gdir = args.golden or os.path.join(here, "golden", args.attribute)
    pred_rows = {r["obj_id"]: r for r in _jsonl(os.path.join(gdir, "predictions.jsonl"))}
    preds = {i: (r.get("pred") or "unknown") for i, r in pred_rows.items()}
    labels = {r["obj_id"]: (r.get("label") or r.get("true") or "unknown") for r in _jsonl(os.path.join(gdir, "labels.jsonl"))}

    ids = [i for i in preds if i in labels]
    if not ids:
        raise SystemExit("No overlap between predictions.jsonl and labels.jsonl")

    # Forced-commit compliance (plr_v1.5_cot): how often the model still
    # answered unknown despite the commit instruction. Measured over ALL
    # matched ids, before any label filtering.
    n_pred_unknown = sum(1 for i in ids if preds[i] == "unknown")
    pred_unknown = {
        "rate": round(n_pred_unknown / len(ids), 4),
        "count": f"{n_pred_unknown}/{len(ids)}",
    }

    # Human-unlabelable crops (label == "unknown", set via `lab label
    # --unknown ...`) are EXCLUDED from accuracy/recall/bias/confusion: if a
    # human cannot decide the attribute from the crop, there is no ground
    # truth to be right or wrong against — under the forced-commit prompt the
    # model must still answer, and counting those answers as errors would
    # systematically punish committed guesses on undecidable crops.
    eval_ids = [i for i in ids if labels[i] != "unknown"]
    n_label_unknown = len(ids) - len(eval_ids)
    if not eval_ids:
        raise SystemExit(
            "All labels are 'unknown' — nothing to score against. "
            "Label at least one crop with a decided class."
        )

    classes = sorted({*(labels[i] for i in eval_ids), *(preds[i] for i in eval_ids)})
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for i in eval_ids:
        confusion[labels[i]][preds[i]] += 1
        correct += labels[i] == preds[i]
    n = len(eval_ids)
    acc = correct / n

    recall = {}
    for c in classes:
        cid = [i for i in eval_ids if labels[i] == c]
        if cid:
            recall[c] = round(sum(preds[i] == c for i in cid) / len(cid), 4)

    # Per-class precision + F1, and macro-F1 over classes that appear in the
    # ground truth (classes only ever predicted have recall undefined).
    precision = {}
    for c in classes:
        pid = [i for i in eval_ids if preds[i] == c]
        if pid:
            precision[c] = round(sum(labels[i] == c for i in pid) / len(pid), 4)
    f1 = {}
    for c in classes:
        r, pr = recall.get(c), precision.get(c)
        if r is None and pr is None:
            continue
        r, pr = r or 0.0, pr or 0.0
        f1[c] = round(2 * pr * r / (pr + r), 4) if (pr + r) > 0 else 0.0
    gt_classes = [c for c in classes if c in recall]
    macro_f1 = (round(sum(f1.get(c, 0.0) for c in gt_classes) / len(gt_classes), 4)
                if gt_classes else None)

    # Confidence / quality splits (values are optional per prediction row).
    is_correct = {i: labels[i] == preds[i] for i in eval_ids}
    margins = {i: pred_rows[i].get("margin") for i in eval_ids}
    qualities = {i: pred_rows[i].get("quality") for i in eval_ids}
    margin_stats = _signal_stats(margins, eval_ids, is_correct, args.margin_threshold)
    quality_stats = _signal_stats(qualities, eval_ids, is_correct, args.quality_threshold)

    # Headline bias pair: dataset manifest declaration wins over the preset.
    pair = None
    try:
        import yaml
        with open(os.path.join(gdir, "manifest.yaml"), encoding="utf-8") as f:
            _mani = yaml.safe_load(f) or {}
        bp = _mani.get("bias_pair")
        if isinstance(bp, (list, tuple)) and len(bp) == 2:
            pair = (str(bp[0]), str(bp[1]))
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 — malformed manifest is validate's job
        print(f"WARNING: manifest.yaml unreadable for bias_pair: {exc}", file=sys.stderr)
    if pair is None:
        pair = BIAS_PAIR.get(args.attribute)

    bias = None
    if pair:
        t_cls, as_cls = pair
        tid = [i for i in eval_ids if labels[i] == t_cls]
        if tid:
            bias = {"pair": f"{t_cls}->{as_cls}",
                    "rate": round(sum(preds[i] == as_cls for i in tid) / len(tid), 4),
                    "count": f"{sum(preds[i] == as_cls for i in tid)}/{len(tid)}"}

    prev = _last_ledger(args.ledger, args.attribute, args.version)

    print(f"=== {args.attribute} eval: {args.version} (n={n}) ===")
    if n_label_unknown:
        print(f"excluded {n_label_unknown} human-unlabelable crop(s) (label=unknown)")
    print(f"pred unknown rate: {pred_unknown['rate']:.3f} ({pred_unknown['count']})")
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
    for name, st in (("margin", margin_stats), ("quality", quality_stats)):
        if st:
            print(f"{name} split (>= {st['threshold']}): "
                  f"high acc={st['high']['accuracy']} (n={st['high']['n']})  "
                  f"low acc={st['low']['accuracy']} (n={st['low']['n']})  "
                  f"mean correct/wrong: {st['mean_correct']}/{st['mean_wrong']}")
    print("recall:    " + ", ".join(f"{k}={v}" for k, v in recall.items()))
    print("precision: " + ", ".join(f"{k}={v}" for k, v in precision.items()))
    print("f1:        " + ", ".join(f"{k}={v}" for k, v in f1.items())
          + (f"   macro_f1={macro_f1}" if macro_f1 is not None else ""))
    print("confusion (rows=true, cols=pred):")
    print("        " + "".join(f"{c:>10}" for c in classes))
    for t in classes:
        print(f"{t:>8}" + "".join(f"{confusion[t][p]:>10}" for p in classes))

    gemma_repo = os.environ.get("IR_GEMMA_REPO", "")
    record = {
        "attribute": args.attribute, "version": args.version,
        "date": args.date or datetime.now().isoformat(timespec="seconds"),
        "n": n, "accuracy": round(acc, 4), "recall": recall,
        "precision": precision, "f1": f1, "macro_f1": macro_f1, "bias": bias,
        "confusion": {t: dict(confusion[t]) for t in classes},
        # plr_v1.5_cot forced-commit metrics: model unknown rate (over all
        # matched ids) and how many crops were human-unlabelable (excluded).
        "pred_unknown": pred_unknown,
        "n_label_unknown": n_label_unknown,
        # Confidence/quality calibration (None when the signal is absent).
        "margin_stats": margin_stats,
        "quality_stats": quality_stats,
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
