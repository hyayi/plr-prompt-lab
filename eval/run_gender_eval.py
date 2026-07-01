#!/usr/bin/env python3
"""Gender golden-set eval (loop-engineering C2).

Joins the STORED model predictions (predictions.jsonl, exported from
ir_plr_index at index time) with human ground-truth labels (labels.jsonl) and
reports per-class accuracy + a confusion matrix. The headline metric is the
false-male rate (true=female predicted male) — the bias the user observed.

No Gemma call needed for the CURRENT prompt version: the predictions are already
in the DB snapshot. Re-running a *changed* prompt on the same crops (without a
full reindex) is a separate step that needs the GPU.

Usage:
    python3 run_gender_eval.py \
        --pred  golden/gender/predictions.jsonl \
        --labels golden/gender/labels.jsonl \
        --version plr_v1.4_cot \
        --ledger ledger.jsonl

labels.jsonl lines: {"obj_id": "1003", "true": "male|female|unknown"}
predictions.jsonl lines: {"obj_id": "1003", "pred": "male", "reason": "...", "margin": "0.9"}
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--pred", default=os.path.join(here, "golden/gender/predictions.jsonl"))
    ap.add_argument("--labels", default=os.path.join(here, "golden/gender/labels.jsonl"))
    ap.add_argument("--version", default="plr_v1.4_cot")
    ap.add_argument("--ledger", default=os.path.join(here, "ledger.jsonl"))
    ap.add_argument("--date", default=None, help="ISO date; defaults to now")
    args = ap.parse_args()

    preds = {r["obj_id"]: (r.get("pred") or "unknown") for r in _load_jsonl(args.pred)}
    labels = {r["obj_id"]: (r.get("true") or "unknown") for r in _load_jsonl(args.labels)}

    # Only score objects that have BOTH a prediction and a human label.
    ids = [i for i in preds if i in labels]
    missing = [i for i in labels if i not in preds]
    if not ids:
        raise SystemExit("No overlapping obj_ids between predictions and labels.")

    # Confusion matrix: confusion[true][pred] = count
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for i in ids:
        t, p = labels[i], preds[i]
        confusion[t][p] += 1
        if t == p:
            correct += 1

    n = len(ids)
    acc = correct / n
    # Headline bias metric: of the truly-female objects, how many were called male.
    female_ids = [i for i in ids if labels[i] == "female"]
    false_male = sum(1 for i in female_ids if preds[i] == "male")
    false_male_rate = (false_male / len(female_ids)) if female_ids else None
    male_ids = [i for i in ids if labels[i] == "male"]
    false_female = sum(1 for i in male_ids if preds[i] == "female")
    false_female_rate = (false_female / len(male_ids)) if male_ids else None

    # Per-class recall (of true X, how many predicted X)
    recall = {}
    for cls in ("male", "female", "unknown"):
        cls_ids = [i for i in ids if labels[i] == cls]
        if cls_ids:
            recall[cls] = sum(1 for i in cls_ids if preds[i] == cls) / len(cls_ids)

    print(f"=== Gender eval: {args.version} ===")
    print(f"scored n={n} (labels without a prediction: {len(missing)})")
    print(f"overall accuracy: {acc:.3f} ({correct}/{n})")
    print(f"false-MALE rate (여→남): {false_male}/{len(female_ids)}"
          f" = {false_male_rate:.3f}" if false_male_rate is not None else
          "false-MALE rate: (no female labels)")
    if false_female_rate is not None:
        print(f"false-FEMALE rate (남→여): {false_female}/{len(male_ids)} = {false_female_rate:.3f}")
    print(f"per-class recall: " + ", ".join(f"{k}={v:.3f}" for k, v in recall.items()))
    print("confusion (rows=true, cols=pred):")
    classes = ["male", "female", "unknown"]
    print("          " + "".join(f"{c:>9}" for c in classes))
    for t in classes:
        print(f"  {t:>7} " + "".join(f"{confusion[t][p]:>9}" for p in classes))

    record = {
        "version": args.version,
        "date": args.date or datetime.now().isoformat(timespec="seconds"),
        "attribute": "gender",
        "n": n,
        "accuracy": round(acc, 4),
        "false_male_rate": round(false_male_rate, 4) if false_male_rate is not None else None,
        "false_female_rate": round(false_female_rate, 4) if false_female_rate is not None else None,
        "recall": {k: round(v, 4) for k, v in recall.items()},
        "confusion": {t: dict(confusion[t]) for t in classes},
    }
    with open(args.ledger, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nappended to ledger: {args.ledger}")


if __name__ == "__main__":
    main()
