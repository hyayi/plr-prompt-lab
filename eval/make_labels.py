#!/usr/bin/env python3
"""Turn a human's short 'these are wrong' input into labels.jsonl.

The reviewer looks at the contact sheets (gender_MALE.png = M1..M50 predicted
male, gender_FEMALE.png = F1..F13 predicted female) and only has to name the
MISCLASSIFIED tiles. Everything not named keeps the model's prediction as its
ground truth. index_map.json maps M#/F# -> obj_id.

Example (M3,M7 are actually women; F2 is actually a man; M40 unclear):
    python3 make_labels.py \
        --female-in-male M3,M7 \
        --male-in-female F2 \
        --unknown M40 \
        --out golden/gender/labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def _parse(csv: str | None) -> set[str]:
    if not csv:
        return set()
    return {tok.strip().upper() for tok in csv.split(",") if tok.strip()}


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-map", default=os.path.join(here, "golden/gender/index_map.json"))
    ap.add_argument("--pred", default=os.path.join(here, "golden/gender/predictions.jsonl"))
    ap.add_argument("--female-in-male", default="", help="M# tiles that are actually FEMALE")
    ap.add_argument("--male-in-female", default="", help="F# tiles that are actually MALE")
    ap.add_argument("--unknown", default="", help="tiles too ambiguous to call")
    ap.add_argument("--out", default=os.path.join(here, "golden/gender/labels.jsonl"))
    args = ap.parse_args()

    index_map = json.load(open(args.index_map))  # {"M1": obj_id, ...}
    pred = {r["obj_id"]: r.get("pred") for r in
            (json.loads(l) for l in open(args.pred) if l.strip())}

    female_in_male = _parse(args.female_in_male)
    male_in_female = _parse(args.male_in_female)
    unknown = _parse(args.unknown)

    known_tiles = set(index_map)
    for label_set, name in [(female_in_male, "female-in-male"),
                            (male_in_female, "male-in-female"),
                            (unknown, "unknown")]:
        bad = label_set - known_tiles
        if bad:
            raise SystemExit(f"Unknown tile ids in --{name}: {sorted(bad)}")

    lines = []
    for tile, obj_id in index_map.items():
        if tile in unknown:
            true = "unknown"
        elif tile in female_in_male:
            true = "female"
        elif tile in male_in_female:
            true = "male"
        else:
            true = pred.get(obj_id, "unknown")  # keep the model's call
        lines.append({"obj_id": obj_id, "label": true, "tile": tile})

    with open(args.out, "w") as f:
        for r in lines:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_flipped = len(female_in_male) + len(male_in_female)
    print(f"wrote {len(lines)} labels to {args.out} "
          f"({n_flipped} corrected, {len(unknown)} unknown)")


if __name__ == "__main__":
    main()
