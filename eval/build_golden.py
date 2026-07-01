#!/usr/bin/env python3
"""Build a golden eval set for one attribute from an indexed video.

Generalises the manual gender-set procedure: pull the model's predictions for an
attribute out of ir_plr_index, stratified-sample crops (all of the rare classes +
a cap of the common ones so both error directions are visible), copy the crops to
a browsable review dir, render numbered contact sheets, and emit predictions.jsonl
+ index_map.json + labels_template.csv. The human then names the misclassified
tiles (see make_labels.py) and run_eval.py scores it.

DB access mirrors the repo convention: `docker exec <db> psql`. Override via env
(IR_DB_CONTAINER, IR_DB_USER, IR_DB_PASS, IR_DB_NAME, RESULT_PATH).

    python3 build_golden.py --video vd_..._9pxm --attribute gender --per-class 50
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

ATTR = {
    "gender": dict(where="object_type='person'",
                   pred="coalesce(plr_json->'attributes'->'gender_scores'->>'selected','unknown')",
                   reason="coalesce(plr_json->'attributes'->'gender_scores'->>'reason','')"),
    "vehicle_type": dict(where="object_type='vehicle'",
                         pred="coalesce(plr_json->'attributes'->'type_topk'->0->>'label','unknown')",
                         reason="''"),
    "military": dict(where="true",
                     pred="coalesce(plr_json->'attributes'->>'military','unknown')",
                     reason="''"),
}


def _psql(sql: str) -> str:
    cont = os.environ.get("IR_DB_CONTAINER", "ziosummary-database")
    user = os.environ.get("IR_DB_USER", "ziovision")
    pw = os.environ.get("IR_DB_PASS", "CHANGE_ME_DB_PASS")
    db = os.environ.get("IR_DB_NAME", "ziosummary_management")
    out = subprocess.run(
        ["docker", "exec", "-e", f"PGPASSWORD={pw}", cont,
         "psql", "-U", user, "-d", db, "-tA", "-c", sql],
        capture_output=True, text=True, check=True)
    return out.stdout


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--attribute", required=True, choices=list(ATTR))
    ap.add_argument("--per-class", type=int, default=50, help="cap per predicted class")
    ap.add_argument("--out", default=None)
    ap.add_argument("--review-dir", default=None, help="browsable crop dir (default ~/<attr>_eval)")
    args = ap.parse_args()

    cfg = ATTR[args.attribute]
    gdir = args.out or os.path.join(here, "golden", args.attribute)
    review = args.review_dir or os.path.join(os.path.expanduser("~"), f"{args.attribute}_eval")
    crops = os.path.join(os.environ.get("RESULT_PATH", "./results"),
                         args.video, "objects")
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(review, exist_ok=True)

    # pull (obj_id, pred, reason) rows
    rows_raw = _psql(
        f"SELECT obj_id||'\t'||{cfg['pred']}||'\t'||{cfg['reason']} "
        f"FROM ir_plr_index WHERE video_id='{args.video}' AND {cfg['where']};")
    rows = []
    for line in rows_raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append({"obj_id": parts[0], "pred": parts[1] or "unknown",
                     "reason": parts[2] if len(parts) > 2 else ""})

    # stratified: cap each predicted class
    by_class: dict[str, list] = {}
    for r in sorted(rows, key=lambda x: x["obj_id"]):
        by_class.setdefault(r["pred"], []).append(r)
    sample = []
    for cls, items in by_class.items():
        sample.extend(items[: args.per_class])

    # export predictions + index map + labels template + copy crops
    with open(os.path.join(gdir, "predictions.jsonl"), "w") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # tiles per class: C1.. where C = class initial(s)
    index_map, per_class_counter = {}, {}
    for r in sorted(sample, key=lambda x: x["pred"]):
        pref = "".join(w[0] for w in r["pred"].split("_"))[:2].upper() or "X"
        per_class_counter[pref] = per_class_counter.get(pref, 0) + 1
        tile = f"{pref}{per_class_counter[pref]}"
        index_map[tile] = r["obj_id"]
        src = os.path.join(crops, r["obj_id"] + ".jpg")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(review, f"{tile}__{r['pred']}__{r['obj_id']}.jpg"))
    json.dump(index_map, open(os.path.join(gdir, "index_map.json"), "w"))

    with open(os.path.join(gdir, "labels_template.csv"), "w") as f:
        f.write("tile,obj_id,pred,true(=fill me)\n")
        inv = {v: k for k, v in index_map.items()}
        for r in sample:
            f.write(f"{inv.get(r['obj_id'],'')},{r['obj_id']},{r['pred']},\n")

    _contact_sheets(sample, index_map, crops, args.attribute)

    dist = {k: len(v) for k, v in by_class.items()}
    print(f"golden[{args.attribute}] for {args.video}: sampled {len(sample)} "
          f"(pred dist {dist})")
    print(f"  crops -> {review}")
    print(f"  meta  -> {gdir} (predictions.jsonl, index_map.json, labels_template.csv)")
    print(f"  next: label the misclassified tiles, then run_eval.py --attribute {args.attribute}")


def _contact_sheets(sample, index_map, crops, attribute) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("  (PIL unavailable — skipping contact sheets)")
        return
    inv = {v: k for k, v in index_map.items()}
    by_class: dict[str, list] = {}
    for r in sample:
        by_class.setdefault(r["pred"], []).append(r)
    CW, CH, COLS, LBL = 150, 200, 7, 22
    home = os.path.expanduser("~")
    for cls, items in by_class.items():
        rows_n = (len(items) + COLS - 1) // COLS
        cv = Image.new("RGB", (COLS * CW, rows_n * (CH + LBL) + 34), (25, 25, 25))
        d = ImageDraw.Draw(cv)
        d.text((6, 8), f"{attribute}={cls} ({len(items)})", fill=(255, 255, 0))
        for i, r in enumerate(items):
            x, y = (i % COLS) * CW, 34 + (i // COLS) * (CH + LBL)
            p = os.path.join(crops, r["obj_id"] + ".jpg")
            try:
                im = Image.open(p).convert("RGB"); im.thumbnail((CW - 6, CH - 6)); cv.paste(im, (x + 3, y + 3))
            except Exception:
                d.rectangle([x + 3, y + 3, x + CW - 3, y + CH - 3], outline=(120, 0, 0))
            d.text((x + 4, y + CH), inv.get(r["obj_id"], ""), fill=(0, 255, 255))
        out = os.path.join(home, f"{attribute}_{cls}.png")
        cv.save(out); print(f"  sheet -> {out}")


if __name__ == "__main__":
    main()
