#!/usr/bin/env python3
"""골든셋 채점기 CLI — 채점 코어는 evalkit/scoring.py::score() (단일 원천).

이 파일은 score()의 **CLI 래퍼**다: argparse → score() 호출 → stdout 리포트 +
provenance(prompt_hash/seed_hash/gemma_repo) 스탬프 + ledger append.
채점 로직(예측 소스 해석·지표 계산)은 전부 scoring.py에 산다 — 평가 서버도
같은 score()를 호출하므로 lab과 서버의 지표가 갈릴 수 없다.

산출 지표: accuracy · 클래스별 recall/precision/F1(+macro) · confusion ·
bias(예: female→male 오분류율) · pred_unknown(강제커밋 준수도) ·
margin/quality split(오답이 저신뢰/저품질에 몰리는가 = 캘리브레이션).
라벨 정책: label=="unknown"(사람도 판별 불가)은 채점에서 제외하고
n_label_unknown으로 별도 보고. 이전 버전과의 Δ를 출력하고 ledger에 append.

Usage:
    python3 run_eval.py --attribute gender \
        --golden golden/gender --version plr_v1.4_cot

predictions.jsonl: {"obj_id": "1003", "attribute": "gender", "pred": "male", ...}
labels.jsonl:      {"obj_id": "1003", "label": "female"}  또는 다속성 {"labels": {...}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# Lab root (one level above eval/) must be importable for the shared helpers
# whether this runs standalone or loaded via importlib from lab.py.
_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LAB_ROOT not in sys.path:
    sys.path.insert(0, _LAB_ROOT)

# Backward-compat re-exports — the scoring core moved to evalkit/scoring.py.
from evalkit.scoring import (  # noqa: E402,F401
    BIAS_PAIR,
    ScoringError,
    score,
    signal_stats as _signal_stats,
)


def _prompt_hash() -> str:
    """Short stable hash of the active prompt surface (shared helper)."""
    from evalkit.provenance import prompt_hash

    return prompt_hash(_LAB_ROOT)


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
    ap.add_argument("--model", default=None,
                    help="registry model name that produced predictions "
                         "(default: the model stamp in predictions.jsonl)")
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

    # ---- 채점 (순수 코어 — 서버와 동일 함수) ----
    try:
        res = score(
            gdir, args.attribute,
            margin_threshold=args.margin_threshold,
            quality_threshold=args.quality_threshold,
        )
    except ScoringError as exc:
        raise SystemExit(str(exc))

    # 모델 이름: 명시 플래그 > 예측 행 스탬프 > "unspecified".
    resolved_model = args.model or res["resolved_model"] or "unspecified"

    n = res["n"]
    acc = res["accuracy"]
    bias = res["bias"]
    pred_unknown = res["pred_unknown"]
    margin_stats, quality_stats = res["margin_stats"], res["quality_stats"]
    classes, confusion = res["classes"], res["confusion"]

    prev = _last_ledger(args.ledger, args.attribute, args.version)

    print(f"=== {args.attribute} eval: {args.version} (n={n}) ===")
    if res["n_label_unknown"]:
        print(f"excluded {res['n_label_unknown']} human-unlabelable crop(s) (label=unknown)")
    print(f"pred unknown rate: {pred_unknown['rate']:.3f} ({pred_unknown['count']})")
    print(f"accuracy: {acc:.3f} ({res['correct']}/{n})", end="")
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
    print("recall:    " + ", ".join(f"{k}={v}" for k, v in res["recall"].items()))
    print("precision: " + ", ".join(f"{k}={v}" for k, v in res["precision"].items()))
    print("f1:        " + ", ".join(f"{k}={v}" for k, v in res["f1"].items())
          + (f"   macro_f1={res['macro_f1']}" if res["macro_f1"] is not None else ""))
    print("confusion (rows=true, cols=pred):")
    print("        " + "".join(f"{c:>10}" for c in classes))
    for t in classes:
        print(f"{t:>8}" + "".join(f"{confusion.get(t, {}).get(p, 0):>10}" for p in classes))

    # ---- provenance + ledger (CLI 래퍼의 소관 — score()는 모름) ----
    gemma_repo = os.environ.get("IR_GEMMA_REPO", "")
    record = {
        "attribute": args.attribute, "version": args.version,
        "date": args.date or datetime.now().isoformat(timespec="seconds"),
        "n": n, "accuracy": acc, "recall": res["recall"],
        "precision": res["precision"], "f1": res["f1"],
        "macro_f1": res["macro_f1"], "bias": bias,
        "confusion": confusion,
        "pred_unknown": pred_unknown,
        "n_label_unknown": res["n_label_unknown"],
        "margin_stats": margin_stats,
        "quality_stats": quality_stats,
        "seed_hash": seed_hash or "",
        "gemma_repo": gemma_repo,
        "dataset": args.dataset or gdir,
        "model": resolved_model,
        "pipeline": args.pipeline,
        "prompt_hash": _prompt_hash(),
    }
    with open(args.ledger, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nledger += {args.ledger}")


if __name__ == "__main__":
    main()
