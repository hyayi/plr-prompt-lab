#!/usr/bin/env python3
"""Search evaluation against golden search set.

Deliverable 2 — run_search_eval.py

Reads:
  eval/golden/search/search_results.jsonl  {query, ranked:[obj_id]}
  eval/golden/search/queries.jsonl         {query, relevant:[obj_id]}

Computes recall@k and precision@k per query + mean, diffs vs the previous
ledger record for (attribute="search", version), and appends a record to
ledger.jsonl mirroring run_eval.py's style.

Ledger record fields:
  attribute, version, date, k,
  recall_at_k, precision_at_k, n_queries,
  seed_hash, gemma_repo

Usage:
    python3 run_search_eval.py --version plr_v1.4_cot
    python3 run_search_eval.py --version plr_v1.4_cot --k 10 --ledger path/to/ledger.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime


def _jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def _last_ledger(ledger_path: str, version: str) -> dict | None:
    """Return the most recent ledger record for attribute="search" with a
    DIFFERENT version (same semantics as run_eval.py's _last_ledger)."""
    if not os.path.exists(ledger_path):
        return None
    prev = None
    for r in _jsonl(ledger_path):
        if r.get("attribute") == "search" and r.get("version") != version:
            prev = r
    return prev


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Recall@k = |relevant ∩ top-k ranked| / |relevant|.

    0.0 when relevant is empty (avoids division-by-zero).
    """
    if not relevant:
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & relevant) / len(relevant)


def _precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Precision@k = |relevant ∩ top-k ranked| / k.

    0.0 when k == 0.
    """
    if k == 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for obj_id in top_k if obj_id in relevant)
    return hits / k


def compute_metrics(
    results_path: str,
    queries_path: str,
    k: int = 5,
) -> tuple[float, float, int]:
    """Compute mean recall@k and precision@k over all queries.

    Returns (mean_recall_at_k, mean_precision_at_k, n_queries).
    Warns to stderr for queries present in queries.jsonl but missing from
    search_results.jsonl (treated as empty ranked list).
    """
    queries = _jsonl(queries_path)          # [{query, relevant:[obj_id]}]
    results = _jsonl(results_path)          # [{query, ranked:[obj_id]}]

    # Index results by query text
    result_map: dict[str, list[str]] = {r["query"]: r["ranked"] for r in results}

    recalls: list[float] = []
    precisions: list[float] = []

    for q_entry in queries:
        query_text = q_entry["query"]
        relevant: set[str] = set(q_entry.get("relevant") or [])
        ranked: list[str] = result_map.get(query_text, [])
        if query_text not in result_map:
            print(
                f"WARNING: query {query_text!r} missing from search_results.jsonl — treating as empty",
                file=sys.stderr,
            )
        recalls.append(_recall_at_k(ranked, relevant, k))
        precisions.append(_precision_at_k(ranked, relevant, k))

    n = len(queries)
    mean_recall = sum(recalls) / n if n else 0.0
    mean_precision = sum(precisions) / n if n else 0.0
    return round(mean_recall, 4), round(mean_precision, 4), n


def main(
    *,
    results_path: str | None = None,
    queries_path: str | None = None,
    ledger_path: str | None = None,
    version: str = "plr_v1.4_cot",
    k: int = 5,
    date: str | None = None,
    seed_hash: str | None = None,
    gemma_repo: str | None = None,
    core_ir_path: str | None = None,
) -> dict:
    """Run search eval, print diff vs prior ledger, append ledger record.

    Can be called from lab.py (all kwargs) or as __main__ (argparse).
    Returns the appended ledger record.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    search_dir = os.path.join(here, "eval", "golden", "search")

    r_path = results_path or os.path.join(search_dir, "search_results.jsonl")
    q_path = queries_path or os.path.join(search_dir, "queries.jsonl")
    l_path = ledger_path or os.path.join(here, "eval", "ledger.jsonl")

    # ---- Resolve seed_hash from SEED.md if not provided ----
    if seed_hash is None:
        seed_hash = _read_seed_hash(here)

    # ---- Resolve gemma_repo from env if not provided ----
    if gemma_repo is None:
        gemma_repo = os.environ.get("IR_GEMMA_REPO", "")

    # ---- Warn if core/ir HEAD != SEED.md hash (stale seed) ----
    _warn_stale_seed(here, seed_hash, core_ir_path)

    mean_recall, mean_precision, n_queries = compute_metrics(r_path, q_path, k)

    prev = _last_ledger(l_path, version)

    print(f"=== search eval: {version} (n_queries={n_queries}, k={k}) ===")
    print(f"recall@{k}:    {mean_recall:.4f}", end="")
    if prev:
        dr = mean_recall - prev.get("recall_at_k", 0.0)
        print(f"   Δ vs {prev['version']}: {dr:+.4f} ({prev.get('recall_at_k', 0.0):.4f} → {mean_recall:.4f})")
    else:
        print("   (no prior version to diff)")

    print(f"precision@{k}: {mean_precision:.4f}", end="")
    if prev:
        dp = mean_precision - prev.get("precision_at_k", 0.0)
        print(f"   Δ vs {prev['version']}: {dp:+.4f} ({prev.get('precision_at_k', 0.0):.4f} → {mean_precision:.4f})")
    else:
        print("   (no prior version to diff)")

    record: dict = {
        "attribute": "search",
        "version": version,
        "date": date or datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "recall_at_k": mean_recall,
        "precision_at_k": mean_precision,
        "n_queries": n_queries,
        "seed_hash": seed_hash or "",
        "gemma_repo": gemma_repo or "",
    }

    with open(l_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nledger += {l_path}")

    return record


def _read_seed_hash(lab_root: str) -> str | None:
    """Read the core/ir HEAD from SEED.md (second bullet under ## Source)."""
    seed_md = os.path.join(lab_root, "SEED.md")
    if not os.path.exists(seed_md):
        return None
    with open(seed_md) as f:
        for line in f:
            # Matches: `- Source `core/ir HEAD`: `<hash>``
            if "core/ir HEAD" in line and "`" in line:
                # Extract last backtick-quoted token
                parts = line.split("`")
                # parts[-2] is the last backtick-enclosed value
                if len(parts) >= 3:
                    candidate = parts[-2].strip()
                    if len(candidate) >= 7:
                        return candidate
    return None


def _warn_stale_seed(lab_root: str, seed_hash: str | None, core_ir_path: str | None) -> None:
    """Print a stderr warning if the live core/ir HEAD != SEED.md hash."""
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
            print(
                f"WARNING: core/ir HEAD ({live_head[:12]}) != SEED.md hash "
                f"({seed_hash[:12]}) — Δ may not be comparable (stale seed).",
                file=sys.stderr,
            )
    except Exception:  # noqa: BLE001
        pass


def _argparse_main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    search_dir = os.path.join(here, "eval", "golden", "search")

    ap = argparse.ArgumentParser(
        description="Evaluate search results against golden queries (recall@k, precision@k)."
    )
    ap.add_argument("--version", default="plr_v1.4_cot", help="PLR version tag")
    ap.add_argument("--k", type=int, default=5, help="rank cutoff")
    ap.add_argument(
        "--results", default=os.path.join(search_dir, "search_results.jsonl"),
        help="search_results.jsonl path",
    )
    ap.add_argument(
        "--queries", default=os.path.join(search_dir, "queries.jsonl"),
        help="queries.jsonl path",
    )
    ap.add_argument(
        "--ledger", default=os.path.join(here, "eval", "ledger.jsonl"),
        help="ledger.jsonl path",
    )
    ap.add_argument("--date", default=None)
    ap.add_argument("--core-ir", default=None, dest="core_ir",
                    help="path to core/ir repo (for stale-seed check)")
    args = ap.parse_args()

    main(
        results_path=args.results,
        queries_path=args.queries,
        ledger_path=args.ledger,
        version=args.version,
        k=args.k,
        date=args.date,
        core_ir_path=args.core_ir,
    )


if __name__ == "__main__":
    _argparse_main()
