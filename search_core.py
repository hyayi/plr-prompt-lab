"""Pure search core — the storage-free hard-filter + attribute-rank step.

Extracted from `image_retrieval.ImageRetrieval.get_similarity` so the same
hard-filter + attribute-match ranking is reachable WITHOUT a database (the
lab / offline eval harness passes candidate rows in directly).

Depends ONLY on `scoring` (pure). It must NOT import storage / psycopg2 /
redis / gemma_backend.

`get_similarity` keeps its full live behavior (template-embedding final_score,
refine, threshold, top_k); it delegates only the reusable hard-filter predicate
to `candidate_passes_hard_filter` here so the SAME filtering logic is shared.
"""

from __future__ import annotations

from typing import Any

import scoring


def candidate_passes_hard_filter(
    query_required: dict[str, list[str]],
    candidate_plr: dict[str, Any],
    query_excluded: dict[str, list[str]] | None = None,
) -> bool:
    """Thin, storage-free wrapper over `scoring.passes_hard_filter`.

    This is the exact predicate the live search loop applies as its coarse
    cascade stage. Kept as a named shared helper so both the live path
    (`get_similarity`) and the offline path (`run_search`) route through one
    definition.
    """
    return scoring.passes_hard_filter(
        query_required, candidate_plr, query_excluded=query_excluded
    )


def run_search(query_json: Any, candidates: list[Any]) -> list[Any]:
    """Hard-filter + attribute-match rank over pre-fetched candidates.

    This is the v1 attribute-based scoring path — it does NOT include the
    template-embedding `final_score` term (that stays in `get_similarity` for
    the live path, where the query embeddings are available).

    Args:
      query_json: the parsed query object exposing `.target`
        (person|vehicle|mixed|event), `.required` (dict[slot -> values]) and
        `.excluded` (dict[slot -> values] | None).
      candidates: list of rows already fetched (NOT fetched here). Each row
        exposes `.plr_json` (dict) and `.object_type` (used to derive
        is_vehicle). A row is a vehicle when `row.object_type == "vehicle"`.

    Returns:
      The candidates that pass the hard filter, ranked by descending
      attribute-match score (ties preserve input order — Python's stable sort).
    """
    query_target = getattr(query_json, "target", None)
    required = getattr(query_json, "required", {}) or {}
    excluded = getattr(query_json, "excluded", None)

    ranked: list[tuple[float, Any]] = []
    for row in candidates:
        # Target filter — same rule as the live loop: person/vehicle queries
        # keep only matching object_type; mixed/event keep both.
        if query_target in {"person", "vehicle"} and row.object_type != query_target:
            continue

        is_vehicle = row.object_type == "vehicle"

        if not candidate_passes_hard_filter(required, row.plr_json, query_excluded=excluded):
            continue

        score, _contrib = scoring.attribute_match(
            required, excluded, row.plr_json, is_vehicle
        )
        ranked.append((score, row))

    ranked.sort(key=lambda t: -t[0])
    return [row for _score, row in ranked]
