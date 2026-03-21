"""Reciprocal Rank Fusion (RRF) for merging ranked result lists.

Combines BM25 keyword results and vector similarity results into a
single ranked list using the RRF formula:
    score(d) = Σ 1 / (k + rank_i(d))
where k is a constant (default 60) and rank_i is the 1-based rank
of document d in result list i.

Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _doc_key(result: dict[str, Any]) -> str:
    """Extract a stable document key from a result dict.

    When ``id`` is absent, builds a composite key from file_path,
    start_line, end_line, and chunk_type to avoid collisions between
    distinct chunks that share the same start_line.
    """
    doc_id = result.get("id", "")
    if doc_id:
        return doc_id
    # Fallback key: include end_line and chunk_type to avoid collisions
    return (
        f"{result['file_path']}:{result['start_line']}"
        f":{result.get('end_line', 0)}:{result.get('chunk_type', '')}"
    )


def fuse_results(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    alpha: float = 0.7,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Merge vector and BM25 results using weighted Reciprocal Rank Fusion.

    Args:
        vector_results: Ranked results from vector similarity search.
        bm25_results: Ranked results from BM25 keyword search.
        alpha: Weight for vector results (0.0-1.0). 1.0 = pure vector,
            0.0 = pure keyword. Default 0.7.
        k: RRF constant controlling how much rank position matters.
            Higher k = less penalty for lower ranks. Default 60.

    Returns:
        Merged, deduplicated results sorted by fused score (descending).
        Each result dict includes:
        - "fused_score": the combined RRF score (use for hybrid thresholding)
        - "vector_score": original vector similarity score (or None)
        - "bm25_score": original BM25 score (or None)
    """
    rrf_scores: dict[str, float] = {}
    vector_docs: dict[str, dict[str, Any]] = {}
    bm25_docs: dict[str, dict[str, Any]] = {}
    vector_orig_scores: dict[str, float] = {}
    bm25_orig_scores: dict[str, float] = {}

    # Score vector results
    for rank_idx, result in enumerate(vector_results):
        doc_id = _doc_key(result)
        rrf_score = alpha / (k + rank_idx + 1)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score
        vector_docs[doc_id] = result
        vector_orig_scores[doc_id] = result.get("score", 0.0)

    # Score BM25 results
    for rank_idx, result in enumerate(bm25_results):
        doc_id = _doc_key(result)
        rrf_score = (1.0 - alpha) / (k + rank_idx + 1)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score
        bm25_docs[doc_id] = result
        bm25_orig_scores[doc_id] = result.get("score", 0.0)

    # Sort by fused score
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results: list[dict[str, Any]] = []
    for doc_id, fused_score in ranked:
        # For duplicates: always prefer vector payload — it comes from the
        # store with richer metadata. When both sources have the doc,
        # vector wins deterministically (no cross-modal score comparison).
        v_score = vector_orig_scores.get(doc_id)
        b_score = bm25_orig_scores.get(doc_id)

        if v_score is not None:
            doc = vector_docs[doc_id]
        else:
            doc = bm25_docs[doc_id]

        merged = dict(doc)
        merged["fused_score"] = round(fused_score, 6)
        merged["vector_score"] = round(v_score, 4) if v_score is not None else None
        merged["bm25_score"] = round(b_score, 4) if b_score is not None else None
        results.append(merged)

    logger.debug(
        "RRF fusion: %d vector + %d bm25 -> %d merged results",
        len(vector_results), len(bm25_results), len(results),
    )
    return results
