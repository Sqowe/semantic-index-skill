"""Cross-encoder reranker for search result refinement.

Uses sentence-transformers CrossEncoder to re-score search results
against the original query. This produces more accurate relevance
scores than bi-encoder similarity alone, at the cost of higher
latency (each query-document pair is scored independently).

Requires the HuggingFace optional dependencies (sentence-transformers).
Only activated when search.rerank_enabled is true in config.
"""

import logging
import sys
from typing import Any, Optional

from .models import EmbeddingError

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using sentence-transformers.

    Lazy-loads the CrossEncoder model on first use. The model is
    downloaded to ~/.cache/huggingface/hub on first run.

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
        device: Device to run on (None=auto, "cpu", "cuda", "mps").

    Raises:
        EmbeddingError: If sentence-transformers is not installed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._model = None  # Lazy-loaded

    def _ensure_model(self) -> None:
        """Load the cross-encoder model if not already loaded."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise EmbeddingError(
                "Reranker requires sentence-transformers. "
                "Install with: pip install -r requirements-huggingface.txt "
                "or run: bash setup.sh --with-huggingface"
            ) from exc

        print(
            f"Loading reranker model {self._model_name}...",
            file=sys.stderr,
        )
        try:
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                trust_remote_code=self._trust_remote_code,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load reranker model {self._model_name}: {exc}"
            ) from exc

        logger.info("Loaded reranker: %s", self._model_name)

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Re-rank search results using the cross-encoder.

        Scores each (query, document) pair and re-sorts by relevance.
        The original scores are preserved; a new 'rerank_score' field
        is added to each result.

        Args:
            query: The original search query.
            results: List of result dicts (must have 'content' field).
            top_n: Number of top results to return after reranking.

        Returns:
            Re-ranked results (up to top_n), sorted by rerank_score
            descending. Each result dict gets a 'rerank_score' field.
        """
        if not results:
            return []

        self._ensure_model()

        # Build query-document pairs
        pairs = [(query, r["content"]) for r in results]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored = []
        for result, score in zip(results, scores):
            entry = dict(result)
            entry["rerank_score"] = round(float(score), 4)
            scored.append(entry)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            "Reranked %d results → returning top %d",
            len(scored), min(top_n, len(scored)),
        )
        return scored[:top_n]
