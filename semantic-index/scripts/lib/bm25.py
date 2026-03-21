"""BM25 keyword index for hybrid search.

Implements Okapi BM25 scoring with a JSON-persisted inverted index.
Built alongside the vector index during build_index.py and queried
during semantic_search.py for hybrid (BM25 + vector) search.

No external dependencies — pure Python implementation.
"""

import json
import logging
import math
import os
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional

from .config import Config, INDEX_DIR_NAME

logger = logging.getLogger(__name__)

BM25_INDEX_FILENAME = "bm25_index.json"

# Simple tokenizer: split on non-alphanumeric, lowercase, drop short tokens
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

# Common stop words to filter out (keeps index lean)
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "not", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "only", "own", "same", "so", "than",
    "too", "very", "just", "if", "it", "its", "this", "that", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
})


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens.

    Also splits camelCase and snake_case identifiers into sub-tokens.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercase tokens with stop words removed.
    """
    raw_tokens = _TOKEN_RE.findall(text)
    tokens: list[str] = []
    for raw in raw_tokens:
        # Split camelCase: "getUserName" -> ["get", "User", "Name"]
        parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw).split()
        for part in parts:
            # Split snake_case (already split by _TOKEN_RE on underscores?
            # No — _TOKEN_RE includes _, so "get_user" is one token)
            for sub in part.split("_"):
                lower = sub.lower()
                if len(lower) >= 2 and lower not in _STOP_WORDS:
                    tokens.append(lower)
    return tokens


class BM25Index:
    """Okapi BM25 inverted index with JSON persistence.

    Stores term frequencies and document metadata for keyword-based
    retrieval. Designed to complement the vector store for hybrid search.

    Attributes:
        k1: Term frequency saturation parameter (default 1.5).
        b: Document length normalization parameter (default 0.75).
    """

    def __init__(
        self,
        project_dir: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._project_dir = project_dir
        self._path = Path(project_dir) / INDEX_DIR_NAME / BM25_INDEX_FILENAME
        self.k1 = k1
        self.b = b

        # Inverted index: term -> {doc_id: term_frequency}
        self._postings: dict[str, dict[str, int]] = {}
        # Document metadata: doc_id -> {file_path, start_line, end_line, ...}
        self._docs: dict[str, dict[str, Any]] = {}
        # Document lengths (token count per doc)
        self._doc_lengths: dict[str, int] = {}
        # Average document length
        self._avg_dl: float = 0.0
        # Total number of documents
        self._n_docs: int = 0

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Build the BM25 index from a list of chunk dicts.

        Each chunk dict must have at minimum: id, content, file_path,
        start_line, end_line, chunk_type, language, symbol_name.

        Args:
            chunks: List of chunk metadata dicts (same shape as store records).
        """
        self._postings.clear()
        self._docs.clear()
        self._doc_lengths.clear()

        for chunk in chunks:
            doc_id = chunk["id"]
            tokens = tokenize(chunk["content"])
            self._doc_lengths[doc_id] = len(tokens)

            # Store document metadata (everything except content to save space)
            self._docs[doc_id] = {
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "chunk_type": chunk["chunk_type"],
                "language": chunk.get("language", ""),
                "symbol_name": chunk.get("symbol_name", ""),
                "token_count": chunk.get("token_count", 0),
                "content": chunk["content"],
            }

            # Build term frequencies
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            for term, freq in tf.items():
                if term not in self._postings:
                    self._postings[term] = {}
                self._postings[term][doc_id] = freq

        self._n_docs = len(self._docs)
        total_length = sum(self._doc_lengths.values())
        self._avg_dl = total_length / self._n_docs if self._n_docs > 0 else 0.0

        logger.info(
            "Built BM25 index: %d docs, %d unique terms, avg doc length %.1f",
            self._n_docs, len(self._postings), self._avg_dl,
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[dict[str, Optional[str]]] = None,
    ) -> list[dict[str, Any]]:
        """Search the BM25 index with a natural language query.

        Args:
            query: Search query string.
            top_k: Maximum number of results.
            filters: Optional filters (language, file_path_glob).

        Returns:
            List of result dicts with score, file_path, content, etc.
            Sorted by descending BM25 score.
        """
        if self._n_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Score each document
        scores: dict[str, float] = {}

        for term in query_tokens:
            if term not in self._postings:
                continue

            postings = self._postings[term]
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = len(postings)
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in postings.items():
                dl = self._doc_lengths.get(doc_id, 0)
                # BM25 TF component
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                score = idf * numerator / denominator

                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Apply filters and build results
        results: list[dict[str, Any]] = []
        for doc_id, score in ranked:
            if len(results) >= top_k:
                break

            doc = self._docs.get(doc_id)
            if doc is None:
                continue

            # Apply language filter
            if filters:
                lang_filter = filters.get("language")
                if lang_filter and doc.get("language") != lang_filter:
                    continue

                # Apply file path glob filter
                path_glob = filters.get("file_path_glob")
                if path_glob and not fnmatch(doc["file_path"], path_glob):
                    continue

            results.append({
                "score": round(score, 4),
                "file_path": doc["file_path"],
                "start_line": doc["start_line"],
                "end_line": doc["end_line"],
                "content": doc["content"],
                "chunk_type": doc["chunk_type"],
                "language": doc.get("language", ""),
                "symbol_name": doc.get("symbol_name", ""),
                "token_count": doc.get("token_count", 0),
                "id": doc_id,
            })

        return results

    def delete_by_file(self, file_path: str) -> None:
        """Remove all documents belonging to a file from the index.

        Args:
            file_path: Relative file path to remove.
        """
        doc_ids_to_remove = [
            doc_id for doc_id, doc in self._docs.items()
            if doc["file_path"] == file_path
        ]
        for doc_id in doc_ids_to_remove:
            del self._docs[doc_id]
            del self._doc_lengths[doc_id]

        # Clean postings
        empty_terms: list[str] = []
        for term, postings in self._postings.items():
            for doc_id in doc_ids_to_remove:
                postings.pop(doc_id, None)
            if not postings:
                empty_terms.append(term)
        for term in empty_terms:
            del self._postings[term]

        # Recalculate stats
        self._n_docs = len(self._docs)
        total_length = sum(self._doc_lengths.values())
        self._avg_dl = total_length / self._n_docs if self._n_docs > 0 else 0.0

    def _purge_doc_postings(self, doc_id: str) -> None:
        """Remove all postings for a single doc_id from the inverted index.

        Args:
            doc_id: Document ID whose term entries should be removed.
        """
        empty_terms: list[str] = []
        for term, postings in self._postings.items():
            if doc_id in postings:
                del postings[doc_id]
                if not postings:
                    empty_terms.append(term)
        for term in empty_terms:
            del self._postings[term]

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Incrementally add chunks to the existing index.

        If a chunk with the same doc_id already exists, its old postings
        are purged before re-indexing to prevent stale term entries.

        Args:
            chunks: List of chunk metadata dicts.
        """
        for chunk in chunks:
            doc_id = chunk["id"]

            # Purge stale postings if this doc_id is being re-added
            if doc_id in self._docs:
                self._purge_doc_postings(doc_id)

            tokens = tokenize(chunk["content"])
            self._doc_lengths[doc_id] = len(tokens)

            self._docs[doc_id] = {
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "chunk_type": chunk["chunk_type"],
                "language": chunk.get("language", ""),
                "symbol_name": chunk.get("symbol_name", ""),
                "token_count": chunk.get("token_count", 0),
                "content": chunk["content"],
            }

            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            for term, freq in tf.items():
                if term not in self._postings:
                    self._postings[term] = {}
                self._postings[term][doc_id] = freq

        # Recalculate stats
        self._n_docs = len(self._docs)
        total_length = sum(self._doc_lengths.values())
        self._avg_dl = total_length / self._n_docs if self._n_docs > 0 else 0.0

    def save(self) -> None:
        """Persist the BM25 index to disk as JSON."""
        data = {
            "version": "1.0",
            "n_docs": self._n_docs,
            "avg_dl": self._avg_dl,
            "postings": self._postings,
            "docs": self._docs,
            "doc_lengths": self._doc_lengths,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        logger.info("Saved BM25 index: %d docs, %d terms", self._n_docs, len(self._postings))

    def load(self) -> bool:
        """Load the BM25 index from disk.

        Returns:
            True if loaded successfully, False if no index exists.
        """
        if not self._path.exists():
            return False

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("BM25 index unreadable, will rebuild: %s", exc)
            return False

        self._postings = raw.get("postings", {})
        self._docs = raw.get("docs", {})
        self._doc_lengths = raw.get("doc_lengths", {})
        self._n_docs = raw.get("n_docs", len(self._docs))
        self._avg_dl = raw.get("avg_dl", 0.0)

        logger.info("Loaded BM25 index: %d docs, %d terms", self._n_docs, len(self._postings))
        return True

    def has_index(self) -> bool:
        """Check if a BM25 index file exists on disk."""
        return self._path.exists()
