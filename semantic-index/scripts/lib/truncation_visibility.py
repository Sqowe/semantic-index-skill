"""Shared helpers for surfacing chunk-truncation in the build summary.

Both ``build_index.py`` (CLI) and ``mcp_server.py`` (MCP transport)
need the same two pieces of logic:

1. Per-batch accumulation: given an ``Embedder`` whose
   ``truncation_stats`` is cumulative across calls within a build,
   update the running total and file set, and return the new total.

2. Final summary population: given the accumulated total and file
   set, return the three result-dict fields
   (``truncated_chunks``, ``truncated_files``, ``truncation_message``)
   plus the per-file DEBUG log records to emit on stderr.

Centralising both pieces means a fix in one place propagates to both
call sites — the duplication was flagged in the Phase 4 review as
fragile, especially given the delta-based approach to per-batch
accumulation.
"""

from __future__ import annotations

import logging
from typing import Iterable

from .embedder import Embedder
from .models import TruncationRecord

logger = logging.getLogger(__name__)


def accumulate_truncation(
    embedder: Embedder,
    total_truncated: int,
    truncated_files: set[str],
) -> int:
    """Update the running truncation total and file set from a batch.

    The ``Embedder`` keeps ``truncation_stats`` cumulative across
    ``embed_chunks`` calls within a single build (it is created fresh
    per build, so per-build scoping is implicit). The per-batch delta
    is the difference between the current length and the running
    total; the slice ``[-batch_truncated:]`` from that delta picks
    up exactly the records this batch produced.

    Args:
        embedder: The build's ``Embedder`` instance.
        total_truncated: Running total carried across batches.
        truncated_files: Running set of project-relative file paths
            that have contributed at least one truncated chunk.

    Returns:
        The new running total. ``truncated_files`` is updated in place
        (it is a set; the caller does not need to reassign).

    Notes:
        The ``batch_truncated > 0`` guard makes the call safe against
        a future ``Embedder`` refactor that resets or shrinks
        ``truncation_stats`` between calls. A negative or zero delta
        is a no-op: the running total is preserved and no file paths
        are added.
    """
    batch_truncated = len(embedder.truncation_stats) - total_truncated
    if batch_truncated > 0:
        total_truncated = len(embedder.truncation_stats)
        for record in embedder.truncation_stats[-batch_truncated:]:
            truncated_files.add(record.file_path)
    return total_truncated


def build_truncation_summary(
    total_truncated: int,
    truncated_files: Iterable[str],
) -> dict:
    """Return the truncation fields to merge into the build result.

    Three fields are always present:

    * ``truncated_chunks`` (int) — count of chunks shortened
    * ``truncated_files`` (list[str], sorted) — unique project-relative
      paths that contributed at least one truncated chunk

    A fourth field, ``truncation_message``, is added only when
    ``total_truncated > 0`` — it is a human-readable summary that
    names the count and the search-coverage consequence.

    Args:
        total_truncated: Running count of truncated chunks.
        truncated_files: Iterable of affected file paths. The caller
            may pass a set; the function sorts and returns a list.
    """
    summary = {
        "truncated_chunks": total_truncated,
        "truncated_files": sorted(truncated_files),
    }
    if total_truncated:
        summary["truncation_message"] = (
            f"{total_truncated} chunk(s) were shortened before "
            "embedding; their endings are not searchable"
        )
    return summary


def log_truncated_files(truncated_files: Iterable[str]) -> None:
    """Emit an INFO log line per affected file path.

    INFO is the right level here: the message is operationally useful
    (operators can see which files lost coverage in the build log)
    but not noisy (one line per file, only when truncations
    occurred). DEBUG would have been filtered by the root logger's
    default WARNING level because the build CLI configures
    ``basicConfig(level=INFO)`` — making the line invisible in
    production even though it is recorded.
    """
    for file_path in sorted(truncated_files):
        logger.info(
            "Truncated chunks in %s; search hits may miss content "
            "past the surviving prefix", file_path,
        )
