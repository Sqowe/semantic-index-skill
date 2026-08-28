# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions describe the SKILL itself. The files under `.index/` carry their own
independent format versions (`config.json` → `schema_version`,
`bm25_index.json` → `version`, `embedding_cache.db` → `version`); those are
noted below whenever one of them changes.

## [Unreleased]

## [0.2.0] — 2026-08-28

The first release after the initial working version. It adds office documents,
an MCP server, hybrid search fixes, and a substantial rework of how content is
divided into chunks and how many tokens each one is allowed to hold.

### Upgrading from 0.1.0

Three changes affect existing indexes. None is automatic.

1. **Run the config migration.** New file extensions and settings do not appear
   in an existing `.index/config.json` on their own:

   ```bash
   python semantic-index/scripts/migrate_config.py --project-dir <path>
   ```

2. **Rebuild the index in full.** Chunk boundaries changed — `.h` files are now
   parsed as C++, YAML and Helm templates have their own chunkers, and the
   last-resort splitter no longer alters the text it splits. Change detection
   compares file contents, not chunker behaviour, so an incremental run will
   leave already-indexed files divided the old way:

   ```bash
   python semantic-index/scripts/build_index.py --project-dir <path> --full
   ```

3. **The embedding cache moved to SQLite.** On the first build after upgrading,
   an existing `embedding_cache.json` is imported into `embedding_cache.db`
   automatically, so no embedding is paid for twice. The old JSON file is left
   in place and never read again — delete it whenever you are satisfied with
   the database. On one project it was 1.9 GB against 391 MB for the
   equivalent database.

### Added

- **Office documents** — PDF, DOCX and PPTX are indexed with format-aware
  chunking: pages, headings and slides respectively. Optional dependencies,
  installed with `setup.sh --with-office`.
- **MCP server** — `mcp_server.py` exposes build, search and status as MCP
  tools alongside the CLI scripts, with typed input validation.
- **C++ files following the `.cc`/`.hh` convention** — `.cc`, `.cxx`, `.hh`,
  `.hxx` and `.txx` are indexed. Only `.cpp` and `.hpp` were recognised before,
  so a codebase using the other convention had its entire C++ source skipped.
- **YAML chunker** — `.yaml` and `.yml` are divided along their own
  indentation: documents at `---`, then top-level keys, then recursively into
  any key still over budget. Nested chunks carry their key path
  (`spec.template.spec.containers[].env`) both as a comment and in metadata.
- **Helm template chunker** — `.tpl` files produce one chunk per
  `{{ define }}` block, named after the template it defines.
- **Truncated chunks in the build summary** — the build reports how many
  chunks had to be shortened to fit the model's context window, and which
  files they came from. Previously this happened silently.
- **`ARCHITECTURE.md`** — the authoritative description of the system's
  structure, components and stability zones.

### Changed

- **Chunk budgets use the embedding model's own tokenizer.** The chunker
  counted with tiktoken `cl100k_base` while the default model `BAAI/bge-m3`
  tokenizes differently — measured median ratio 1.30, worst case 2.13. When
  the model's tokenizer can be loaded, counts now match what the API will
  charge; when it cannot, budgets shrink by a safety factor instead.
- **The grammar for a source file is chosen by the file, not its extension.**
  A `.h` holds C in a C project and C++ in a C++ one. The declared grammar is
  tried first and an alternate only when that parse reports errors. On one C++
  codebase the C grammar failed on 98% of headers where the C++ grammar failed
  on 3%.
- **The embedding cache is a SQLite database**, `embedding_cache.db`, rather
  than one JSON document rewritten after every batch. On a 23,000-file project:
  startup 30 s → 0.00 s, per-batch save 30 s → 0.00 s, file 1.9 GB → 391 MB.
- **Rate limiting no longer consumes the retry budget.** An HTTP 429 is the
  server asking for a wait, not a failed request, and now has its own
  allowance. A build that previously died on three rate limits in a row rode
  through 44 of them.
- **Deletions happen in one pass.** Removing files from the index was one
  transaction per file in the vector store and a full sweep of the inverted
  index per file in the keyword index. Removing 1,672 files from a
  203,000-chunk index went from 31 minutes to 2.3 seconds.
- **Builds finish by compacting the vector store.** Version history used to
  accumulate across every build ever run — one project had 83,958 metadata
  files, and each build made the next one slower. Compaction brought that
  table from 3.1 GB in 31,947 versions to 1.9 GB in 2, with the row count
  unchanged.
- **Embedding cache format version raised to `2.0`** (SQLite).

### Fixed

- **The last-resort splitter altered the text it split.** Pieces were rebuilt
  with `decode(encode(x))`, which does not round-trip for a SentencePiece
  tokenizer: whitespace runs collapsed, newlines and tabs became single
  spaces, and a word boundary marked only by whitespace disappeared. Indexed
  chunks did not match their source files, and every piece reported a single
  line number. The split now cuts the original string at the character offsets
  the tokenizer reports.
- **Chunks could exceed `chunking.max_tokens`.** Both the code chunker and the
  blank-line fallback passed content through whole when it had no boundary to
  cut on — a minified line, a base64 blob, a file written without blank lines.
- **A class-like node could leave an over-budget chunk behind.** When no method
  was found, the "everything not covered by a method" fallback emitted the
  whole class unchecked. C++ nested namespaces hid the methods one level down,
  so this fired constantly: 63 oversized chunks per 200 `.cc` files, now none.
- **`--filter-path` was ignored in vector and hybrid search.** The filter was
  applied only in the keyword path, so it had no effect in vector mode and
  leaked unfiltered chunks into the fused results.
- **Oversized inputs no longer fail a whole build.** Batches are split
  automatically on a context-length error (HTTP 400, 413 and 422, and the same
  errors wrapped in a 200 response), and individual chunks over the model's
  window are shortened before they are sent.
- **Malformed API responses are handled** rather than raising an opaque error.
- **File paths in delete predicates are escaped.** A path containing a quote
  could previously change what the predicate meant — on a delete, that means
  rows disappearing that should not.

### Performance

- Batch splitting now isolates the offending chunk directly instead of halving
  blindly, which cost five extra requests to find one bad chunk in a batch
  of 32.
- Chunks known to be unembeddable are no longer sent to the API at all.

### Documentation

- **Exclusion path rules** — `.gitignore` syntax anchors any pattern
  containing a slash to the project root, which reads backwards from the
  intent: `jenkins/custom-files/` is narrower than `upgrade/`, not wider. On
  one project that silently left 34,030 chunks indexed that were meant to be
  excluded. README now states the rule with worked examples.
- `README.md`, `SKILL.md` and `references/supported-languages.md` updated for
  office documents, hybrid search, reranking, the MCP server, and the new
  chunkers.
- `CLAUDE.md` expanded with the working agreement and the split between
  `ARCHITECTURE.md` and the `AI_*.md` coding rules.
- Prior implementation conversations collected under `docs/chats/`.

## [0.1.0] — 2026-03-22

First working version: CLI scripts for build, search and status; the chunking
pipeline for code, markdown and DITA; hybrid search combining vector
similarity and BM25 with reciprocal rank fusion; and a pytest suite.
