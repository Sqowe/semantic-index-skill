# semantic-index-skill

A portable SKILL for embedding-based semantic search of codebases and documentation. The
core deliverable is the `semantic-index/` directory — a self-contained package of CLI scripts,
a `SKILL.md` instruction file, and an optional MCP server bridge — installed once into an AI
tool's skills folder and pointed at any project via `--project-dir`.

> Status: **implemented and in use.** CLI scripts (build/search/status), the chunking pipeline
> (code, markdown, DITA, office), hybrid search (vector + BM25 + RRF), and the MCP server are
> built and covered by a pytest suite. [ARCHITECTURE.md](ARCHITECTURE.md) is the authoritative
> design source; [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) tracks the phased build.
>
> Build & test: `cd semantic-index/scripts && bash setup.sh`, `pytest semantic-index/tests/`.

## Read before making changes

1. [ARCHITECTURE.md](ARCHITECTURE.md) — system structure, components, stability zones.
2. The relevant `AI_*.md` file(s) for the code you are touching — coding rules (see below).
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — full implementation spec and phases.
4. [CHANGELOG.md](CHANGELOG.md) — what shipped in each version, and what an upgrade
   from an earlier one requires. Add an entry under `## [Unreleased]` for any change
   a user of the SKILL would notice.
5. `docs/chats/` — previous implementation context and decisions, before touching anything
   with prior history there.

## Coding rules live in `AI_*.md` (do not duplicate them here or in ARCHITECTURE.md)

| File | Scope |
| --- | --- |
| [AI.md](AI.md) | General Python conventions — PEP8, type hints, docstrings, error handling, `.env`/config loading, testing (applies repo-wide, superseded by AI_SKILL.md inside `semantic-index/`) |
| [AI_SKILL.md](AI_SKILL.md) | SKILL-specific conventions for `semantic-index/**` — CLI JSON stdout / logs stderr, exit codes, Tree-sitter grammar loading, LanceDB usage, JSON-config pattern, 300-line module limit |

`ARCHITECTURE.md` and the `AI_*.md` files must not redefine or duplicate each other's content.

## Working agreement

- **Confirm before acting.** Never create, edit, or delete files, run state-changing commands,
  or write to external systems without explicit user approval. First explain the situation,
  propose specifics (which files, what changes, what commands), then wait for a clear "yes."
  Read-only work (reading, searching, analyzing, answering) needs no confirmation.
  Exception: if the user says "just do it" / "go ahead," proceed directly.
- **Don't commit unprompted.** Run `git add` / `git commit` / `git push` only when the user
  explicitly asks — never as an unrequested side-effect of another task.
- **Never fake a passing index or search result.** The `.index/` artifacts (manifest, vector
  store, BM25 index) must reflect what was actually chunked and embedded — never hand-craft
  fixture data, skip the hasher's change detection, or swallow an embedding/API error just to
  make a build or test look green.
- **Stop and ask** if anything is unclear or contradictory.

## Tooling

- Python 3.10+, managed via a per-installation venv (`semantic-index/scripts/setup.sh` creates
  `scripts/.venv`); no Poetry/pipenv, plain `requirements*.txt` files split by optional feature
  (`--with-huggingface`, `--with-office`, `--with-mcp`).
- Quality gate: `pytest` (`semantic-index/tests/`, 10 test files). No linter/formatter is wired
  into CI yet — follow the PEP8 + type-hint conventions in [AI.md](AI.md) manually.
- Secrets: `OPENROUTER_API_KEY` and `SEMANTIC_INDEX_*` are read from environment variables only
  (see [AI_SKILL.md](AI_SKILL.md)); never commit an API key or write one into `.index/config.json`
  in a sample/test fixture.
- No Docker, no CI pipeline, no `.env` files are part of this project — the SKILL itself is
  file-based and dependency-light by design (see ARCHITECTURE.md §6).
