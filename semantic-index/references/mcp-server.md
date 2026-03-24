# MCP Server Transport

The semantic-index skill can be used as an MCP (Model Context Protocol) server,
exposing the same indexing, search, and status functionality as the CLI scripts
through MCP tools. This is an alternative to the CLI-based SKILL interface.

## When to Use MCP vs CLI (SKILL)

- **SKILL (CLI)**: The primary interface. The AI reads `SKILL.md`, runs Python
  scripts, and parses JSON output. Works in Kiro and Claude Code out of the box.
- **MCP server**: For setups where you want the AI to call tools directly via
  MCP protocol instead of running shell commands. Useful when your IDE or agent
  framework natively supports MCP tool discovery.

Both interfaces use the same underlying code and produce identical results.

## Setup

Install MCP dependencies (in addition to core setup):

```bash
cd ~/.kiro/skills/semantic-index/scripts
bash setup.sh --with-mcp
```

This installs `mcp[cli]` into the skill's virtual environment.

If you also need office document support or local embeddings, combine flags:

```bash
bash setup.sh --with-mcp --with-office --with-huggingface
```

## Running the Server

### stdio transport (default)

```bash
~/.kiro/skills/semantic-index/scripts/.venv/bin/python \
  ~/.kiro/skills/semantic-index/scripts/mcp_server.py
```

### Streamable HTTP transport

```bash
~/.kiro/skills/semantic-index/scripts/.venv/bin/python \
  ~/.kiro/skills/semantic-index/scripts/mcp_server.py --transport http
```

## IDE Configuration

### Kiro

Add to `.kiro/settings/mcp.json` (workspace level) or `~/.kiro/settings/mcp.json`
(user level):

```json
{
  "mcpServers": {
    "semantic-index": {
      "command": "~/.kiro/skills/semantic-index/scripts/.venv/bin/python",
      "args": ["~/.kiro/skills/semantic-index/scripts/mcp_server.py"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-your-key-here"
      },
      "disabled": false,
      "autoApprove": [
        "semantic_index_status",
        "semantic_index_search"
      ]
    }
  }
}
```

### Claude Code

Add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "semantic-index": {
      "command": "~/.kiro/skills/semantic-index/scripts/.venv/bin/python",
      "args": ["~/.kiro/skills/semantic-index/scripts/mcp_server.py"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-your-key-here"
      }
    }
  }
}
```

## Available Tools

### semantic_index_build

Build or incrementally update the semantic index for a project.

Parameters:
- `project_dir` (string, required): Absolute path to the project root
- `config_path` (string, optional): Path to config.json (default: `<project_dir>/.index/config.json`)
- `full_reindex` (boolean, optional): Force full re-index, ignoring manifest (default: false)
- `batch_size` (integer, optional): Files per batch, 1–500 (default: 50)

### semantic_index_search

Search the semantic index by meaning using natural language queries.

Parameters:
- `project_dir` (string, required): Absolute path to the project root
- `query` (string, required): Natural language search query
- `top_k` (integer, optional): Max results, 1–100 (default: from config)
- `threshold` (float, optional): Min similarity 0.0–1.0 (default: from config)
- `filter_lang` (string, optional): Filter by language (e.g., "python")
- `filter_path` (string, optional): Filter by file path glob (e.g., "src/**")
- `mode` (string, optional): `vector`, `keyword`, or `hybrid` (default: from config)
- `alpha` (float, optional): Hybrid balance 0.0–1.0 (default: from config)
- `rerank` (boolean, optional): Enable cross-encoder reranking (default: from config)

### semantic_index_status

Check the health and statistics of the semantic index.

Parameters:
- `project_dir` (string, required): Absolute path to the project root

## Environment Variables

The MCP server respects the same environment variables as the CLI scripts:
- `OPENROUTER_API_KEY` — API key for OpenRouter embedding provider
- `SEMANTIC_INDEX_PROVIDER` — override embedding provider
- `SEMANTIC_INDEX_MODEL` — override embedding model
- `SEMANTIC_INDEX_DIMENSIONS` — override embedding dimensions

Set these in the `env` block of your MCP config, or export them in your shell
before starting the server.
