# <img src="logo.png" width="120" alt="MCP Raven"> MCP Raven

MCP memory server that uses LLM agents to remember and recall context for you. Built on [mcp-agent](https://github.com/lastmile-ai/mcp-agent).

Inspired by [MemPalace](https://github.com/jasonjmcghee/mempalace) — named after the ravens that carry messages between castles.

> I built this for myself and use it every day. Figured I'd open source it in case anyone finds it useful. PRs welcome.

## Philosophy

Embeddings, vector databases, and RAG pipelines make sense at scale. But for personal memory — a few hundred facts and notes — they're more infrastructure than the problem needs. mcp-raven makes a different trade-off: **just put it all in the context window.**

GPT-4o has 128k tokens. Claude and Gemini both offer 1M. A typical personal memory store fits in a fraction of that. Context windows are growing faster than your memory collection will — so instead of building retrieval infrastructure, we lean on what LLMs already do well: reading and reasoning over text.

- **Your files are the database.** Facts live in a YAML file. Memories live in markdown. No opaque storage — everything is inspectable and version-controllable.
- **Context stuffing over embeddings.** Load everything into the LLM context and let it do the semantic matching natively. Zero retrieval errors, zero false negatives, zero infrastructure.
- **Prune aggressively, don't scale storage.** If context gets too large, the answer is cleaning up stale data — not building smarter retrieval. That's what temporal validity is for.
- **A cheap LLM does the thinking.** Categorization, deduplication, and synthesis are delegated to a fast, inexpensive model. Works with any OpenAI-compatible provider.

## How it works

Two MCP tools exposed via [mcp-agent](https://github.com/lastmile-ai/mcp-agent):

### `remember(context, source?)`

An LLM agent categorizes the input and writes it to the right place:

| Category | What | Storage |
|----------|------|---------|
| fact | Queryable facts — job, tools, hardware, preferences | `facts.yaml` (KG-style, temporal validity) |
| feedback | Behavioral preferences and corrections | `memory_dir/feedback_*.md` |
| project | Ongoing work, goals, decisions | `memory_dir/project_*.md` |
| reference | Pointers to external resources | `memory_dir/reference_*.md` |

Multi-topic inputs are split into multiple entries automatically. Ephemeral context is discarded.

### `recall(query)`

Loads all stored memories into the LLM context and asks it to synthesize an answer. Returns a response with sources and confidence level.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An API key for any OpenAI-compatible provider (OpenAI, [OpenRouter](https://openrouter.ai/), Ollama, LM Studio, etc.)

### Install

```bash
git clone https://github.com/arthurcgc/mcp-raven.git
cd mcp-raven
uv sync
```

### Configure

Create a `.env` file with your API key:

```bash
OPENAI_API_KEY=your-api-key
```

Optionally configure storage paths (defaults shown):

```bash
RAVEN_FACTS_FILE=~/notes/me/facts.yaml
RAVEN_MEMORY_DIR=~/.claude/memory
```

Configure the LLM provider and model in `mcp_agent.config.yaml`:

```yaml
openai:
  base_url: https://api.openai.com/v1  # or any OpenAI-compatible endpoint
  default_model: gpt-4o-mini
```

Works with OpenAI, OpenRouter, Ollama (`http://localhost:11434/v1`), LM Studio, or any provider that speaks the OpenAI API.

> **What I actually use:** [OpenRouter](https://openrouter.ai/) with `qwen/qwen-turbo` at $0.03/M input, $0.13/M output. Tested against Gemini 2.5 Flash ($0.30/$2.50) and DeepSeek V3.2 ($0.26/$0.38) — same pass rate, same speed, fraction of the cost. It's basically free real estate.

### Run as MCP server

```bash
./run.sh
```

Or manually:

```bash
source .env && uv run main.py
```

### Connect to Claude Code

Add to your `.claude.json`:

```json
{
  "mcpServers": {
    "raven": {
      "type": "stdio",
      "command": "/path/to/mcp-raven/run.sh",
      "args": [],
      "env": {}
    }
  }
}
```

Add permissions in `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "mcp__raven__remember",
      "mcp__raven__recall"
    ]
  }
}
```

## Storage format

### facts.yaml

```yaml
facts:
  - subject: Alice
    predicate: works_at
    object: "Acme Corp — backend engineering team"
    valid_from: "2026-01-15"
    valid_to: null
    confidence: 1.0
  - subject: Alice
    predicate: prefers
    object: "Neovim over VS Code"
    valid_from: "2026-03-01"
    valid_to: null
    confidence: 1.0
```

Facts support temporal validity — `valid_to` is set when a fact is superseded, so the reader can prefer current facts over expired ones.

### Memory files

Markdown with frontmatter:

```markdown
---
name: No git push -u
description: No git push -u
type: feedback
---

Use `git push origin <branch>`, never set upstream tracking.
```

## Tests

```bash
# Unit tests (no API key needed)
uv run pytest test_memory.py -k "not Integration"

# All tests (needs OPENAI_API_KEY)
source .env && uv run pytest test_memory.py -v
```

## Architecture

```
Claude Code / any MCP client
    │
    ├── remember(context) ──→ LLM categorizes ──→ writes to facts.yaml or memory_dir/*.md
    │
    └── recall(query) ──→ loads all memories ──→ LLM synthesizes answer
```

Built on [mcp-agent](https://github.com/lastmile-ai/mcp-agent) by LastMile AI. Works with any OpenAI-compatible LLM provider.

## License

MIT
