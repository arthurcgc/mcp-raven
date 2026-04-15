"""
mcp-raven — LLM-powered memory server for Claude Code.

- Facts stored in ~/notes/me/facts.yaml (KG-style, temporal validity)
- Feedback/project/reference stored in ~/.claude/memory/ (markdown with frontmatter)
- LLM handles categorization (writer) and synthesis (reader)

Exposed as MCP tools: remember + recall
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.core.context import Context
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from models import MemoryDecision, RecallResult
from prompts import WRITER_INSTRUCTION, READER_INSTRUCTION

FACTS_FILE = Path(os.environ.get("RAVEN_FACTS_FILE", Path.home() / "notes" / "me" / "facts.yaml"))
MEMORY_DIR = Path(os.environ.get("RAVEN_MEMORY_DIR", Path.home() / ".claude" / "memory"))
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

app = MCPApp(name="mcp_raven", description="Long-term memory for Claude Code")


def _parse_llm_json(raw: str, model_class):
    """Parse LLM output as JSON, handling markdown fences, nested 'data' wrappers, and arrays."""
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    text = re.sub(r"\n?```\s*$", "", text)

    data = json.loads(text)

    def _flatten(obj: dict) -> dict:
        """Flatten nested 'data' wrapper if present (Gemini quirk)."""
        if "data" in obj and isinstance(obj["data"], dict):
            nested = obj.pop("data")
            for k, v in nested.items():
                if k not in obj or obj[k] is None:
                    obj[k] = v
        return obj

    # Handle both single object and array responses
    if isinstance(data, list):
        return [model_class.model_validate(_flatten(item)) for item in data]
    return model_class.model_validate(_flatten(data))


def _load_facts() -> list[dict]:
    """Load existing facts from YAML file."""
    if not FACTS_FILE.exists():
        return []
    with open(FACTS_FILE) as f:
        data = yaml.safe_load(f) or {}
    return data.get("facts", [])


def _save_facts(facts: list[dict]) -> None:
    """Write facts back to YAML file."""
    FACTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FACTS_FILE, "w") as f:
        yaml.dump(
            {"facts": facts},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )


def _save_memory_file(filename: str, title: str, category: str, content: str) -> Path:
    """Write a markdown memory file with frontmatter."""
    filepath = MEMORY_DIR / filename
    frontmatter = f"---\nname: {title}\ndescription: {title}\ntype: {category}\n---\n\n"
    filepath.write_text(frontmatter + content)
    return filepath


def _update_memory_index(filename: str, title: str) -> None:
    """Add an entry to MEMORY.md if not already present."""
    if not MEMORY_INDEX.exists():
        MEMORY_INDEX.write_text("")

    existing = MEMORY_INDEX.read_text()
    entry = f"- [{title}]({filename})"
    if filename not in existing:
        with open(MEMORY_INDEX, "a") as f:
            f.write(f"{entry}\n")


NOTES_DIR = Path(os.environ.get("RAVEN_NOTES_DIR", Path.home() / "notes"))


def _load_all_memories() -> str:
    """Load all memories — facts, memory files, and all notes."""
    parts = []

    # Load facts
    facts = _load_facts()
    if facts:
        parts.append("=== Facts (facts.yaml) ===\n" + yaml.dump(facts, default_flow_style=False))

    # Load all memory markdown files
    if MEMORY_DIR.exists():
        for md_file in sorted(MEMORY_DIR.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            content = md_file.read_text().strip()
            if content:
                parts.append(f"=== memory/{md_file.name} ===\n{content}")

    # Load all notes
    if NOTES_DIR.exists():
        for md_file in sorted(NOTES_DIR.rglob("*.md")):
            # Skip obsidian internals and files already loaded via MEMORY_DIR symlink
            if ".obsidian" in md_file.parts:
                continue
            try:
                resolved = md_file.resolve()
                if MEMORY_DIR.exists() and str(resolved).startswith(str(MEMORY_DIR.resolve())):
                    continue
                content = md_file.read_text().strip()
                if content:
                    relative = md_file.relative_to(NOTES_DIR)
                    parts.append(f"=== notes/{relative} ===\n{content}")
            except (OSError, ValueError):
                continue

    return "\n\n".join(parts) if parts else ""


@app.tool()
async def remember(context: str, source: str = "", app_ctx: Optional[Context] = None) -> str:
    """
    Analyze context and store it in long-term memory if worth keeping.

    Args:
        context: The information to potentially memorize (raw conversation context).
        source: Optional source label (e.g. "session", "user-request").
    """
    logger = app_ctx.app.logger
    logger.info(f"remember called — source={source}, context_len={len(context)}")

    today = date.today().isoformat()
    message = (
        f"Today's date: {today}\n"
        f"Source: {source or 'conversation'}\n\n"
        f"Context to evaluate:\n{context}"
    )

    # Load existing facts for deduplication context
    existing_facts = _load_facts()
    if existing_facts:
        recent = existing_facts[-20:]  # last 20 facts for context
        facts_context = yaml.dump(recent, default_flow_style=False)
        message += f"\n\nExisting recent facts (for deduplication):\n{facts_context}"

    writer = Agent(
        name="memory_writer",
        instruction=WRITER_INSTRUCTION,
        server_names=[],
        context=app_ctx,
    )

    async with writer:
        llm = await writer.attach_llm(OpenAIAugmentedLLM)
        raw = await llm.generate_str(message=message)
        try:
            parsed = _parse_llm_json(raw, MemoryDecision)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse writer response: {e}\nRaw: {raw}")
            return f"Error parsing writer response: {e}"

    # Normalize to list
    decisions = parsed if isinstance(parsed, list) else [parsed]

    results = []
    facts = _load_facts()
    facts_changed = False

    for decision in decisions:
        logger.info(f"Writer decision: action={decision.action}, category={decision.category}")

        if decision.action == "discard":
            results.append(f"Discarded — {decision.reasoning}")
            continue

        if decision.category == "fact":
            # Dedup: skip if identical fact already exists (same subject+predicate+object)
            duplicate = any(
                f.get("subject") == decision.subject
                and f.get("predicate") == decision.predicate
                and f.get("object") == decision.object
                and f.get("valid_to") is None
                for f in facts
            )
            if duplicate:
                results.append(f"Skipped duplicate: {decision.subject} → {decision.predicate} → {decision.object}")
                continue

            # If updating, invalidate old matching facts
            if decision.action == "update":
                for fact in facts:
                    if (fact.get("subject") == decision.subject
                            and fact.get("predicate") == decision.predicate
                            and fact.get("valid_to") is None):
                        fact["valid_to"] = today

            new_fact = {
                "subject": decision.subject,
                "predicate": decision.predicate,
                "object": decision.object,
                "valid_from": decision.valid_from or today,
                "valid_to": decision.valid_to,
                "confidence": 1.0,
            }
            facts.append(new_fact)
            facts_changed = True
            results.append(f"Stored fact: {decision.subject} → {decision.predicate} → {decision.object}")

        else:
            filename = decision.filename or f"{decision.category}_{today}.md"
            title = decision.title or decision.category
            content = decision.content or decision.reasoning

            filepath = _save_memory_file(filename, title, decision.category, content)
            _update_memory_index(filename, title)
            results.append(f"Stored {decision.category}: {filepath}")

    if facts_changed:
        _save_facts(facts)

    return "\n".join(results)


@app.tool()
async def recall(query: str, app_ctx: Optional[Context] = None) -> str:
    """
    Search long-term memory and return a synthesized answer.

    Args:
        query: Natural language question about past context, facts, or preferences.
    """
    logger = app_ctx.app.logger
    logger.info(f"recall called — query={query}")

    all_memories = _load_all_memories()

    if not all_memories:
        return "Nothing found in local memory."

    message = (
        f"Query: {query}\n\n"
        f"All memories:\n{all_memories}"
    )

    reader = Agent(
        name="memory_reader",
        instruction=READER_INSTRUCTION,
        server_names=[],
        context=app_ctx,
    )

    async with reader:
        llm = await reader.attach_llm(OpenAIAugmentedLLM)
        raw = await llm.generate_str(message=message)
        result: RecallResult = _parse_llm_json(raw, RecallResult)

    logger.info(f"Reader result: confidence={result.confidence}, sources={result.sources}")

    if result.confidence == "none":
        return "Nothing found in local memory."

    return f"{result.answer}\n\nSources: {', '.join(result.sources)}\nConfidence: {result.confidence}"


async def main():
    async with app.run() as agent_app:
        # Expose as MCP server
        from mcp_agent.server.app_server import create_mcp_server_for_app

        mcp_server = create_mcp_server_for_app(agent_app)
        await mcp_server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
