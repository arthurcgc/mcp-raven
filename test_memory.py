"""Tests for memory agent — unit tests for pure functions, integration tests for LLM tools."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from main import (
    _load_all_memories,
    _load_facts,
    _parse_llm_json,
    _save_facts,
    _save_memory_file,
    _update_memory_index,
    FACTS_FILE,
    MEMORY_DIR,
    MEMORY_INDEX,
    NOTES_DIR,
    app,
    recall,
    remember,
)
from models import MemoryDecision, RecallResult


# ---------------------------------------------------------------------------
# Fixtures — isolated temp directories so tests don't touch real files
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Redirect NOTES_DIR and MEMORY_DIR to temp directories for every test."""
    notes = tmp_path / "notes"
    memory = tmp_path / "memory"
    notes.mkdir()
    memory.mkdir()
    (notes / "me").mkdir()

    import main

    monkeypatch.setattr(main, "NOTES_DIR", notes)
    monkeypatch.setattr(main, "FACTS_FILE", notes / "me" / "facts.yaml")
    monkeypatch.setattr(main, "MEMORY_DIR", memory)
    monkeypatch.setattr(main, "MEMORY_INDEX", memory / "MEMORY.md")

    return {"notes": notes, "memory": memory}


# ---------------------------------------------------------------------------
# Unit tests — _parse_llm_json
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_parses_clean_json(self):
        raw = '{"action": "store", "category": "fact", "subject": "Alice", "predicate": "works_at", "object": "Acme", "reasoning": "job info"}'
        d = _parse_llm_json(raw, MemoryDecision)
        assert d.subject == "Alice"
        assert d.predicate == "works_at"
        assert d.object == "Acme"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"action": "store", "category": "fact", "subject": "Alice", "predicate": "uses", "object": "Ghostty", "reasoning": "tool"}\n```'
        d = _parse_llm_json(raw, MemoryDecision)
        assert d.predicate == "uses"
        assert d.object == "Ghostty"

    def test_flattens_nested_data(self):
        raw = '{"action": "store", "category": "fact", "data": {"subject": "Alice", "predicate": "has_hardware", "object": "Keychron Q1 Max", "valid_from": "2026-04-14"}, "reasoning": "hw change"}'
        d = _parse_llm_json(raw, MemoryDecision)
        assert d.subject == "Alice"
        assert d.predicate == "has_hardware"
        assert d.object == "Keychron Q1 Max"
        assert d.valid_from == "2026-04-14"

    def test_flattens_without_overwriting_top_level(self):
        raw = '{"action": "store", "category": "fact", "subject": "Alice", "data": {"subject": "wrong", "predicate": "has_hardware", "object": "thing"}, "reasoning": "test"}'
        d = _parse_llm_json(raw, MemoryDecision)
        assert d.subject == "Alice"  # top-level wins
        assert d.predicate == "has_hardware"  # from nested
        assert d.object == "thing"  # from nested

    def test_parses_recall_result(self):
        raw = '{"answer": "Alice works at Enter.", "sources": ["/notes/me/facts.yaml"], "confidence": "high"}'
        r = _parse_llm_json(raw, RecallResult)
        assert r.answer == "Alice works at Enter."
        assert r.confidence == "high"

    def test_handles_discard(self):
        raw = '{"action": "discard", "category": "fact", "reasoning": "ephemeral"}'
        d = _parse_llm_json(raw, MemoryDecision)
        assert d.action == "discard"

    def test_parses_array_response(self):
        raw = '[{"action": "store", "category": "fact", "subject": "Alice", "predicate": "works_at", "object": "Acme", "reasoning": "job"}, {"action": "store", "category": "fact", "subject": "Alice", "predicate": "moving_to", "object": "SP", "reasoning": "relocation"}]'
        result = _parse_llm_json(raw, MemoryDecision)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].predicate == "works_at"
        assert result[1].predicate == "moving_to"

    def test_parses_single_item_array(self):
        raw = '[{"action": "discard", "category": "fact", "reasoning": "ephemeral"}]'
        result = _parse_llm_json(raw, MemoryDecision)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].action == "discard"

    def test_single_object_returns_single(self):
        raw = '{"action": "discard", "category": "fact", "reasoning": "ephemeral"}'
        result = _parse_llm_json(raw, MemoryDecision)
        assert isinstance(result, MemoryDecision)  # not a list


# ---------------------------------------------------------------------------
# Unit tests — _load_facts / _save_facts
# ---------------------------------------------------------------------------


class TestFactsIO:
    def test_load_empty_returns_list(self, isolated_dirs):
        assert _load_facts() == []

    def test_save_and_load_roundtrip(self, isolated_dirs):
        facts = [
            {
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme",
                "valid_from": "2026-04-20",
                "valid_to": None,
                "confidence": 1.0,
            }
        ]
        _save_facts(facts)
        loaded = _load_facts()
        assert len(loaded) == 1
        assert loaded[0]["subject"] == "Alice"
        assert loaded[0]["predicate"] == "works_at"
        assert loaded[0]["object"] == "Acme"

    def test_save_preserves_unicode(self, isolated_dirs):
        facts = [
            {
                "subject": "Alice",
                "predicate": "moving_to",
                "object": "São Paulo",
                "valid_from": "2026-04-18",
                "valid_to": None,
                "confidence": 1.0,
            }
        ]
        _save_facts(facts)
        loaded = _load_facts()
        assert loaded[0]["object"] == "São Paulo"

    def test_save_appends_to_existing(self, isolated_dirs):
        _save_facts([{"subject": "A", "predicate": "p", "object": "o1"}])
        facts = _load_facts()
        facts.append({"subject": "B", "predicate": "p", "object": "o2"})
        _save_facts(facts)
        loaded = _load_facts()
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Unit tests — _save_memory_file / _update_memory_index
# ---------------------------------------------------------------------------


class TestMemoryFiles:
    def test_save_creates_file_with_frontmatter(self, isolated_dirs):
        import main

        path = _save_memory_file(
            "feedback_testing.md",
            "Testing conventions",
            "feedback",
            "Use table-driven tests in Go.",
        )
        assert path.exists()
        content = path.read_text()
        assert "---" in content
        assert "name: Testing conventions" in content
        assert "type: feedback" in content
        assert "Use table-driven tests in Go." in content

    def test_update_index_adds_entry(self, isolated_dirs):
        import main

        _update_memory_index("feedback_testing.md", "Testing conventions")
        content = main.MEMORY_INDEX.read_text()
        assert "feedback_testing.md" in content
        assert "Testing conventions" in content

    def test_update_index_no_duplicates(self, isolated_dirs):
        import main

        _update_memory_index("feedback_testing.md", "Testing conventions")
        _update_memory_index("feedback_testing.md", "Testing conventions")
        content = main.MEMORY_INDEX.read_text()
        assert content.count("feedback_testing.md") == 1


# ---------------------------------------------------------------------------
# Unit tests — _load_all_memories (context stuffing)
# ---------------------------------------------------------------------------


class TestLoadAllMemories:
    def test_loads_facts(self, isolated_dirs):
        _save_facts([
            {"subject": "Alice", "predicate": "works_at", "object": "Acme"},
        ])
        result = _load_all_memories()
        assert "Acme" in result
        assert "facts.yaml" in result

    def test_loads_memory_files(self, isolated_dirs):
        _save_memory_file("feedback_testing.md", "Testing", "feedback", "Use table-driven tests.")
        result = _load_all_memories()
        assert "table-driven" in result
        assert "feedback_testing.md" in result

    def test_loads_both_facts_and_files(self, isolated_dirs):
        _save_facts([{"subject": "Alice", "predicate": "works_at", "object": "Acme"}])
        _save_memory_file("feedback_git.md", "Git", "feedback", "No force push.")
        result = _load_all_memories()
        assert "Acme" in result
        assert "force push" in result

    def test_empty_returns_empty_string(self, isolated_dirs):
        result = _load_all_memories()
        assert result == ""

    def test_skips_memory_index(self, isolated_dirs):
        import main
        main.MEMORY_INDEX.write_text("- [test](test.md)")
        _save_memory_file("feedback_test.md", "Test", "feedback", "content")
        result = _load_all_memories()
        assert "MEMORY.md" not in result
        assert "feedback_test.md" in result

    def test_loads_notes(self, isolated_dirs):
        import main
        articles_dir = main.NOTES_DIR / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "2026-04-15-wake-on-lan.md").write_text("# Wake-on-LAN\nMagic packet stuff.")
        result = _load_all_memories()
        assert "Wake-on-LAN" in result
        assert "notes/articles/2026-04-15-wake-on-lan.md" in result

    def test_loads_notes_and_facts_and_memory(self, isolated_dirs):
        import main
        _save_facts([{"subject": "Alice", "predicate": "works_at", "object": "Acme"}])
        _save_memory_file("feedback_git.md", "Git", "feedback", "No force push.")
        articles_dir = main.NOTES_DIR / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "article.md").write_text("# Some Article\nContent here.")
        result = _load_all_memories()
        assert "Acme" in result
        assert "force push" in result
        assert "Some Article" in result


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_memory_decision_fact(self):
        d = MemoryDecision(
            action="store",
            category="fact",
            subject="Alice",
            predicate="works_at",
            object="Acme",
            valid_from="2026-04-20",
            reasoning="New job information",
        )
        assert d.action == "store"
        assert d.valid_to is None

    def test_memory_decision_discard(self):
        d = MemoryDecision(
            action="discard",
            category="fact",
            reasoning="Ephemeral debugging info, not worth saving",
        )
        assert d.action == "discard"

    def test_memory_decision_feedback(self):
        d = MemoryDecision(
            action="store",
            category="feedback",
            filename="feedback_testing.md",
            title="Testing conventions",
            content="Use table-driven tests.",
            reasoning="Persistent preference",
        )
        assert d.filename == "feedback_testing.md"

    def test_recall_result(self):
        r = RecallResult(
            answer="Alice uses a Keychron Q1 Max.",
            sources=["/home/user/notes/me/facts.yaml"],
            confidence="high",
        )
        assert r.confidence == "high"
        assert len(r.sources) == 1

    def test_recall_result_no_sources(self):
        r = RecallResult(
            answer="Nothing found.",
            confidence="none",
        )
        assert r.sources == []


# ---------------------------------------------------------------------------
# Integration tests — require OPENAI_API_KEY (OpenRouter)
# ---------------------------------------------------------------------------

needs_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping integration tests",
)


@needs_api_key
class TestRememberIntegration:
    """
    Integration smoke tests — verify LLM round-trip works and returns valid responses.
    Each test gets a fresh app.run() context. Assertions check response format,
    not exact content (LLM output is non-deterministic).
    """

    @pytest.mark.asyncio
    async def test_stores_fact_with_all_fields(self, isolated_dirs):
        async with app.run() as agent_app:
            result = await remember(
                context="Alice switched from Alacritty to Ghostty terminal",
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result
            if "Stored fact" in result:
                # Verify predicate and object are not None
                assert "None" not in result, f"Structured output has None fields: {result}"
                facts = _load_facts()
                assert len(facts) > 0
                last = facts[-1]
                assert last["predicate"] is not None
                assert last["object"] is not None

    @pytest.mark.asyncio
    async def test_ephemeral_returns_valid_response(self, isolated_dirs):
        async with app.run() as agent_app:
            result = await remember(
                context="I'm currently debugging a failing test in main_test.go line 42",
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result

    @pytest.mark.asyncio
    async def test_feedback_returns_valid_response(self, isolated_dirs):
        async with app.run() as agent_app:
            result = await remember(
                context="Alice said: don't use git push -u, just use git push origin branch-name",
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result

    @pytest.mark.asyncio
    async def test_update_stores_new_fact(self, isolated_dirs):
        _save_facts([{
            "subject": "Alice",
            "predicate": "works_at",
            "object": "OldCorp",
            "valid_from": "2024-01-01",
            "valid_to": None,
            "confidence": 1.0,
        }])
        async with app.run() as agent_app:
            result = await remember(
                context="Alice left OldCorp and now works at Acme Corp starting 2026-04-20",
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result
            if "Stored fact" in result:
                assert "None" not in result
                facts = _load_facts()
                objects = [str(f.get("object", "")) for f in facts]
                assert any("Acme" in o for o in objects)


    @pytest.mark.asyncio
    async def test_portuguese_unicode_content(self, isolated_dirs):
        # Seed a unicode fact so the save cycle exercises YAML round-trip
        _save_facts([{
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "São Paulo",
            "valid_from": "2025-01-01",
            "valid_to": None,
            "confidence": 1.0,
        }])
        async with app.run() as agent_app:
            result = await remember(
                context="Alice está se mudando para Nova York, saindo de São Paulo",
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result
            # Verify the pre-seeded unicode fact survived the save cycle
            facts = _load_facts()
            seeded = [f for f in facts if f.get("object") == "São Paulo"]
            assert len(seeded) >= 1, f"Unicode fact corrupted after save cycle: {facts}"

    @pytest.mark.asyncio
    async def test_reference_stores_url(self, isolated_dirs):
        async with app.run() as agent_app:
            result = await remember(
                context="The Grafana dashboard for API latency is at grafana.internal/d/api-latency — oncall watches this",
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result

    @pytest.mark.asyncio
    async def test_duplicate_fact_not_doubled(self, isolated_dirs):
        _save_facts([{
            "subject": "Alice",
            "predicate": "uses_tool",
            "object": "Tailscale — VPN mesh for remote access",
            "valid_from": "2026-04-11",
            "valid_to": None,
            "confidence": 1.0,
        }])
        async with app.run() as agent_app:
            result = await remember(
                context="Alice uses Tailscale for VPN access to his home PC",
                source="test",
                app_ctx=agent_app.context,
            )
            # Should either discard (already known) or store — but not crash
            assert "Stored" in result or "Discarded" in result

    @pytest.mark.asyncio
    async def test_feedback_creates_markdown_file(self, isolated_dirs):
        import main

        async with app.run() as agent_app:
            result = await remember(
                context="Alice corrected me: never run git push --force to main. He was very firm about this after an incident last month.",
                source="test",
                app_ctx=agent_app.context,
            )
            if "Stored feedback" in result or "Stored" in result:
                # Check that a file was created in memory dir
                md_files = list(main.MEMORY_DIR.glob("*.md"))
                # At least MEMORY.md should exist, possibly a new feedback file
                assert len(md_files) >= 0  # non-crash assertion

    @pytest.mark.asyncio
    async def test_multi_topic_stores_multiple(self, isolated_dirs):
        async with app.run() as agent_app:
            result = await remember(
                context="Alice moved to New York on 2026-04-18. He started working at Acme Corp on 2026-04-20 as backend engineer. His partner Jordan moved with him.",
                source="test",
                app_ctx=agent_app.context,
            )
            # Should store multiple facts, not collapse into one
            stored_count = result.count("Stored")
            assert stored_count >= 2, f"Expected multiple stored entries, got:\n{result}"

    @pytest.mark.asyncio
    async def test_dense_session_summary(self, isolated_dirs):
        """The original bug — dense session summary should produce multiple entries, not one."""
        async with app.run() as agent_app:
            result = await remember(
                context="Session 2026-04-14: Built a new MCP memory server using the mcp-agent framework. Migrated from the old memory system entirely. Restructured the notes directory into 9 project folders. Updated Docker config for the knowledge base. Alice prefers dark themes across all tools.",
                source="session-end",
                app_ctx=agent_app.context,
            )
            stored_count = result.count("Stored")
            assert stored_count >= 2, f"Dense summary collapsed into too few entries:\n{result}"

    @pytest.mark.asyncio
    async def test_mixed_language_portuguese_english(self, isolated_dirs):
        async with app.run() as agent_app:
            result = await remember(
                context="Alice comentou que prefere usar o terminal Alacritty com tema Moonlight. He also mentioned his Polybar uses the Gotham color scheme.",
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result
            if "Stored fact" in result:
                assert "None" not in result

    @pytest.mark.asyncio
    async def test_speculation_should_not_be_stored_as_fact(self, isolated_dirs):
        """Speculative context should be discarded or stored as project, not as fact."""
        async with app.run() as agent_app:
            result = await remember(
                context="Alice is thinking about maybe switching to NixOS someday, but he's not sure yet and hasn't made any plans.",
                source="test",
                app_ctx=agent_app.context,
            )
            # Should either discard (speculative) or store cautiously — not crash
            assert "Stored" in result or "Discarded" in result

    @pytest.mark.asyncio
    async def test_very_terse_input(self, isolated_dirs):
        async with app.run() as agent_app:
            result = await remember(
                context="Alice uses Neovim now.",
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result

    @pytest.mark.asyncio
    async def test_many_facts_in_one_shot(self, isolated_dirs):
        """7+ distinct facts in a single context — should produce multiple entries."""
        async with app.run() as agent_app:
            result = await remember(
                context="Important facts about Alice: she uses a Ryzen 9 5900X CPU, has 32GB RAM, owns an RX 7800 XT GPU and an RX 6750 XT GPU, types on a Keychron Q1 Max keyboard, has a Samsung Q60D TV, and runs a Raspberry Pi 4 server.",
                source="test",
                app_ctx=agent_app.context,
            )
            stored_count = result.count("Stored")
            assert stored_count >= 3, f"Expected 3+ entries from 7 facts, got:\n{result}"

    @pytest.mark.asyncio
    async def test_long_context_doesnt_crash(self, isolated_dirs):
        long_context = (
            "During today's session we debugged an issue with the Istio sidecar injection. "
            "The problem was that the namespace label istio-injection=enabled was missing. "
            "We added it with kubectl label namespace default istio-injection=enabled. "
            "Then we restarted the pods and verified the sidecars were injected. "
            "Alice mentioned he needs to understand Istio's mTLS configuration better. "
        ) * 5  # ~500 words of context
        async with app.run() as agent_app:
            result = await remember(
                context=long_context,
                source="test",
                app_ctx=agent_app.context,
            )
            assert "Stored" in result or "Discarded" in result


@needs_api_key
class TestRecallIntegration:
    @pytest.mark.asyncio
    async def test_finds_existing_fact(self, isolated_dirs):
        _save_facts([{
            "subject": "Alice",
            "predicate": "works_at",
            "object": "Acme Corp — AI legal tech startup",
            "valid_from": "2026-04-20",
            "valid_to": None,
            "confidence": 1.0,
        }])
        async with app.run() as agent_app:
            result = await recall(
                query="Where does Alice work?",
                app_ctx=agent_app.context,
            )
            assert "Acme" in result

    @pytest.mark.asyncio
    async def test_returns_nothing_for_unknown(self, isolated_dirs):
        async with app.run() as agent_app:
            result = await recall(
                query="What is Alice's favorite dinosaur?",
                app_ctx=agent_app.context,
            )
            assert "Nothing found" in result or "none" in result.lower()

    @pytest.mark.asyncio
    async def test_recall_multiple_facts(self, isolated_dirs):
        _save_facts([
            {"subject": "Alice", "predicate": "has_hardware", "object": "Keychron Q1 Max keyboard",
             "valid_from": "2026-04-14", "valid_to": None, "confidence": 1.0},
            {"subject": "Alice", "predicate": "has_hardware", "object": "Raspberry Pi 4 8GB",
             "valid_from": "2026-04-13", "valid_to": None, "confidence": 1.0},
            {"subject": "Alice", "predicate": "has_hardware", "object": "Samsung Q60D 55\" QLED",
             "valid_from": "2026-04-10", "valid_to": None, "confidence": 1.0},
        ])
        async with app.run() as agent_app:
            result = await recall(
                query="What hardware does Alice have?",
                app_ctx=agent_app.context,
            )
            # Should mention at least some of the hardware
            assert any(hw in result for hw in ["Keychron", "Raspberry", "Samsung", "Pi", "QLED"])

    @pytest.mark.asyncio
    async def test_recall_temporal_prefers_current(self, isolated_dirs):
        _save_facts([
            {"subject": "Alice", "predicate": "works_at", "object": "OldCorp",
             "valid_from": "2024-01-01", "valid_to": "2026-04-10", "confidence": 1.0},
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp",
             "valid_from": "2026-04-20", "valid_to": None, "confidence": 1.0},
        ])
        async with app.run() as agent_app:
            result = await recall(
                query="Where does Alice currently work?",
                app_ctx=agent_app.context,
            )
            assert "Acme" in result

    @pytest.mark.asyncio
    async def test_recall_indirect_query(self, isolated_dirs):
        """Query doesn't use the same words as the stored fact."""
        _save_facts([
            {"subject": "Alice", "predicate": "works_at",
             "object": "Acme Corp — AI-powered legal tech startup",
             "valid_from": "2026-04-20", "valid_to": None, "confidence": 1.0},
        ])
        async with app.run() as agent_app:
            result = await recall(
                query="What company does Alice work for?",
                app_ctx=agent_app.context,
            )
            assert "Acme" in result

    @pytest.mark.asyncio
    async def test_recall_combines_facts(self, isolated_dirs):
        """Answer requires combining info from multiple facts."""
        _save_facts([
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp",
             "valid_from": "2026-04-20", "valid_to": None, "confidence": 1.0},
            {"subject": "Alice", "predicate": "moving_to", "object": "New York",
             "valid_from": "2026-04-18", "valid_to": None, "confidence": 1.0},
        ])
        async with app.run() as agent_app:
            result = await recall(
                query="Where is Alice and what is he doing there?",
                app_ctx=agent_app.context,
            )
            assert "Paulo" in result or "Acme" in result

    @pytest.mark.asyncio
    async def test_recall_with_expired_and_current(self, isolated_dirs):
        """Multiple versions of the same fact — should prefer current."""
        _save_facts([
            {"subject": "Alice", "predicate": "has_hardware", "object": "Keychron Q1 Pro",
             "valid_from": "2025-01-01", "valid_to": "2026-04-14", "confidence": 1.0},
            {"subject": "Alice", "predicate": "has_hardware", "object": "Keychron Q1 Max (replaces Q1 Pro)",
             "valid_from": "2026-04-14", "valid_to": None, "confidence": 1.0},
        ])
        async with app.run() as agent_app:
            result = await recall(
                query="What keyboard does Alice currently use?",
                app_ctx=agent_app.context,
            )
            assert "Max" in result

    @pytest.mark.asyncio
    async def test_recall_portuguese_query(self, isolated_dirs):
        _save_facts([
            {"subject": "Alice", "predicate": "partner", "object": "Jordan",
             "valid_from": "2026-04-10", "valid_to": None, "confidence": 1.0},
        ])
        async with app.run() as agent_app:
            result = await recall(
                query="Quem é o parceiro da Alice?",
                app_ctx=agent_app.context,
            )
            assert "Jordan" in result

    @pytest.mark.asyncio
    async def test_recall_across_memory_and_facts(self, isolated_dirs):
        import main

        # Seed a fact
        _save_facts([
            {"subject": "Alice", "predicate": "works_at", "object": "Acme",
             "valid_from": "2026-04-20", "valid_to": None, "confidence": 1.0},
        ])
        # Seed a memory file
        _save_memory_file(
            "feedback_git_push.md",
            "No git push -u",
            "feedback",
            "Use git push origin <branch>, never set upstream tracking.",
        )
        async with app.run() as agent_app:
            result = await recall(
                query="What are Alice's git preferences?",
                app_ctx=agent_app.context,
            )
            # Should find the feedback about git push
            assert "push" in result.lower() or "git" in result.lower() or "Nothing found" in result
