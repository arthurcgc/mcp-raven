"""Pydantic models for memory agent structured output."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MemoryDecision(BaseModel):
    """LLM's decision on how to categorize and store a piece of context."""

    action: Literal["store", "update", "discard"] = Field(
        description="Whether to store new, update existing, or discard",
    )
    category: Literal["fact", "feedback", "project", "reference"] = Field(
        description="Type of memory: fact (about the user), feedback (preferences), "
        "project (ongoing work), reference (external pointers)",
    )
    subject: Optional[str] = Field(
        default=None,
        description="For facts: the entity this is about (e.g. 'Alice', 'homelab')",
    )
    predicate: Optional[str] = Field(
        default=None,
        description="For facts: the relationship (e.g. 'works_at', 'prefers')",
    )
    object: Optional[str] = Field(
        default=None,
        description="For facts: the value (e.g. 'Acme Corp')",
    )
    valid_from: Optional[str] = Field(
        default=None,
        description="ISO date when this became true (YYYY-MM-DD)",
    )
    valid_to: Optional[str] = Field(
        default=None,
        description="ISO date when this stopped being true (YYYY-MM-DD), null if current",
    )
    filename: Optional[str] = Field(
        default=None,
        description="For non-fact categories: suggested filename (e.g. 'feedback_testing.md')",
    )
    title: Optional[str] = Field(
        default=None,
        description="For non-fact categories: short title for the memory",
    )
    content: Optional[str] = Field(
        default=None,
        description="For non-fact categories: formatted markdown content to write",
    )
    reasoning: str = Field(
        description="Brief explanation of why this categorization was chosen",
    )


class RecallResult(BaseModel):
    """LLM's synthesized answer from search results."""

    answer: str = Field(
        description="Synthesized answer to the query based on search results",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="File paths where the information was found",
    )
    confidence: Literal["high", "medium", "low", "none"] = Field(
        description="How confident the answer is based on available data",
    )
