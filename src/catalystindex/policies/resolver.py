from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LLMMetadataConfig:
    enabled: bool = False
    model: str = "text-embedding-3-large"
    summary_length: int = 220
    max_terms: int = 6


@dataclass(frozen=True)
class ChunkingPolicy:
    policy_name: str
    chunk_modes: Tuple[str, ...]
    window_size: int
    window_overlap: int
    max_chunk_tokens: int
    chunk_tiers: Tuple[str, ...]
    highlight_phrases: Tuple[str, ...] = ()
    required_metadata: Tuple[str, ...] = ()
    llm_metadata: LLMMetadataConfig = LLMMetadataConfig()


DEFAULT_POLICY = ChunkingPolicy(
    policy_name="default",
    chunk_modes=("semantic", "window"),
    window_size=400,
    window_overlap=80,
    max_chunk_tokens=500,
    chunk_tiers=("semantic", "window"),
    required_metadata=("policy", "namespace", "section_slug"),
)


POLICY_OVERRIDES: Dict[str, ChunkingPolicy] = {
    "dsm5": ChunkingPolicy(
        policy_name="dsm5",
        chunk_modes=("criteria", "highlight", "window"),
        window_size=360,
        window_overlap=60,
        max_chunk_tokens=450,
        chunk_tiers=("criteria", "semantic", "highlight"),
        highlight_phrases=("warning", "risk", "suicide"),
        llm_metadata=LLMMetadataConfig(enabled=True, summary_length=240, max_terms=8),
    ),
    "treatment_planner": ChunkingPolicy(
        policy_name="treatment_planner",
        chunk_modes=("semantic", "window"),
        window_size=512,
        window_overlap=128,
        max_chunk_tokens=620,
        chunk_tiers=("semantic", "window"),
        llm_metadata=LLMMetadataConfig(enabled=True, model="gpt-4o-mini", summary_length=260, max_terms=10),
    ),
}


def resolve_policy(policy_name: str | None) -> ChunkingPolicy:
    """Resolve chunking policy using explicit advisor/schema selection."""

    if policy_name and policy_name in POLICY_OVERRIDES:
        return POLICY_OVERRIDES[policy_name]

    return DEFAULT_POLICY


__all__ = ["ChunkingPolicy", "resolve_policy"]
