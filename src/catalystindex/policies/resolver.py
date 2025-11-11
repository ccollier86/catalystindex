from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ChunkingPolicy:
    policy_name: str
    chunk_mode: str
    window_size: int
    window_overlap: int
    max_chunk_tokens: int
    chunk_tiers: List[str]


DEFAULT_POLICY = ChunkingPolicy(
    policy_name="default",
    chunk_mode="window",
    window_size=400,
    window_overlap=80,
    max_chunk_tokens=500,
    chunk_tiers=["semantic", "window"],
)


POLICY_OVERRIDES: Dict[str, ChunkingPolicy] = {
    "dsm5": ChunkingPolicy(
        policy_name="dsm5",
        chunk_mode="criteria",
        window_size=360,
        window_overlap=60,
        max_chunk_tokens=450,
        chunk_tiers=["criteria", "semantic", "highlight"],
    ),
    "treatment_planner": ChunkingPolicy(
        policy_name="treatment_planner",
        chunk_mode="window",
        window_size=512,
        window_overlap=128,
        max_chunk_tokens=620,
        chunk_tiers=["semantic", "window"],
    ),
}


def resolve_policy(document_title: str, schema_name: str | None) -> ChunkingPolicy:
    """Resolve chunking policy using schema override or heuristics."""

    if schema_name and schema_name in POLICY_OVERRIDES:
        return POLICY_OVERRIDES[schema_name]

    lower_title = document_title.lower()
    if "criteria" in lower_title or "diagnosis" in lower_title:
        return POLICY_OVERRIDES["dsm5"]

    return DEFAULT_POLICY


__all__ = ["ChunkingPolicy", "resolve_policy"]
