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
    # LLM enrichment disabled - qodex_parse provides enhanced metadata in mode="full"
    llm_metadata=LLMMetadataConfig(enabled=False),
)

# Policy templates for LLM advisor to select from
# LLM advisor analyzes document and chooses best template or synthesizes custom policy
# LLM enrichment disabled for all policies - qodex_parse handles it in mode="full"
POLICY_OVERRIDES: Dict[str, ChunkingPolicy] = {
    "ccbhc": ChunkingPolicy(
        policy_name="ccbhc",
        chunk_modes=("criteria", "semantic", "window"),
        window_size=350,
        window_overlap=70,
        max_chunk_tokens=450,
        chunk_tiers=("criteria", "semantic", "window"),
        highlight_phrases=("must", "required", "shall", "staffing", "services", "certification"),
        required_metadata=("policy", "namespace", "section_slug", "criteria_type"),
        llm_metadata=LLMMetadataConfig(enabled=False),
    ),
    "dsm5": ChunkingPolicy(
        policy_name="dsm5",
        chunk_modes=("criteria", "semantic"),
        window_size=300,
        window_overlap=60,
        max_chunk_tokens=400,
        chunk_tiers=("criteria", "semantic"),
        highlight_phrases=("diagnosis", "criteria", "symptoms", "disorder", "DSM"),
        required_metadata=("policy", "namespace", "section_slug", "disorder_code"),
        llm_metadata=LLMMetadataConfig(enabled=False),
    ),
    "treatment_planner": ChunkingPolicy(
        policy_name="treatment_planner",
        chunk_modes=("semantic", "window"),
        window_size=450,
        window_overlap=90,
        max_chunk_tokens=550,
        chunk_tiers=("semantic", "window"),
        highlight_phrases=("goal", "objective", "intervention", "outcome", "treatment"),
        required_metadata=("policy", "namespace", "section_slug", "treatment_phase"),
        llm_metadata=LLMMetadataConfig(enabled=False),
    ),
    "clinical_note": ChunkingPolicy(
        policy_name="clinical_note",
        chunk_modes=("semantic",),
        window_size=300,
        window_overlap=50,
        max_chunk_tokens=400,
        chunk_tiers=("semantic",),
        highlight_phrases=("patient", "diagnosis", "treatment", "medication", "progress"),
        required_metadata=("policy", "namespace", "section_slug", "note_type"),
        llm_metadata=LLMMetadataConfig(enabled=False),
    ),
    "legal_document": ChunkingPolicy(
        policy_name="legal_document",
        chunk_modes=("semantic", "window"),
        window_size=500,
        window_overlap=100,
        max_chunk_tokens=600,
        chunk_tiers=("semantic", "window"),
        highlight_phrases=("statute", "regulation", "case law", "precedent", "jurisdiction"),
        required_metadata=("policy", "namespace", "section_slug", "citation"),
        llm_metadata=LLMMetadataConfig(enabled=False),
    ),
    "research_paper": ChunkingPolicy(
        policy_name="research_paper",
        chunk_modes=("semantic", "window"),
        window_size=550,
        window_overlap=110,
        max_chunk_tokens=650,
        chunk_tiers=("semantic", "window"),
        highlight_phrases=("hypothesis", "methodology", "results", "conclusion", "significance"),
        required_metadata=("policy", "namespace", "section_slug", "paper_section"),
        llm_metadata=LLMMetadataConfig(enabled=False),
    ),
    "policy_manual": ChunkingPolicy(
        policy_name="policy_manual",
        chunk_modes=("semantic", "window"),
        window_size=400,
        window_overlap=80,
        max_chunk_tokens=500,
        chunk_tiers=("semantic", "window"),
        highlight_phrases=("policy", "procedure", "compliance", "requirement", "standard"),
        required_metadata=("policy", "namespace", "section_slug", "policy_number"),
        llm_metadata=LLMMetadataConfig(enabled=False),
    ),
    "technical_spec": ChunkingPolicy(
        policy_name="technical_spec",
        chunk_modes=("semantic", "window"),
        window_size=350,
        window_overlap=70,
        max_chunk_tokens=450,
        chunk_tiers=("semantic", "window"),
        highlight_phrases=("specification", "requirement", "interface", "protocol", "API"),
        required_metadata=("policy", "namespace", "section_slug", "spec_version"),
        llm_metadata=LLMMetadataConfig(enabled=False),
    ),
}


def resolve_policy(policy_name: str | None, fallback: str | None = None) -> ChunkingPolicy:
    """Resolve chunking policy using explicit advisor/schema selection."""

    candidate = policy_name or fallback
    if candidate and candidate in POLICY_OVERRIDES:
        return POLICY_OVERRIDES[candidate]

    return DEFAULT_POLICY


__all__ = ["ChunkingPolicy", "resolve_policy"]
