from __future__ import annotations

from itertools import count
from typing import Iterable, List

from ..models.common import ChunkRecord, SectionText
from ..policies.resolver import ChunkingPolicy


class ChunkingEngine:
    """Generate chunks from section text using a policy."""

    def __init__(self, *, namespace: str = "catalyst") -> None:
        self._namespace = namespace

    def generate_chunks(
        self,
        sections: Iterable[SectionText],
        policy: ChunkingPolicy,
        document_id: str,
    ) -> List[ChunkRecord]:
        chunk_counter = count(start=1)
        chunks: List[ChunkRecord] = []
        for section in sections:
            tokens = section.text.split()
            window_size = max(policy.window_size // 4, 1)
            overlap = max(policy.window_overlap // 4, 0)
            index = 0
            previous_chunk_id: str | None = None
            while index < len(tokens):
                next_index = min(len(tokens), index + window_size)
                text = " ".join(tokens[index:next_index])
                chunk_id = f"{document_id}|{section.section_slug}|{next(chunk_counter)}"
                chunk = ChunkRecord(
                    chunk_id=chunk_id,
                    section_slug=section.section_slug,
                    text=text,
                    chunk_tier=policy.chunk_tiers[0],
                    start_page=section.start_page,
                    end_page=section.end_page,
                    bbox_pointer=section.bbox_pointer,
                    summary=None,
                    key_terms=[],
                    requires_previous=previous_chunk_id is not None,
                    prev_chunk_id=previous_chunk_id,
                    confidence_note=None,
                    metadata={
                        **section.metadata,
                        "policy": policy.policy_name,
                        "namespace": self._namespace,
                    },
                )
                chunks.append(chunk)
                previous_chunk_id = chunk_id
                if next_index == len(tokens):
                    break
                index = max(next_index - overlap, index + 1)
        return chunks


__all__ = ["ChunkingEngine"]
