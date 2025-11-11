from __future__ import annotations

import re
from itertools import count
from typing import Iterable, List, Sequence

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
            previous_chunk_id: str | None = None
            normalized = _normalize_text(section.text)
            mode_chunks: List[tuple[str, str]] = []
            for mode in policy.chunk_modes:
                mode_chunks.extend(self._chunk_section(mode, normalized, policy, section))
            for tier, text in mode_chunks:
                if not text.strip():
                    continue
                chunk_id = f"{document_id}|{section.section_slug}|{next(chunk_counter)}"
                chunk_metadata = {
                    **section.metadata,
                    "policy": policy.policy_name,
                    "namespace": self._namespace,
                    "section_title": section.title,
                    "chunk_mode": tier,
                    "token_count": len(text.split()),
                }
                chunk = ChunkRecord(
                    chunk_id=chunk_id,
                    section_slug=section.section_slug,
                    text=text,
                    chunk_tier=tier if tier in policy.chunk_tiers else policy.chunk_tiers[0],
                    start_page=section.start_page,
                    end_page=section.end_page,
                    bbox_pointer=section.bbox_pointer,
                    summary=None,
                    key_terms=[],
                    requires_previous=previous_chunk_id is not None,
                    prev_chunk_id=previous_chunk_id,
                    confidence_note=None,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)
                previous_chunk_id = chunk_id
        return chunks

    def _chunk_section(
        self,
        mode: str,
        text: str,
        policy: ChunkingPolicy,
        section: SectionText,
    ) -> List[tuple[str, str]]:
        if mode == "highlight":
            return [("highlight", highlight) for highlight in _extract_highlights(text, policy.highlight_phrases)]
        if mode == "criteria":
            return [("criteria", chunk) for chunk in _split_by_criteria(text)]
        if mode == "semantic":
            return [("semantic", paragraph) for paragraph in _split_paragraphs(text, policy.max_chunk_tokens)]
        if mode == "window":
            return _window_chunks(text, policy)
        return [(mode, text)]


__all__ = ["ChunkingEngine"]


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    bullet_pattern = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
    text = bullet_pattern.sub("- ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _split_paragraphs(text: str, max_tokens: int) -> List[str]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    for paragraph in paragraphs:
        tokens = paragraph.split()
        if not tokens:
            continue
        if len(tokens) <= max_tokens:
            chunks.append(paragraph.strip())
            continue
        window_policy = ChunkingPolicy(
            policy_name="paragraph_window",
            chunk_modes=("window",),
            window_size=max_tokens,
            window_overlap=max_tokens // 5,
            max_chunk_tokens=max_tokens,
            chunk_tiers=("window",),
        )
        chunks.extend(text for _, text in _window_chunks(paragraph, window_policy))
    return chunks


def _window_chunks(text: str, policy: ChunkingPolicy) -> List[tuple[str, str]]:
    tokens = text.split()
    if not tokens:
        return []
    window_size = max(policy.window_size // 4, 1)
    overlap = max(policy.window_overlap // 4, 0)
    index = 0
    windows: List[tuple[str, str]] = []
    while index < len(tokens):
        next_index = min(len(tokens), index + window_size)
        window_text = " ".join(tokens[index:next_index])
        windows.append(("window", window_text))
        if next_index == len(tokens):
            break
        index = max(next_index - overlap, index + 1)
    return windows


def _split_by_criteria(text: str) -> List[str]:
    pattern = re.compile(r"(criterion\s+[a-z0-9]+\.?)(?=\s)", re.IGNORECASE)
    segments: List[str] = []
    last_index = 0
    for match in pattern.finditer(text):
        start = match.start()
        if start > last_index:
            segments.append(text[last_index:start].strip())
        last_index = start
    if last_index < len(text):
        segments.append(text[last_index:].strip())
    return [segment for segment in segments if segment]


def _extract_highlights(text: str, phrases: Sequence[str]) -> List[str]:
    highlights: List[str] = []
    lines = re.split(r"[\n\.]+", text)
    lowered_phrases = [phrase.lower() for phrase in phrases]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(phrase in lower for phrase in lowered_phrases) or stripped.isupper():
            highlights.append(stripped)
    return highlights
