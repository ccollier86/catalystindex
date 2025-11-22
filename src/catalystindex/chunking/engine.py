from __future__ import annotations

import logging
import re
from itertools import count
from typing import Iterable, List, Sequence

from ..models.common import ChunkRecord, SectionText
from ..policies.resolver import ChunkingPolicy

logger = logging.getLogger(__name__)

# Import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    _TOKENIZER = None


def _count_tokens(text: str) -> int:
    """Count actual tokens using tiktoken (accurate) or estimate from words (fallback)."""
    if TIKTOKEN_AVAILABLE and _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    # Fallback: word count * 1.3 (conservative estimate)
    word_count = len(text.split())
    return int(word_count * 1.3)


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
        sections_list = list(sections)

        logger.info(
            f"Starting chunking for document '{document_id}' with policy '{policy.policy_name}' "
            f"across {len(sections_list)} sections"
        )

        for section in sections_list:
            previous_chunk_id: str | None = None
            normalized = _normalize_text(section.text)
            token_count = _count_tokens(normalized)

            logger.debug(
                f"Processing section '{section.section_slug}' (pages {section.start_page}-{section.end_page}, "
                f"{token_count} tokens, modes={policy.chunk_modes})"
            )

            mode_chunks: List[tuple[str, str]] = []
            for mode in policy.chunk_modes:
                mode_chunks.extend(self._chunk_section(mode, normalized, policy, section))

            section_chunk_count = 0
            for tier, text in mode_chunks:
                if not text.strip():
                    continue
                chunk_id = f"{document_id}|{section.section_slug}|{next(chunk_counter)}"
                chunk_token_count = _count_tokens(text)

                # CRITICAL: Validate chunk size using ACTUAL tokens
                if chunk_token_count > policy.max_chunk_tokens:
                    logger.error(
                        f"Chunk {chunk_id} has {chunk_token_count} tokens, which exceeds policy max {policy.max_chunk_tokens}! "
                        f"This indicates a chunking failure and violates the policy contract."
                    )
                    # Skip chunks that exceed the limit - they should NEVER be created
                    continue

                chunk_metadata = {
                    **section.metadata,
                    "policy": policy.policy_name,
                    "namespace": self._namespace,
                    "section_title": section.title,
                    "chunk_mode": tier,
                    "token_count": chunk_token_count,
                    # Preserve OpenParse-specific fields for visual/spatial features
                    "bbox": section.bbox,
                    "node_type": section.node_type,
                    "image_refs": section.image_refs,
                    "text_height": section.text_height,
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
                section_chunk_count += 1

            logger.debug(f"Section '{section.section_slug}' produced {section_chunk_count} chunks")

        logger.info(
            f"Chunking complete for document '{document_id}': created {len(chunks)} chunks from {len(sections_list)} sections "
            f"(avg {len(chunks) // max(len(sections_list), 1)} chunks/section)"
        )

        return chunks

    def _chunk_section(
        self,
        mode: str,
        text: str,
        policy: ChunkingPolicy,
        section: SectionText,
    ) -> List[tuple[str, str]]:
        # Validate section size - prevent massive chunks using ACTUAL token count
        token_count = _count_tokens(text)
        if token_count > 15000:
            logger.warning(
                f"Section '{section.section_slug}' has {token_count} tokens (pages {section.start_page}-{section.end_page}). "
                f"Forcing window chunking to prevent massive chunks."
            )
            # Force window chunking for oversized sections
            mode = "window"

        if mode == "highlight":
            return [("highlight", highlight) for highlight in _extract_highlights(text, policy.highlight_phrases)]
        if mode == "criteria":
            return [("criteria", chunk) for chunk in _split_by_criteria(text)]
        if mode == "semantic":
            chunks = _split_paragraphs(text, policy.max_chunk_tokens)
            # Validate semantic chunking results - ensure it actually split the text
            if not chunks:
                logger.warning(f"Semantic chunking produced no chunks for section '{section.section_slug}'. Falling back to window chunking.")
                return _window_chunks(text, policy)
            if len(chunks) == 1 and token_count > policy.max_chunk_tokens * 1.5:
                logger.warning(
                    f"Semantic chunking produced 1 chunk of {token_count} tokens for section '{section.section_slug}' "
                    f"(exceeds policy max {policy.max_chunk_tokens}). Falling back to window chunking."
                )
                return _window_chunks(text, policy)
            return [("semantic", paragraph) for paragraph in chunks]
        if mode == "window":
            return _window_chunks(text, policy)

        # CRITICAL FIX: Never return entire text as one chunk
        # Default mode falls back to window chunking to ensure splitting
        logger.warning(f"Unknown chunk mode '{mode}' for section '{section.section_slug}'. Falling back to window chunking.")
        return _window_chunks(text, policy)


__all__ = ["ChunkingEngine"]


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    bullet_pattern = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
    text = bullet_pattern.sub("- ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _split_paragraphs(text: str, max_tokens: int) -> List[str]:
    """
    Split text by paragraphs (double newlines), with window chunking for oversized paragraphs.

    Args:
        text: Input text to split
        max_tokens: Maximum ACTUAL tokens per paragraph chunk (using tiktoken)

    Returns:
        List of paragraph chunks, with large paragraphs split into windows
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []

    total_tokens = _count_tokens(text)
    logger.debug(f"Splitting {total_tokens} tokens into paragraphs (found {len(paragraphs)} paragraphs, max_tokens={max_tokens})")

    for paragraph in paragraphs:
        paragraph_stripped = paragraph.strip()
        if not paragraph_stripped:
            continue

        paragraph_tokens = _count_tokens(paragraph_stripped)

        if paragraph_tokens <= max_tokens:
            chunks.append(paragraph_stripped)
            continue

        # Paragraph exceeds max_tokens - split it with window chunking
        logger.debug(f"Paragraph has {paragraph_tokens} tokens (> {max_tokens}). Applying window chunking.")
        window_policy = ChunkingPolicy(
            policy_name="paragraph_window",
            chunk_modes=("window",),
            window_size=max_tokens,
            window_overlap=max_tokens // 5,
            max_chunk_tokens=max_tokens,
            chunk_tiers=("window",),
        )
        window_results = _window_chunks(paragraph_stripped, window_policy)
        chunks.extend(text for _, text in window_results)

    logger.debug(f"Semantic chunking produced {len(chunks)} chunks from {len(paragraphs)} paragraphs")
    return chunks


def _window_chunks(text: str, policy: ChunkingPolicy) -> List[tuple[str, str]]:
    """
    Split text into overlapping windows using ACTUAL token count (tiktoken).

    CRITICAL: This function now uses tiktoken for accurate token counting.
    The policy window_size and window_overlap are in ACTUAL TOKEN units.
    """
    if not text.strip():
        return []

    # Use policy settings (now in ACTUAL token units)
    target_window_tokens = max(policy.window_size, 100)  # Minimum 100 tokens
    target_overlap_tokens = max(policy.window_overlap, 10)  # Minimum 10 tokens

    # Log if window seems unusually large
    if target_window_tokens > 3000:
        logger.warning(
            f"Window size {target_window_tokens} tokens is very large (>3000). "
            f"This may indicate policy misconfiguration."
        )

    # Split into words for easier manipulation while respecting token boundaries
    words = text.split()
    if not words:
        return []

    windows: List[tuple[str, str]] = []
    word_index = 0

    while word_index < len(words):
        # Build window by adding words until we reach target token count
        window_words = []
        window_token_count = 0

        for i in range(word_index, len(words)):
            test_window = " ".join(words[word_index:i+1])
            test_tokens = _count_tokens(test_window)

            if test_tokens > target_window_tokens and window_words:
                # We've exceeded the limit, use what we have
                break

            window_words = words[word_index:i+1]
            window_token_count = test_tokens

            # If we've consumed all words, we're done
            if i == len(words) - 1:
                break

        if not window_words:
            # Edge case: single word exceeds limit, take it anyway
            window_words = [words[word_index]]
            window_token_count = _count_tokens(words[word_index])
            word_index += 1
        else:
            # Calculate overlap in words
            overlap_words = 0
            if word_index + len(window_words) < len(words):
                # Not the last window - calculate overlap
                for j in range(min(len(window_words), 50)):  # Check up to 50 words for overlap
                    overlap_test = " ".join(window_words[-(j+1):])
                    overlap_tokens = _count_tokens(overlap_test)
                    if overlap_tokens >= target_overlap_tokens:
                        overlap_words = j + 1
                        break

                # Move index forward by window size minus overlap
                word_index += len(window_words) - overlap_words
            else:
                # Last window - no overlap needed
                word_index = len(words)

        window_text = " ".join(window_words)
        windows.append(("window", window_text))

        # Safety: prevent infinite loops
        if word_index <= 0 or (len(windows) > 1 and word_index == len(words)):
            break

    actual_token_count = _count_tokens(text)
    logger.debug(
        f"Created {len(windows)} window chunks from {actual_token_count} tokens "
        f"(target_window={target_window_tokens}, target_overlap={target_overlap_tokens})"
    )
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
