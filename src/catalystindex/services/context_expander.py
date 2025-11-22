"""NVIDIA-style context expansion service.

This service implements the NVIDIA RAG pattern:
1. Retrieve small, precise chunks for accuracy
2. Expand to full pages for LLM context
3. Maintains bidirectional chunk↔page pointers

Benefits:
- Small chunks give precise retrieval
- Full pages give rich context to LLMs
- Best of both worlds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ..models.common import ChunkRecord, PageRecord, RetrievalResult, Tenant
from ..storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ExpandedContext:
    """Result of context expansion with both chunks and pages."""

    chunks: List[ChunkRecord]  # Original retrieved chunks
    pages: List[PageRecord]  # Expanded full pages
    chunk_scores: Dict[str, float]  # chunk_id -> relevance score
    page_ids: List[str]  # Ordered list of page IDs for deduplication


class ContextExpander:
    """Expand retrieved chunks to full pages for LLM context."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def expand_to_pages(
        self,
        *,
        tenant: Tenant,
        retrieval_results: List[RetrievalResult],
        knowledge_base_id: str,
    ) -> ExpandedContext:
        """
        Expand retrieved chunks to full pages.

        Args:
            tenant: Tenant context
            retrieval_results: Retrieved chunks with scores
            knowledge_base_id: Knowledge base identifier

        Returns:
            ExpandedContext with chunks and expanded pages
        """
        if not retrieval_results:
            return ExpandedContext(
                chunks=[],
                pages=[],
                chunk_scores={},
                page_ids=[],
            )

        # Extract chunks and scores
        chunks = [result.chunk for result in retrieval_results]
        chunk_scores = {result.chunk.chunk_id: result.score for result in retrieval_results}

        # Collect all page IDs from chunks (maintaining order and avoiding duplicates)
        page_ids_ordered: List[str] = []
        seen_page_ids: Set[str] = set()

        for chunk in chunks:
            # Get page ID from chunk metadata
            page_id = chunk.metadata.get("page_id")
            if page_id and page_id not in seen_page_ids:
                page_ids_ordered.append(page_id)
                seen_page_ids.add(page_id)

        if not page_ids_ordered:
            logger.warning("No page IDs found in chunk metadata - cannot expand to pages")
            return ExpandedContext(
                chunks=chunks,
                pages=[],
                chunk_scores=chunk_scores,
                page_ids=[],
            )

        # Fetch page records from vector store
        try:
            pages = self._fetch_pages(
                tenant=tenant,
                page_ids=page_ids_ordered,
                knowledge_base_id=knowledge_base_id,
            )

            logger.info(
                f"Expanded {len(chunks)} chunks to {len(pages)} pages "
                f"(from {len(page_ids_ordered)} unique page IDs)"
            )

            return ExpandedContext(
                chunks=chunks,
                pages=pages,
                chunk_scores=chunk_scores,
                page_ids=page_ids_ordered,
            )

        except Exception as exc:
            logger.error(f"Failed to expand chunks to pages: {exc}", exc_info=True)
            return ExpandedContext(
                chunks=chunks,
                pages=[],
                chunk_scores=chunk_scores,
                page_ids=page_ids_ordered,
            )

    def _fetch_pages(
        self,
        *,
        tenant: Tenant,
        page_ids: List[str],
        knowledge_base_id: str,
    ) -> List[PageRecord]:
        """
        Fetch page records by page IDs from vector store.
        """
        try:
            # Use "default" track for pages (same as chunks)
            pages = self._vector_store.get_pages(
                tenant=tenant,
                page_ids=page_ids,
                track="default",
                knowledge_base_id=knowledge_base_id,
            )
            return pages
        except Exception as exc:
            logger.error(f"Failed to fetch pages from vector store: {exc}", exc_info=True)
            return []

    def format_for_llm(self, expanded: ExpandedContext) -> str:
        """
        Format expanded context for LLM consumption.

        Creates a readable format with:
        - Page headers showing page numbers
        - Full page text
        - Separator between pages
        """
        if not expanded.pages:
            # Fallback: return chunk text if no pages available
            return "\n\n---\n\n".join(chunk.text for chunk in expanded.chunks)

        # Format pages with headers
        formatted_pages: List[str] = []
        for page in expanded.pages:
            header = f"=== Page {page.page_number} ==="
            formatted_pages.append(f"{header}\n\n{page.text}")

        return "\n\n---\n\n".join(formatted_pages)


__all__ = ["ContextExpander", "ExpandedContext"]
