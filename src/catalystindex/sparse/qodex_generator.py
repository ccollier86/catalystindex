from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, Set

from catalystindex.models.common import ChunkRecord


# English stop words for sparse vector filtering (reduces hot keys by ~40%)
DEFAULT_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "will", "with", "the", "this", "but", "they", "have",
    "had", "what", "when", "where", "who", "which", "why", "how",
})


class QodexSparseGenerator:
    """Generates sparse vectors from qodex-parse metadata for Qdrant hybrid search.

    Leverages qodex-enhanced metadata (keywords, search_terms) to build weighted
    sparse vectors that prioritize LLM-extracted terms over raw text tokens.

    Weighting Strategy:
        - search_terms (query-optimized): 2.0x weight
        - keywords (LLM-extracted): 1.5x weight
        - text tokens (TF baseline): 1.0x weight

    Compatible with Qdrant's named sparse vectors using SHA1 hashing for indices.
    """

    def __init__(
        self,
        search_term_weight: float = 2.0,
        keyword_weight: float = 1.5,
        text_weight: float = 1.0,
        max_sparse_dim: int = 131071,
        stop_words: Set[str] | None = None,
        enable_stop_word_filtering: bool = True,
    ):
        """Initialize the sparse vector generator.

        Args:
            search_term_weight: Weight multiplier for qodex search_terms (default: 2.0)
            keyword_weight: Weight multiplier for qodex keywords (default: 1.5)
            text_weight: Weight multiplier for text tokens (default: 1.0)
            max_sparse_dim: Maximum sparse vector dimension for hash modulo (default: 131071)
            stop_words: Custom stop word set (default: DEFAULT_STOP_WORDS)
            enable_stop_word_filtering: Whether to filter stop words (default: True, 5-15% speedup)
        """
        self.search_term_weight = search_term_weight
        self.keyword_weight = keyword_weight
        self.text_weight = text_weight
        self.max_sparse_dim = max_sparse_dim
        self.stop_words = stop_words if stop_words is not None else DEFAULT_STOP_WORDS
        self.enable_stop_word_filtering = enable_stop_word_filtering

    def generate(self, chunk: ChunkRecord) -> Dict[int, float] | None:
        """Generate sparse vector from chunk text and qodex metadata.

        Args:
            chunk: ChunkRecord with text and metadata (may include keywords, search_terms)

        Returns:
            Sparse vector as {index: weight} dict, or None if chunk has no content
        """
        if not chunk.text:
            return None

        # Extract qodex metadata
        metadata = chunk.metadata or {}
        search_terms = self._extract_terms(metadata.get("search_terms"))
        keywords = self._extract_terms(metadata.get("keywords"))

        # Tokenize chunk text
        text_tokens = chunk.text.lower().split()

        # Build weighted sparse vector
        sparse_vector: Dict[int, float] = {}

        # Priority 1: search_terms (query-optimized, highest weight)
        for term in search_terms:
            term_tokens = term.lower().split()
            for token in term_tokens:
                if not token:
                    continue
                index = self._hash_token(token)
                # Accumulate weight if token appears multiple times
                sparse_vector[index] = sparse_vector.get(index, 0.0) + self.search_term_weight

        # Priority 2: keywords (LLM-extracted, medium weight)
        for keyword in keywords:
            keyword_tokens = keyword.lower().split()
            for token in keyword_tokens:
                if not token:
                    continue
                index = self._hash_token(token)
                # Only add if not already weighted from search_terms
                if index not in sparse_vector:
                    sparse_vector[index] = self.keyword_weight

        # Priority 3: text tokens (TF baseline, standard weight)
        # Filter stop words from text tokens (not from qodex metadata)
        if self.enable_stop_word_filtering:
            text_tokens = [token for token in text_tokens if token not in self.stop_words]

        token_counts = Counter(text_tokens)
        max_count = max(token_counts.values()) if token_counts else 1

        for token, count in token_counts.items():
            if not token:
                continue
            index = self._hash_token(token)
            # Only add if not already weighted from qodex metadata
            if index not in sparse_vector:
                # Normalize by max count (TF normalization)
                tf_weight = (count / max_count) * self.text_weight
                sparse_vector[index] = tf_weight

        return sparse_vector if sparse_vector else None

    def generate_from_query(self, query: str) -> Dict[int, float] | None:
        """Generate sparse vector from plain text query (no metadata).

        For query-time sparse generation, only text tokenization is used since
        queries don't have qodex metadata (keywords, search_terms).

        Args:
            query: Plain text query string

        Returns:
            Sparse vector as {index: weight} dict, or None if query is empty
        """
        if not query:
            return None

        # Tokenize query text
        text_tokens = query.lower().split()

        # Filter stop words
        if self.enable_stop_word_filtering:
            text_tokens = [token for token in text_tokens if token not in self.stop_words]

        if not text_tokens:
            return None

        # Build sparse vector with TF normalization
        token_counts = Counter(text_tokens)
        max_count = max(token_counts.values()) if token_counts else 1

        sparse_vector: Dict[int, float] = {}
        for token, count in token_counts.items():
            if not token:
                continue
            index = self._hash_token(token)
            # Normalize by max count (TF normalization)
            tf_weight = (count / max_count) * self.text_weight
            sparse_vector[index] = tf_weight

        return sparse_vector if sparse_vector else None

    def _extract_terms(self, value: Any) -> list[str]:
        """Extract terms from metadata value (handles list or None).

        Args:
            value: Metadata value (List[str], str, or None)

        Returns:
            List of string terms, empty list if value is None or invalid
        """
        if value is None:
            return []
        if isinstance(value, list):
            return [str(term) for term in value if term]
        if isinstance(value, str):
            return [value] if value else []
        return []

    def _hash_token(self, token: str) -> int:
        """Hash token to sparse vector index using SHA1.

        Args:
            token: Token string to hash

        Returns:
            Integer index in range [1, max_sparse_dim]
        """
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % self.max_sparse_dim
        # Ensure index is never 0 (1-indexed for Qdrant compatibility)
        return index or 1


__all__ = ["QodexSparseGenerator"]
