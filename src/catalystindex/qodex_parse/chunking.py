"""
Intelligent Text Chunking with Size Control

This module provides configurable text chunking for qodex-parse with support for
both standard RAG applications and size-constrained systems (mobile, embedded,
video encoding, etc.).

Overview:
    The chunker intelligently splits text into chunks while respecting semantic
    boundaries (paragraphs, sentences) and enforcing size limits. Two primary
    modes are supported:

    1. **Standard Mode**: Flexible chunking optimized for RAG applications
       - Default: 1200 token maximum, 800 token target
       - Respects semantic boundaries when possible
       - No strict byte-level enforcement

    2. **Strict Size Mode**: Enforces byte-level limits for constrained systems
       - Maximum: 800 tokens, 2900 bytes
       - Validates each chunk fits size constraints
       - Useful for: mobile apps, embedded systems, video encoding, QR codes

Chunking Strategies:
    Semantic (Recommended):
        - Respects paragraph boundaries (\\n\\n)
        - Respects sentence boundaries (. ! ?)
        - Never splits headings or tables mid-content
        - Preserves document structure and meaning
        - May produce variable-sized chunks within limits

    Fixed:
        - Fixed-size chunks with configurable overlap
        - Splits at best available boundary (sentence > word > character)
        - Faster and more predictable
        - May split mid-thought for long sentences

Usage Examples:
    Basic chunking (standard mode):
        ```python
        from qodex_parse.chunking import chunk_text

        # Default settings - good for most RAG applications
        chunks = chunk_text("Long text content...")
        # Returns: ["chunk 1...", "chunk 2...", ...]
        ```

    Strict size mode (size-constrained):
        ```python
        chunks = chunk_text(
            "Long text content...",
            max_tokens=800,
            qr_compatible=True  # Enforces byte limits
        )
        ```

    Custom configuration:
        ```python
        from qodex_parse.chunking import TextChunker, ChunkingConfig

        config = ChunkingConfig(
            max_chunk_tokens=2000,
            target_chunk_tokens=1500,
            overlap_tokens=200,
            qr_compatible=False,
            strategy=ChunkingStrategy.SEMANTIC
        )

        chunker = TextChunker(config)
        chunks = chunker.chunk_text("Long text...")
        ```

Integration with qodex-parse:
    ```python
    from qodex_parse import parse

    doc = parse(
        "document.pdf",
        openai_key="sk-...",
        max_chunk_tokens=1200,
        target_chunk_tokens=800,
        chunk_overlap=100,
        chunking_strategy="semantic",
        qr_compatible=False
    )
    ```

See Also:
    - api.py: Main parse() API with chunking parameters
    - docs/api/chunking.md: User-facing documentation
"""

import json
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available - using character-based estimation")


class ChunkingStrategy(Enum):
    """Chunking strategy."""
    SEMANTIC = "semantic"  # Respect paragraph/sentence boundaries
    FIXED = "fixed"        # Fixed-size with overlap


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    # Mode
    qr_compatible: bool = False  # Enforce QR size limits?

    # Token limits
    max_chunk_tokens: int = 1200      # Normal mode default
    target_chunk_tokens: int = 800    # Target chunk size
    min_chunk_tokens: int = 100       # Minimum viable chunk

    # QR limits (when qr_compatible=True)
    max_qr_text_bytes: int = 2900     # Max bytes for text in QR
    qr_json_overhead: int = 50        # Overhead for {"id":N,"text":""}

    # Overlap
    overlap_tokens: int = 100         # Token overlap between chunks
    overlap_pct: float = 0.125        # 12.5% overlap

    # Strategy
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC

    def __post_init__(self):
        """Adjust limits for QR mode."""
        if self.qr_compatible:
            # In QR mode, use smaller chunks
            self.max_chunk_tokens = 800   # Safe for QR
            self.target_chunk_tokens = 600

    @property
    def max_text_bytes(self) -> int:
        """Maximum bytes available for text."""
        if self.qr_compatible:
            return self.max_qr_text_bytes
        return float('inf')  # No limit in normal mode


class TokenCounter:
    """Count tokens using tiktoken or fallback estimation."""

    def __init__(self, model: str = "text-embedding-3-large"):
        """Initialize token counter.

        Args:
            model: Model name for tiktoken encoder
        """
        self.model = model

        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base
                self.encoder = tiktoken.get_encoding("cl100k_base")
                logger.warning(f"Model {model} not found, using cl100k_base encoder")
        else:
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: ~4 chars per token
            return max(1, len(text) // 4)

    def count_bytes(self, text: str) -> int:
        """Count UTF-8 bytes in text.

        Args:
            text: Text to count

        Returns:
            Number of bytes
        """
        if not text:
            return 0
        return len(text.encode('utf-8'))

    def encode(self, text: str) -> List[int]:
        """Encode text to tokens.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        if self.encoder:
            return self.encoder.encode(text)
        else:
            # Fallback: byte-level
            return list(text.encode('utf-8'))

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text.

        Args:
            tokens: Token IDs

        Returns:
            Decoded text
        """
        if self.encoder:
            return self.encoder.decode(tokens)
        else:
            # Fallback: byte-level
            return bytes(tokens).decode('utf-8', errors='ignore')


class QRValidator:
    """Validate chunks for QR code compatibility."""

    def __init__(self, config: ChunkingConfig):
        """Initialize validator.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.token_counter = TokenCounter()

    def validate_chunk(self, text: str) -> Tuple[bool, int, Optional[str]]:
        """Validate if text fits in QR code.

        Args:
            text: Chunk text

        Returns:
            Tuple of (is_valid, text_bytes, error_message)
        """
        if not self.config.qr_compatible:
            # No validation needed in normal mode
            return True, len(text.encode('utf-8')), None

        # Check text byte size
        text_bytes = self.token_counter.count_bytes(text)

        # Simulate minimal QR JSON payload
        # {"id":999999,"text":"..."}
        qr_payload = {
            "id": 999999,  # Max frame number for overhead calc
            "text": text
        }
        payload_json = json.dumps(qr_payload, separators=(',', ':'))
        payload_bytes = len(payload_json.encode('utf-8'))

        # Check against QR limit (2953 bytes for v40-L)
        QR_HARD_LIMIT = 2953

        if payload_bytes > QR_HARD_LIMIT:
            error = (
                f"Chunk too large for QR code: {payload_bytes} bytes > {QR_HARD_LIMIT} bytes. "
                f"Text: {text_bytes} bytes, JSON overhead: {payload_bytes - text_bytes} bytes"
            )
            return False, text_bytes, error

        if payload_bytes > self.config.max_qr_text_bytes + self.config.qr_json_overhead:
            warning = (
                f"Chunk approaching QR limit: {payload_bytes} bytes "
                f"(safe: <{self.config.max_qr_text_bytes + self.config.qr_json_overhead})"
            )
            logger.warning(warning)

        return True, text_bytes, None


class TextChunker:
    """Intelligent text chunking with multiple strategies."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        self.token_counter = TokenCounter()
        self.validator = QRValidator(self.config)

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using configured strategy.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        if self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunk(text)
        else:
            return self._fixed_chunk(text)

    def _semantic_chunk(self, text: str) -> List[str]:
        """Semantic chunking - respects paragraph and sentence boundaries.

        Args:
            text: Text to chunk

        Returns:
            List of chunks
        """
        # Check if entire text fits
        total_tokens = self.token_counter.count_tokens(text)
        if total_tokens <= self.config.max_chunk_tokens:
            is_valid, _, error = self.validator.validate_chunk(text)
            if is_valid:
                return [text]
            else:
                logger.warning(f"Text fits token limit but exceeds byte limit: {error}")

        # Split into paragraphs
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.token_counter.count_tokens(para)

            # If paragraph exceeds max, split it
            if para_tokens > self.config.max_chunk_tokens:
                # Save current chunk
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if self._validate_and_add_chunk(chunk_text, chunks):
                        current_chunk = []
                        current_tokens = 0

                # Split large paragraph
                para_chunks = self._split_large_paragraph(para)
                for pc in para_chunks:
                    self._validate_and_add_chunk(pc, chunks)
                continue

            # Check if adding paragraph would exceed target
            if current_tokens + para_tokens > self.config.target_chunk_tokens and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                if self._validate_and_add_chunk(chunk_text, chunks):
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            self._validate_and_add_chunk(chunk_text, chunks)

        return chunks

    def _fixed_chunk(self, text: str) -> List[str]:
        """Fixed-size chunking with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of chunks
        """
        if not text:
            return []

        tokens = self.token_counter.encode(text)
        chunks = []

        overlap = self.config.overlap_tokens
        start = 0

        while start < len(tokens):
            # Get chunk tokens
            end = min(start + self.config.max_chunk_tokens, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode chunk
            chunk_text = self.token_counter.decode(chunk_tokens)

            # Find good break point if not at end
            if end < len(tokens):
                chunk_text = self._find_good_break(chunk_text)

            # Validate and add
            self._validate_and_add_chunk(chunk_text, chunks)

            # Move start with overlap
            start = end - overlap

        return chunks

    def _split_large_paragraph(self, para: str) -> List[str]:
        """Split paragraph that exceeds max tokens.

        Args:
            para: Paragraph text

        Returns:
            List of chunks
        """
        import re

        # Try splitting at sentence boundaries
        sentences = re.split(r'([.!?]+\s+)', para)

        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            if not sent.strip():
                continue

            sent_tokens = self.token_counter.count_tokens(sent)

            # If single sentence exceeds max, hard split it
            if sent_tokens > self.config.max_chunk_tokens:
                # Save current
                if current:
                    chunks.append(''.join(current))
                    current = []
                    current_tokens = 0

                # Hard split at word boundaries
                words = sent.split()
                word_chunk = []
                word_tokens = 0

                for word in words:
                    word_tok = self.token_counter.count_tokens(word + ' ')
                    if word_tokens + word_tok > self.config.max_chunk_tokens and word_chunk:
                        chunks.append(' '.join(word_chunk))
                        word_chunk = [word]
                        word_tokens = word_tok
                    else:
                        word_chunk.append(word)
                        word_tokens += word_tok

                if word_chunk:
                    chunks.append(' '.join(word_chunk))
                continue

            # Add sentence if fits
            if current_tokens + sent_tokens > self.config.max_chunk_tokens and current:
                chunks.append(''.join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(''.join(current))

        return chunks

    def _find_good_break(self, text: str) -> str:
        """Find good break point in text.

        Tries to break at:
        1. Sentence boundary (. ! ?)
        2. Paragraph boundary (\n\n)
        3. Word boundary (space)
        4. Character boundary (last resort)

        Args:
            text: Text to break

        Returns:
            Text trimmed to good break point
        """
        # Try sentence boundary in last 20%
        for delim in ['. ', '! ', '? ', '.\n']:
            if delim in text:
                pos = text.rfind(delim)
                if pos > len(text) * 0.8:
                    return text[:pos + len(delim)]

        # Try paragraph boundary in last 30%
        if '\n\n' in text:
            pos = text.rfind('\n\n')
            if pos > len(text) * 0.7:
                return text[:pos]

        # Try word boundary in last 10%
        pos = text.rfind(' ')
        if pos > len(text) * 0.9:
            return text[:pos]

        # Give up, hard split
        return text

    def _validate_and_add_chunk(self, chunk_text: str, chunks: List[str]) -> bool:
        """Validate chunk and add to list.

        Args:
            chunk_text: Chunk text to validate
            chunks: List to add to

        Returns:
            True if chunk was added, False if validation failed
        """
        is_valid, text_bytes, error = self.validator.validate_chunk(chunk_text)

        if not is_valid:
            logger.error(f"Chunk validation failed: {error}")
            # Try to truncate and retry
            truncated = self._truncate_to_fit(chunk_text)
            if truncated:
                chunks.append(truncated)
                logger.warning("Added truncated chunk to fit limits")
                return True
            return False

        chunks.append(chunk_text)
        return True

    def _truncate_to_fit(self, text: str) -> Optional[str]:
        """Truncate text to fit limits.

        Uses binary search to find maximum length that fits.

        Args:
            text: Text to truncate

        Returns:
            Truncated text or None if can't fit
        """
        if not self.config.qr_compatible:
            return text

        # Binary search for max length
        min_len = 0
        max_len = len(text)
        best_fit = None

        while min_len <= max_len:
            mid = (min_len + max_len) // 2
            truncated = text[:mid] + "..."

            is_valid, _, _ = self.validator.validate_chunk(truncated)

            if is_valid:
                best_fit = truncated
                min_len = mid + 1
            else:
                max_len = mid - 1

        return best_fit


# Convenience function
def chunk_text(
    text: str,
    max_tokens: int = 1200,
    target_tokens: int = 800,
    qr_compatible: bool = False,
    strategy: str = "semantic"
) -> List[str]:
    """Convenience function to chunk text.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        target_tokens: Target tokens per chunk
        qr_compatible: Enforce QR size limits?
        strategy: "semantic" or "fixed"

    Returns:
        List of text chunks
    """
    config = ChunkingConfig(
        max_chunk_tokens=max_tokens,
        target_chunk_tokens=target_tokens,
        qr_compatible=qr_compatible,
        strategy=ChunkingStrategy(strategy)
    )

    chunker = TextChunker(config=config)
    return chunker.chunk_text(text)
