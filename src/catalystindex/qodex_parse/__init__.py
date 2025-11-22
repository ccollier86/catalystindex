"""
Qodex-Parse - Universal Document Parsing Library

A modern, production-ready library for parsing PDF, DOCX, HTML, and Markdown
documents with rich metadata extraction. Built for RAG applications, document
viewers, research tools, and knowledge bases.

Quick Start (Zero Config):
    ```python
    from qodex_parse import parse

    # Works immediately - no configuration required!
    doc = parse("document.pdf")  # Auto-detects FREE mode

    # Access parsed content
    for chunk in doc.chunks:
        print(chunk.text)  # Markdown-formatted text
        if chunk.type == "table":
            print(chunk.table.markdown)  # Table as Markdown
    ```

With OpenAI (Premium Features):
    ```python
    # Explicit API key
    doc = parse("document.pdf", openai_key="sk-...")

    # Or from environment (recommended)
    export OPENAI_API_KEY="sk-..."
    doc = parse("document.pdf")  # Auto-upgrades to FULL mode

    # Access semantic features
    for chunk in doc.chunks:
        print(chunk.semantic.keywords)  # LLM-generated keywords
        print(chunk.semantic.neighbors)  # Similar chunks
    ```

Key Features:
    ✅ **Multi-format support**: PDF, DOCX, HTML, Markdown
    ✅ **Markdown conversion**: Structure-preserving text extraction
    ✅ **Smart heading detection**: Multi-signal algorithm (7 signals)
    ✅ **Table extraction**: Cell-level structure with Markdown export
    ✅ **Image extraction**: With bounding box metadata
    ✅ **Spatial metadata**: Reading order, bounding boxes, pointers
    ✅ **Semantic layer**: Embeddings and nearest neighbors (requires OpenAI)
    ✅ **Enhanced metadata**: LLM keywords and questions (requires OpenAI)
    ✅ **Layout detection**: Multi-column, academic papers, complex layouts
    ✅ **Chunking control**: Semantic or fixed, with size enforcement
    ✅ **FREE basic mode**: No API key required for structure extraction

Processing Modes:
    auto (default)
        Auto-detect based on API key availability. If OpenAI key is found
        (parameter or environment), uses FULL mode. Otherwise uses basic/FREE.

    full
        Complete feature set - all metadata layers enabled. Requires OpenAI.
        Includes: spatial + semantic + enhanced metadata.

    basic (FREE)
        Structure extraction only - no OpenAI required.
        Includes: text, headings, tables, images, spatial metadata.
        Excludes: embeddings, semantic layer, LLM enhancement.

    spatial
        Structure + embeddings for vector search. Requires OpenAI.
        Good for search without LLM enhancement.

    semantic
        Structure + semantic layer (neighbors, clusters). Requires OpenAI.
        Includes embeddings but excludes LLM keywords/questions.

Examples:
    Basic usage (zero config):
        >>> from qodex_parse import parse
        >>> doc = parse("document.pdf")
        >>> print(len(doc.chunks))
        142

    Batch processing:
        >>> from qodex_parse import parse_batch
        >>> docs = parse_batch(
        ...     ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
        ...     openai_key="sk-..."
        ... )
        >>> print(sum(len(doc.chunks) for doc in docs))
        486

    Custom configuration:
        >>> doc = parse(
        ...     "paper.pdf",
        ...     openai_key="sk-...",
        ...     layout_mode="academic",  # 2-column detection
        ...     max_chunk_tokens=2000,   # Larger chunks
        ...     qr_compatible=True,      # Size enforcement
        ... )

    Size-constrained (mobile/embedded):
        >>> doc = parse(
        ...     "document.pdf",
        ...     qr_compatible=True,  # Enforces 800 token max, 2900 byte limit
        ...     chunking_strategy="semantic"
        ... )

For More Information:
    - API Reference: See api.py for detailed parameter documentation
    - Chunking Guide: See chunking.py for size control options
    - Converters: See converters/ for text-to-Markdown details
    - Documentation: https://qodex-parse.readthedocs.io
"""

# Main API (what most users need)
from .api import (
    parse,
    parse_batch,
    parse_free,
    parse_premium,
    help as api_help,
    MODES,
)

# Advanced API (for power users)
# TODO: Uncomment when implemented
# from .core.parser import QodexParser, ParserConfig
# from .core.models import QodexDocument, QodexChunk, ChunkType
# from .core.schemas import (
#     Bbox,
#     SpatialMetadata,
#     SemanticMetadata,
#     EnhancedMetadata,
# )

# Converters
from .converters import TextToMarkdownConverter

__version__ = "1.0.0"

# Public API
__all__ = [
    # Simple API (most users)
    "parse",
    "parse_batch",
    "parse_free",
    "parse_premium",
    "api_help",
    "MODES",

    # Advanced API (power users)
    # "QodexParser",
    # "ParserConfig",
    # "QodexDocument",
    # "QodexChunk",
    # "ChunkType",
    # "Bbox",
    # "SpatialMetadata",
    # "SemanticMetadata",
    # "EnhancedMetadata",

    # Converters
    "TextToMarkdownConverter",
]


def _print_welcome():
    """Print welcome message when imported."""
    import sys
    if hasattr(sys, 'ps1'):  # Interactive mode
        print()
        print("=" * 60)
        print("QODEX-PARSE - Universal Document Parsing")
        print("=" * 60)
        print()
        print("Quick Start:")
        print('  doc = parse("document.pdf", openai_key="sk-...")')
        print()
        print("FREE mode:")
        print('  doc = parse("document.pdf", mode="basic")')
        print()
        print("For help: api_help()")
        print("=" * 60)
        print()


# Show welcome in interactive mode
# _print_welcome()  # Uncomment to enable welcome message
