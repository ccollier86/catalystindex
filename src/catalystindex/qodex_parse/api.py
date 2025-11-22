"""
Qodex-Parse Main API

This module provides the primary user-facing API for qodex-parse, a universal
document parsing library that extracts text, tables, images, and builds rich
metadata layers for RAG applications, document viewers, and research tools.

The API is designed to be dead simple by default while supporting advanced
use cases through progressive complexity:

Basic Usage (Zero Config):
    ```python
    from qodex_parse import parse

    # Works immediately - no configuration needed!
    doc = parse("document.pdf")  # Auto-detects FREE mode
    ```

With OpenAI (Premium Features):
    ```python
    # Provide key explicitly
    doc = parse("document.pdf", openai_key="sk-...")

    # Or set environment variable
    export OPENAI_API_KEY="sk-..."
    doc = parse("document.pdf")  # Auto-upgrades to FULL mode
    ```

Advanced Usage:
    ```python
    doc = parse(
        "document.pdf",
        openai_key="sk-...",
        layout_mode="academic",  # Optimize for 2-column papers
        max_chunk_tokens=2000,   # Larger chunks for context
        qr_compatible=True,      # Enforce size limits
    )
    ```

Features:
    - PDF, DOCX, HTML, Markdown support
    - Markdown text extraction with structure preservation
    - Advanced heading detection (multi-signal algorithm)
    - Table and image extraction (Docling-powered)
    - Spatial metadata (bounding boxes, reading order)
    - Semantic layer (embeddings, nearest neighbors)
    - Enhanced metadata (LLM-generated keywords + questions)
    - Multi-column layout detection
    - FREE basic mode (no API key required)
    - Smart auto-detection of API keys and modes

Architecture:
    qodex-parse uses a layered approach:

    1. **Extraction Layer**: Docling + OpenParse for robust PDF parsing
    2. **Conversion Layer**: Text-to-Markdown with heading detection
    3. **Chunking Layer**: Semantic or fixed chunking with size control
    4. **Spatial Layer**: Bounding boxes, pointers, reading order
    5. **Semantic Layer**: Embeddings, nearest neighbors (requires OpenAI)
    6. **Enhanced Layer**: LLM keywords and questions (requires OpenAI)

Design Philosophy:
    - **Zero friction**: Just `parse("file.pdf")` works
    - **Progressive complexity**: Simple for beginners, powerful for experts
    - **Sensible defaults**: Do the right thing automatically
    - **Clear errors**: Tell users what's wrong and how to fix it
    - **No surprises**: Predictable, well-documented behavior
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import actual modules
from .core.parser import QodexParser
from .core.schemas import QodexDocument


def parse(
    file_path: str,
    openai_key: Optional[str] = None,
    mode: str = "auto",
    layout_mode: str = "auto",
    tables: bool = True,
    images: bool = True,
    headings: bool = True,
    # Chunking parameters
    max_chunk_tokens: int = 1200,
    target_chunk_tokens: int = 800,
    chunk_overlap: int = 100,
    chunking_strategy: str = "semantic",
    qr_compatible: bool = False,
):
    """
    Parse a document with qodex-parse - dead simple by default!

    This is the main entry point to qodex-parse. By default, it automatically
    detects whether you have an OpenAI API key available and chooses the
    appropriate processing mode:

    - If OpenAI key is found (parameter or environment), uses FULL mode
    - If no key is available, falls back to FREE basic mode

    This means you can just call `parse("file.pdf")` and it will work!

    Args:
        file_path: Path to PDF, DOCX, HTML, or Markdown file to parse.

        openai_key: OpenAI API key for premium features (embeddings, LLM).
            If not provided, automatically checks OPENAI_API_KEY environment
            variable. If no key is found, falls back to FREE basic mode.

        mode: Processing mode - controls which features are enabled.
            - "auto" (default): Auto-detect based on API key availability
            - "full": Everything! Spatial + semantic + enhanced (requires key)
            - "basic": FREE - Structure only (no OpenAI needed)
            - "spatial": Structure + embeddings (requires key)
            - "semantic": Structure + semantic layer (requires key)

        layout_mode: Page layout detection mode.
            - "auto" (default): Auto-detect layout complexity
            - "simple": Single column, top-to-bottom reading order
            - "multi_column": Multi-column detection (newspapers, academic)
            - "academic": Academic paper layout (2-col with abstract)

        tables: Whether to extract and parse tables (default: True).
            When enabled, uses Docling's table extraction with cell-level
            structure and Markdown formatting.

        images: Whether to extract and export images (default: True).
            Extracts images with bounding box metadata and optional base64
            encoding or file export.

        headings: Whether to detect headings (default: True).
            Uses multi-signal heading detection algorithm combining font size,
            weight, position, and content analysis.

        max_chunk_tokens: Maximum tokens allowed per chunk (default: 1200).
            Chunks exceeding this limit will be split at semantic boundaries.
            Automatically reduced to 800 when qr_compatible=True.

        target_chunk_tokens: Target chunk size in tokens (default: 800).
            The chunker aims for this size when possible, balancing context
            preservation with chunk manageability.

        chunk_overlap: Token overlap between consecutive chunks (default: 100).
            Helps preserve context across chunk boundaries. Approximately
            1-2 sentences of overlap.

        chunking_strategy: Strategy for splitting text into chunks.
            - "semantic" (default): Respects paragraph/sentence boundaries,
              never splits headings or tables, preserves structure
            - "fixed": Fixed-size chunks with overlap, faster but may split
              mid-thought

        qr_compatible: Enforce strict size limits for constrained systems.
            When True:
            - Automatically sets max_chunk_tokens=800
            - Validates each chunk fits within 2900 byte limit
            - Useful for mobile apps, embedded systems, video encoding, or
              any application requiring strict size guarantees

    Returns:
        QodexDocument: Parsed document with chunks and metadata layers.
            The document contains:
            - chunks: List of text chunks with metadata
            - spatial_graph: Spatial relationships (prev/next/above/below)
            - semantic_index: Nearest neighbor index (if mode != "basic")
            - enhanced_metadata: LLM keywords and questions (if mode == "full")

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        ValueError: If invalid mode or missing API key for paid modes.
        NotImplementedError: Currently raised as parser is not yet implemented.

    Examples:
        Basic usage (zero config):
            >>> from qodex_parse import parse
            >>> doc = parse("document.pdf")  # Auto-detects FREE mode
            >>> print(len(doc.chunks))
            142

        With OpenAI key (premium features):
            >>> doc = parse("document.pdf", openai_key="sk-...")
            >>> # Or set OPENAI_API_KEY environment variable
            >>> doc = parse("document.pdf")  # Auto-detects FULL mode

        Custom configuration:
            >>> doc = parse(
            ...     "document.pdf",
            ...     openai_key="sk-...",
            ...     layout_mode="academic",
            ...     max_chunk_tokens=2000,
            ...     chunking_strategy="semantic"
            ... )

        Size-constrained mode:
            >>> doc = parse(
            ...     "document.pdf",
            ...     openai_key="sk-...",
            ...     qr_compatible=True  # Enforces strict size limits
            ... )

        Optimized for speed:
            >>> doc = parse(
            ...     "simple_doc.pdf",
            ...     mode="basic",  # No embeddings/LLM
            ...     layout_mode="simple",  # Skip layout detection
            ...     tables=False,  # Skip table extraction
            ...     images=False   # Skip image extraction
            ... )
    """
    # === Step 1: Resolve OpenAI API key ===
    # Try to get key from environment if not explicitly provided
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            logger.debug("Using OpenAI API key from environment")

    # === Step 2: Auto-detect processing mode ===
    # If mode is "auto", determine based on API key availability
    if mode == "auto":
        if openai_key:
            mode = "full"
            logger.info("✓ OpenAI key detected - using FULL mode (all features)")
        else:
            mode = "basic"
            logger.info(
                "ℹ No OpenAI key found - using FREE mode (structure only)\n"
                "  To enable premium features:\n"
                "  1. Pass openai_key parameter, or\n"
                "  2. Set OPENAI_API_KEY environment variable"
            )

    # === Step 3: Security - Path traversal protection ===
    # Convert to Path and resolve to absolute path
    file_path = Path(file_path)

    # Security: Prevent path traversal attacks
    try:
        # Resolve to absolute path (handles .., symlinks, etc.)
        file_path = file_path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise ValueError(
            f"Invalid file path: {file_path}\n"
            f"Error: {e}"
        )

    # Security: Check for dangerous path patterns
    path_str = str(file_path)
    if ".." in path_str or path_str.startswith("/etc") or path_str.startswith("/sys"):
        raise ValueError(
            f"Potentially unsafe file path detected: {file_path}\n"
            f"For security, paths with '..' or system directories are not allowed."
        )

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Please check the path and try again."
        )

    # Security: Validate file type by extension
    supported_extensions = {'.pdf', '.docx', '.html', '.htm', '.md', '.markdown', '.txt'}
    if file_path.suffix.lower() not in supported_extensions:
        logger.warning(
            f"Unsupported file extension: {file_path.suffix}\n"
            f"Supported: {', '.join(sorted(supported_extensions))}\n"
            f"Proceeding anyway, but parsing may fail."
        )

    # Security: Check file size (prevent DoS via huge files)
    max_file_size = 500 * 1024 * 1024  # 500MB limit
    file_size = file_path.stat().st_size
    if file_size > max_file_size:
        raise ValueError(
            f"File too large: {file_size / (1024*1024):.1f}MB\n"
            f"Maximum supported size: {max_file_size / (1024*1024):.0f}MB\n"
            f"Large files should be processed in chunks or split into smaller documents."
        )

    # === Step 4: Validate mode ===
    valid_modes = ["auto", "full", "basic", "spatial", "semantic"]
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}\n\n"
            f"Mode descriptions:\n"
            f"  - 'auto': Auto-detect based on API key (recommended)\n"
            f"  - 'full': Everything! Spatial + semantic + enhanced\n"
            f"  - 'basic': FREE - Structure only (no OpenAI)\n"
            f"  - 'spatial': Structure + embeddings\n"
            f"  - 'semantic': Structure + semantic layer"
        )

    # === Step 5: Check API key requirements for paid modes ===
    if mode in ["full", "spatial", "semantic"] and not openai_key:
        raise ValueError(
            f"Mode '{mode}' requires an OpenAI API key.\n\n"
            f"Options:\n"
            f"  1. Provide key: parse(file_path, openai_key='sk-...')\n"
            f"  2. Set environment: export OPENAI_API_KEY='sk-...'\n"
            f"  3. Use FREE mode: parse(file_path, mode='basic')"
        )

    # === Step 6: Determine feature flags based on mode ===
    # These flags control which processing layers are enabled
    if mode == "basic":
        # FREE mode - no OpenAI features
        # Extract structure (text, headings, tables) but no embeddings/LLM
        spatial = True   # Basic spatial metadata (bbox, pointers)
        semantic = False  # No embeddings or semantic search
        enhanced = False  # No LLM-generated metadata

    elif mode == "spatial":
        # Structure + embeddings only
        # Good for vector search without LLM enhancement
        spatial = True   # Full spatial metadata
        semantic = False  # No semantic layer (neighbors, clusters)
        enhanced = False  # No LLM enhancement

    elif mode == "semantic":
        # Structure + semantic layer (no enhanced)
        # Includes embeddings and nearest neighbors, but no LLM keywords
        spatial = True   # Full spatial metadata
        semantic = True  # Semantic layer (embeddings, neighbors)
        enhanced = False  # No LLM keywords/questions

    else:  # mode == "full"
        # EVERYTHING!
        # Complete feature set with all metadata layers
        spatial = True   # Full spatial metadata
        semantic = True  # Semantic layer
        enhanced = True  # LLM-generated keywords and questions

    # === Step 7: Validate layout mode ===
    valid_layout_modes = ["auto", "simple", "multi_column", "academic"]
    if layout_mode not in valid_layout_modes:
        raise ValueError(
            f"Invalid layout_mode '{layout_mode}'. "
            f"Must be one of: {', '.join(valid_layout_modes)}"
        )

    # === Step 8: Adjust chunk sizes for QR compatibility ===
    if qr_compatible and max_chunk_tokens > 800:
        logger.info("QR-compatible mode: reducing max_chunk_tokens to 800")
        max_chunk_tokens = 800

    # === Step 9: Create parser and process document ===
    logger.info(f"Parsing {file_path.name} with mode='{mode}'")

    # Determine which extraction features to use based on mode
    tables_extraction = "pymupdf" if tables else "none"
    images_extraction = "base64" if images else "none"

    try:
        # Create parser with configured options
        parser = QodexParser(
            pdf_path=str(file_path),
            # Extraction options
            tables=tables_extraction,
            images=images_extraction,
            # Enhancement options
            headings=headings,
            semantic=semantic,
            enhanced=enhanced,
            # API key
            openai_key=openai_key,
        )

        # Parse the document
        doc = parser.parse()

        logger.info(f"✓ Parsed {len(doc.chunks)} chunks from {doc.num_pages} pages")
        return doc

    except Exception as e:
        # Provide helpful error messages
        logger.error(f"Failed to parse {file_path.name}: {e}")
        raise RuntimeError(
            f"Parsing failed for {file_path.name}\n"
            f"Error: {e}\n\n"
            f"Troubleshooting:\n"
            f"  - Check if file is corrupted or encrypted\n"
            f"  - Try with simpler mode: mode='basic'\n"
            f"  - Reduce features: tables=False, images=False\n"
            f"  - Check OpenAI API key if using semantic/enhanced features"
        ) from e


def parse_batch(
    file_paths: List[str],
    openai_key: Optional[str] = None,
    mode: str = "auto",
    layout_mode: str = "auto",
    cross_document_semantics: bool = False,
    # Chunking parameters
    max_chunk_tokens: int = 1200,
    target_chunk_tokens: int = 800,
    qr_compatible: bool = False,
) -> List:
    """
    Parse multiple documents in batch.

    Parses multiple files with the same configuration. Optionally builds
    semantic relationships ACROSS documents for multi-document RAG systems.

    Args:
        file_paths: List of file paths to parse.

        openai_key: OpenAI API key. Auto-detects from environment if not provided.

        mode: Processing mode (same options as `parse()`).
            Default "auto" detects based on API key availability.

        layout_mode: Layout detection mode (same options as `parse()`).

        cross_document_semantics: Build semantic links ACROSS documents.
            When True, semantic neighbors can point to chunks in OTHER documents.
            Requires mode="full" or "semantic" (embeddings needed).
            Useful for multi-document knowledge bases where documents reference
            each other.

        max_chunk_tokens: Maximum tokens per chunk (default: 1200).

        target_chunk_tokens: Target chunk size (default: 800).

        qr_compatible: Enforce strict size limits (default: False).

    Returns:
        List[QodexDocument]: List of parsed documents with metadata.

    Raises:
        FileNotFoundError: If any file doesn't exist.
        ValueError: If invalid parameters or missing API key.
        NotImplementedError: For cross_document_semantics (not yet supported).

    Examples:
        Parse multiple files:
            >>> from qodex_parse import parse_batch
            >>> docs = parse_batch(
            ...     ["doc1.pdf", "doc2.pdf", "doc3.docx"],
            ...     openai_key="sk-..."
            ... )
            >>> print(len(docs))
            3

        With cross-document semantics:
            >>> docs = parse_batch(
            ...     ["intro.pdf", "advanced.pdf"],
            ...     openai_key="sk-...",
            ...     cross_document_semantics=True
            ... )
            >>> # Now chunks from intro.pdf can have neighbors in advanced.pdf!

        Size-constrained batch:
            >>> docs = parse_batch(
            ...     ["book1.pdf", "book2.pdf"],
            ...     openai_key="sk-...",
            ...     qr_compatible=True
            ... )
    """
    # Validate cross-document semantics requirements
    if cross_document_semantics:
        # Cross-doc semantics requires embeddings
        if mode == "basic":
            raise ValueError(
                "cross_document_semantics=True requires mode='full' or 'semantic'\n"
                "(Basic mode has no embeddings/semantic layer)"
            )

        # Try to get key from environment if not provided
        if not openai_key:
            openai_key = os.getenv("OPENAI_API_KEY")

        if not openai_key:
            raise ValueError(
                "cross_document_semantics=True requires OpenAI API key\n"
                "Either provide openai_key parameter or set OPENAI_API_KEY env var"
            )

    # Validate all files exist before starting
    for file_path in file_paths:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    if not cross_document_semantics:
        # Simple case: parse each file independently
        # This is the common path - just process files one by one
        docs = []
        for file_path in file_paths:
            doc = parse(
                file_path,
                openai_key=openai_key,
                mode=mode,
                layout_mode=layout_mode,
                max_chunk_tokens=max_chunk_tokens,
                target_chunk_tokens=target_chunk_tokens,
                qr_compatible=qr_compatible,
            )
            docs.append(doc)
        return docs

    else:
        # Complex case: parse all, then build cross-doc semantics
        # This requires batch processor (separate layer)
        raise NotImplementedError(
            "cross_document_semantics requires batch processor.\n\n"
            "The batch processor is a SEPARATE LAYER (not part of qodex-parse core).\n"
            "It will:\n"
            "  1. Parse all documents independently\n"
            "  2. Build unified semantic index across documents\n"
            "  3. Link related chunks across document boundaries\n\n"
            "See BATCH_PROCESSING.md for architecture details."
        )


# === Convenience Functions ===


def parse_free(file_path: str, **kwargs):
    """
    Parse in FREE mode (no OpenAI needed).

    Convenience function for basic parsing without OpenAI features.
    Equivalent to: `parse(file_path, mode="basic")`

    Args:
        file_path: Path to document.
        **kwargs: Additional arguments passed to `parse()`.

    Returns:
        QodexDocument with structure (no semantic/enhanced metadata).

    Example:
        >>> from qodex_parse import parse_free
        >>> doc = parse_free("document.pdf")
        >>> # Same as: parse("document.pdf", mode="basic")
    """
    return parse(file_path, mode="basic", **kwargs)


def parse_premium(file_path: str, openai_key: str, **kwargs):
    """
    Parse with FULL PREMIUM experience.

    Convenience function for full-featured parsing with all metadata layers.
    Equivalent to: `parse(file_path, openai_key=key, mode="full")`

    Args:
        file_path: Path to document.
        openai_key: OpenAI API key.
        **kwargs: Additional arguments passed to `parse()`.

    Returns:
        QodexDocument with ALL metadata layers.

    Example:
        >>> from qodex_parse import parse_premium
        >>> doc = parse_premium("document.pdf", openai_key="sk-...")
        >>> # Same as: parse("document.pdf", openai_key="sk-...", mode="full")
    """
    return parse(file_path, openai_key=openai_key, mode="full", **kwargs)


# === Mode Reference ===

MODES = {
    "auto": "Auto-detect based on API key availability (recommended)",
    "full": "Everything! Spatial + semantic + enhanced (requires OpenAI)",
    "basic": "FREE - Structure only (no OpenAI needed)",
    "spatial": "Structure + embeddings (requires OpenAI)",
    "semantic": "Structure + semantic layer (requires OpenAI)",
}


def help():
    """Print quick reference guide for qodex-parse API."""
    print("=" * 70)
    print("QODEX-PARSE - Universal Document Parsing Library")
    print("=" * 70)
    print()
    print("BASIC USAGE (Zero Config):")
    print()
    print("  from qodex_parse import parse")
    print()
    print('  doc = parse("document.pdf")  # Auto-detects FREE mode')
    print()
    print("-" * 70)
    print("WITH OPENAI (Premium Features):")
    print()
    print('  doc = parse("document.pdf", openai_key="sk-...")')
    print()
    print("  # Or set environment variable:")
    print('  export OPENAI_API_KEY="sk-..."')
    print('  doc = parse("document.pdf")  # Auto-upgrades to FULL mode')
    print()
    print("-" * 70)
    print("MODES:")
    print()
    for mode, description in MODES.items():
        print(f"  {mode:10} - {description}")
    print()
    print("-" * 70)
    print("OPTIONS:")
    print()
    print("  layout_mode='auto'           - Layout detection (auto/simple/multi_column/academic)")
    print("  tables=True/False            - Extract tables (default: True)")
    print("  images=True/False            - Extract images (default: True)")
    print("  headings=True/False          - Detect headings (default: True)")
    print()
    print("CHUNKING:")
    print()
    print("  max_chunk_tokens=1200        - Maximum tokens per chunk")
    print("  target_chunk_tokens=800      - Target chunk size")
    print("  chunking_strategy='semantic' - 'semantic' or 'fixed'")
    print("  qr_compatible=True           - Enforce strict size limits")
    print()
    print("=" * 70)
    print("For full documentation: https://qodex-parse.readthedocs.io")
    print("=" * 70)
