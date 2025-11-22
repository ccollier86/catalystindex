"""
QodexParser - Main parser class with simple API

Simpler than OpenParse:
    from qodex_parse import QodexParser

    # Basic use
    doc = QodexParser("file.pdf").parse()

    # Advanced use
    parser = QodexParser(
        "file.pdf",
        tables="docling",
        images="base64",
        headings=True,
        semantic=True,
        openai_key="..."
    )
    doc = parser.parse()
"""

import uuid
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Tuple
from .schemas import (
    QodexDocument,
    QodexChunk,
    SpatialMetadata,
    SemanticMetadata,
    ChunkType,
    Bbox,
)
from ..extractors.headings import EnhancedHeadingDetector, HeadingHierarchyBuilder
from ..extractors.tables import TableExtractor
from ..extractors.images import ImageExtractor
from ..enrichment.spatial import SpatialGraphBuilder
from ..enrichment.semantic import SemanticLayerBuilder
from ..enrichment.enhanced import EnhancedMetadataBuilder


class QodexParser:
    """
    Main parser class - simple API for rich document parsing.

    Features:
    - Modern PyMuPDF text extraction with bounding boxes
    - Enhanced heading detection (multi-signal)
    - PyMuPDF table extraction (2025 best practice with find_tables())
    - Image export (base64 or file)
    - Spatial enrichment (prev/next/above/below/heading hierarchy)
    - Semantic layer (meaning as metadata, not reorganization)
    """

    def __init__(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        # Extraction options
        tables: Literal["pymupdf", "none"] = "pymupdf",
        images: Literal["base64", "file", "both", "none"] = "none",
        image_output_dir: Optional[str] = None,
        # Enhancement options
        headings: bool = True,
        semantic: bool = False,
        enhanced: bool = False,
        # API keys
        openai_key: Optional[str] = None,
        # Page filtering
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ):
        """
        Args:
            pdf_path: Path to PDF file
            doc_id: Optional document ID (auto-generated if not provided)
            tables: Table extraction method ("pymupdf" or "none")
            images: Image extraction method ("base64", "file", "both", "none")
            image_output_dir: Output directory for image files (required if images="file" or "both")
            headings: Enable enhanced heading detection
            semantic: Enable semantic layer (requires openai_key)
            enhanced: Enable LLM-enhanced metadata - keywords & questions (requires openai_key)
            openai_key: OpenAI API key (required for semantic/enhanced layers)
            start_page: Start page (0-indexed, inclusive)
            end_page: End page (0-indexed, inclusive)
        """
        self.pdf_path = Path(pdf_path)
        self.doc_id = doc_id or str(uuid.uuid4())
        self.tables = tables
        self.images = images
        self.image_output_dir = Path(image_output_dir) if image_output_dir else None
        self.headings = headings
        self.semantic = semantic
        self.enhanced = enhanced
        self.openai_key = openai_key
        self.start_page = start_page
        self.end_page = end_page

        # Validation
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        if self.images in ["file", "both"] and not self.image_output_dir:
            raise ValueError("image_output_dir required when images='file' or 'both'")

        if self.semantic and not self.openai_key:
            raise ValueError("openai_key required when semantic=True")

        if self.enhanced and not self.openai_key:
            raise ValueError("openai_key required when enhanced=True")

        # Initialize extractors
        self.table_extractor = None
        if self.tables != "none":
            self.table_extractor = TableExtractor(method=self.tables)

        self.image_extractor = None
        if self.images != "none":
            self.image_extractor = ImageExtractor(
                export_format=self.images,
                output_dir=self.image_output_dir,
            )

    def parse(self) -> QodexDocument:
        """
        Parse PDF and return QodexDocument with rich metadata.

        Process:
        1. Extract text blocks with PyMuPDF (get bounding boxes)
        2. Enhance heading detection
        3. Extract tables and images
        4. Convert to QodexChunks
        5. Build spatial enrichment layer (navigation relationships)
        6. Build semantic layer (if enabled)
        7. Build enhanced metadata layer (if enabled)
        8. Return QodexDocument

        Returns:
            QodexDocument with chunks and metadata
        """
        print(f"Parsing {self.pdf_path.name}...")

        # Step 1: Extract text blocks with PyMuPDF
        print("  [1/7] Extracting text blocks with PyMuPDF...")
        text_blocks = self._extract_text_blocks()

        # Step 2: Enhance heading detection
        if self.headings:
            print("  [2/7] Enhancing heading detection...")
            text_blocks = self._enhance_headings(text_blocks)
        else:
            print("  [2/7] Skipping heading enhancement")

        # Step 3: Extract tables and images
        print("  [3/7] Extracting tables and images...")
        tables_by_page, images_by_page = self._extract_tables_and_images()

        # Step 4: Convert to QodexChunks
        print("  [4/7] Converting to QodexChunks...")
        chunks = self._convert_to_chunks(text_blocks, tables_by_page, images_by_page)

        # Step 5: Build spatial enrichment layer
        print("  [5/7] Building spatial enrichment layer...")
        chunks = self._build_spatial_enrichment(chunks)

        # Step 6: Build semantic layer
        if self.semantic:
            print("  [6/7] Building semantic layer...")
            chunks = self._build_semantic_layer(chunks)
        else:
            print("  [6/7] Skipping semantic layer")

        # Step 7: Build enhanced metadata layer
        if self.enhanced:
            print("  [7/7] Building enhanced metadata (keywords & questions)...")
            chunks = self._build_enhanced_layer(chunks)
        else:
            print("  [7/7] Skipping enhanced metadata")

        # Create document
        doc = QodexDocument(
            doc_id=self.doc_id,
            filename=self.pdf_path.name,
            num_pages=self._get_num_pages(),
            chunks=chunks,
            metadata={
                "tables_enabled": self.tables != "none",
                "images_enabled": self.images != "none",
                "headings_enabled": self.headings,
                "semantic_enabled": self.semantic,
                "enhanced_enabled": self.enhanced,
            }
        )

        print(f"✓ Parsed {len(chunks)} chunks from {doc.num_pages} pages")
        return doc

    def _extract_text_blocks(self) -> List[Dict[str, Any]]:
        """
        Extract text blocks using PyMuPDF with bounding boxes.

        Returns list of text blocks, each with:
        - text: str
        - bbox: (x0, y0, x1, y1)
        - page: int
        - font_size: float
        - font_name: str
        """
        doc = fitz.open(str(self.pdf_path))
        blocks = []

        # Determine page range
        start = self.start_page if self.start_page is not None else 0
        end = self.end_page if self.end_page is not None else len(doc) - 1

        for page_num in range(start, min(end + 1, len(doc))):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height

            # Get text blocks with formatting info
            text_dict = page.get_text("dict")

            for block in text_dict["blocks"]:
                # Skip image blocks
                if block.get("type") != 0:  # 0 = text block
                    continue

                # Extract lines from block
                lines = []
                max_font_size = 0
                font_name = ""

                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        span_size = span.get("size", 0)
                        if span_size > max_font_size:
                            max_font_size = span_size
                            font_name = span.get("font", "")

                    if line_text.strip():
                        lines.append(line_text)

                # Combine lines into text
                text = "\n".join(lines).strip()
                if not text:
                    continue

                # Get bounding box
                bbox = block.get("bbox", (0, 0, 0, 0))

                # Create text block dict
                text_block = {
                    "text": text,
                    "bbox": bbox,
                    "page": page_num,
                    "page_width": page_width,
                    "page_height": page_height,
                    "font_size": max_font_size,
                    "font_name": font_name,
                    "is_bold": "Bold" in font_name or "bold" in font_name.lower(),
                }

                blocks.append(text_block)

        doc.close()
        return blocks

    def _enhance_headings(self, nodes: List[Any]) -> List[Any]:
        """Enhance heading detection"""
        detector = EnhancedHeadingDetector()
        nodes = detector.process(nodes)

        hierarchy = HeadingHierarchyBuilder()
        nodes = hierarchy.process(nodes)

        return nodes

    def _extract_tables_and_images(self) -> tuple[Dict[int, List], Dict[int, List]]:
        """Extract tables and images from all pages"""
        tables_by_page = {}
        images_by_page = {}

        # Determine page range
        start = self.start_page if self.start_page is not None else 0
        end = self.end_page if self.end_page is not None else self._get_num_pages() - 1

        for page_num in range(start, end + 1):
            # Extract tables
            if self.table_extractor:
                tables = self.table_extractor.extract_tables_from_page(
                    str(self.pdf_path),
                    page_num,
                )
                if tables:
                    tables_by_page[page_num] = tables

            # Extract images
            if self.image_extractor:
                images = self.image_extractor.extract_images_from_page(
                    str(self.pdf_path),
                    page_num,
                    self.doc_id,
                )
                if images:
                    images_by_page[page_num] = images

        return tables_by_page, images_by_page

    def _convert_to_chunks(
        self,
        text_blocks: List[Dict[str, Any]],
        tables_by_page: Dict[int, List],
        images_by_page: Dict[int, List],
    ) -> List[QodexChunk]:
        """Convert text blocks to QodexChunks"""
        chunks = []

        # Convert text blocks
        for block in text_blocks:
            chunk = self._block_to_chunk(block)
            if chunk:
                chunks.append(chunk)

        # Add table chunks
        for page, tables in tables_by_page.items():
            for table_idx, table_meta in enumerate(tables):
                chunk_id = f"{self.doc_id}_table_p{page}_{table_idx}"

                # Create chunk for table
                chunk = QodexChunk(
                    chunk_id=chunk_id,
                    doc_id=self.doc_id,
                    text=table_meta.markdown or "",
                    type=ChunkType.TABLE,
                    tokens=len(table_meta.markdown.split()) if table_meta.markdown else 0,
                    spatial=SpatialMetadata(
                        page=page,
                        bbox=Bbox(0, 0, 100, 100, page, 612, 792),  # Placeholder bbox
                    ),
                    table_meta=table_meta,
                )
                chunks.append(chunk)

        # Add image chunks
        for page, images in images_by_page.items():
            for img_idx, image_meta in enumerate(images):
                chunk_id = f"{self.doc_id}_image_p{page}_{img_idx}"

                # Create chunk for image
                chunk = QodexChunk(
                    chunk_id=chunk_id,
                    doc_id=self.doc_id,
                    text=image_meta.ocr_text or image_meta.caption or "",
                    type=ChunkType.IMAGE,
                    tokens=0,
                    spatial=SpatialMetadata(
                        page=page,
                        bbox=Bbox(0, 0, 100, 100, page, 612, 792),  # Placeholder bbox
                    ),
                    image_meta=image_meta,
                )
                chunks.append(chunk)

        return chunks

    def _block_to_chunk(self, block: Dict[str, Any]) -> Optional[QodexChunk]:
        """Convert text block dictionary to QodexChunk"""
        text = block.get("text", "").strip()
        if not text:
            return None

        # Generate chunk ID
        chunk_id = f"{self.doc_id}_{str(uuid.uuid4())[:8]}"

        # Determine chunk type (will be refined by heading detection)
        chunk_type = ChunkType.TEXT

        # Create bounding box
        bbox_tuple = block.get("bbox", (0, 0, 0, 0))
        bbox = Bbox(
            x0=bbox_tuple[0],
            y0=bbox_tuple[1],
            x1=bbox_tuple[2],
            y1=bbox_tuple[3],
            page=block.get("page", 0),
            page_width=block.get("page_width", 612),
            page_height=block.get("page_height", 792),
        )

        # Check if block was marked as heading by heading detection
        if block.get("is_heading", False):
            chunk_type = ChunkType.HEADING

        # Create spatial metadata
        spatial = SpatialMetadata(
            page=block.get("page", 0),
            bbox=bbox,
            heading_level=block.get("heading_level"),
        )

        # Create chunk
        chunk = QodexChunk(
            chunk_id=chunk_id,
            doc_id=self.doc_id,
            text=text,
            type=chunk_type,
            tokens=len(text.split()),
            spatial=spatial,
        )

        return chunk

    def _build_spatial_enrichment(self, chunks: List[QodexChunk]) -> List[QodexChunk]:
        """Build spatial enrichment layer (navigation relationships)"""
        builder = SpatialGraphBuilder()
        return builder.build_graph(chunks)

    def _build_semantic_layer(self, chunks: List[QodexChunk]) -> List[QodexChunk]:
        """Build semantic layer"""
        builder = SemanticLayerBuilder(openai_api_key=self.openai_key)
        return builder.build_semantic_layer(chunks)

    def _build_enhanced_layer(self, chunks: List[QodexChunk]) -> List[QodexChunk]:
        """Build enhanced metadata layer"""
        builder = EnhancedMetadataBuilder(openai_api_key=self.openai_key)
        return builder.build_enhanced_layer(chunks)

    def _get_num_pages(self) -> int:
        """Get number of pages in PDF"""
        try:
            import fitz
            doc = fitz.open(str(self.pdf_path))
            num_pages = len(doc)
            doc.close()
            return num_pages
        except Exception as e:
            logger.warning(f"Failed to get page count from {self.pdf_path}: {e}")
            return 0
