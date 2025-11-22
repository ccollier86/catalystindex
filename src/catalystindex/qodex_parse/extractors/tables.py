"""
Table extraction using PyMuPDF find_tables() (2025 best practice)

Based on:
- PyMuPDF 1.26.x: Native table finder with bbox support and markdown export
- Minimal dependencies, actively maintained, RAG-focused
"""

from typing import List, Optional, Literal, Dict, Any
import pandas as pd
from ..core.schemas import TableMetadata


class TableExtractor:
    """
    Extract tables from PDFs using PyMuPDF find_tables().

    PyMuPDF 1.26.x provides:
    - Native table finder (page.find_tables())
    - DataFrame export (table.to_pandas())
    - Markdown export (table.to_markdown())
    - Cell bounding boxes
    - Header detection
    """

    def __init__(
        self,
        method: Literal["pymupdf", "none"] = "pymupdf",
    ):
        """
        Args:
            method: Extraction method to use ("pymupdf" or "none")
        """
        self.method = method

    def extract_tables_from_page(
        self,
        pdf_path: str,
        page_num: int,
    ) -> List[TableMetadata]:
        """
        Extract tables from a specific page using PyMuPDF find_tables().

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            List of TableMetadata objects
        """
        if self.method == "none":
            return []

        return self._extract_with_pymupdf(pdf_path, page_num)

    def _extract_with_pymupdf(
        self,
        pdf_path: str,
        page_num: int,
    ) -> List[TableMetadata]:
        """
        Extract tables using PyMuPDF 1.26.x find_tables() API.

        Modern approach (2025):
        - page.find_tables() detects tables
        - table.to_pandas() exports DataFrame
        - table.to_markdown() exports markdown
        - Cell bboxes available via table.cells
        """
        try:
            import fitz  # PyMuPDF

            # Open PDF
            doc = fitz.open(pdf_path)
            page = doc[page_num]

            # Find tables on page using modern API
            tables_finder = page.find_tables()

            result = []
            for table in tables_finder.tables:
                # Convert to DataFrame (modern API)
                df = table.to_pandas()

                # Skip empty tables
                if df.empty:
                    continue

                # Drop completely empty rows
                df = df.dropna(how="all")
                if df.empty:
                    continue

                # Get markdown (modern API)
                try:
                    markdown = table.to_markdown()
                except:
                    # Fallback to pandas if to_markdown() not available
                    markdown = df.to_markdown(index=False)

                # Create TableMetadata
                meta = TableMetadata(
                    num_rows=len(df),
                    num_cols=len(df.columns),
                    has_header=True,  # PyMuPDF detects headers
                    extraction_method="pymupdf",
                    confidence=None,
                    data=df.values.tolist(),
                    df_json=df.to_json(orient='records'),
                    markdown=markdown,
                )
                result.append(meta)

            doc.close()
            return result

        except Exception as e:
            print(f"PyMuPDF table extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return []


class TableEnricher:
    """
    Enrich table metadata with additional analysis.

    Optional enhancements:
    - Detect numeric columns
    - Identify wide vs long format
    - Suggest transformations
    """

    @staticmethod
    def analyze_table(meta: TableMetadata) -> Dict[str, Any]:
        """
        Analyze table structure and content.

        Returns:
            Dictionary with analysis results
        """
        if not meta.df_json:
            return {}

        # Load DataFrame
        df = pd.read_json(meta.df_json)

        # Analyze columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Detect format
        is_wide = len(df.columns) > len(df)
        suggested_format = "long" if is_wide else "wide"

        return {
            "numeric_columns": numeric_cols,
            "text_columns": text_cols,
            "is_wide_format": is_wide,
            "suggested_format": suggested_format,
            "total_cells": len(df) * len(df.columns),
            "empty_cells": df.isnull().sum().sum(),
        }

    @staticmethod
    def to_tidy_format(meta: TableMetadata) -> Optional[str]:
        """
        Convert wide table to tidy (long) format.

        Returns:
            JSON string of tidy DataFrame, or None if conversion not applicable
        """
        if not meta.df_json:
            return None

        try:
            df = pd.read_json(meta.df_json)

            # Simple tidy conversion: melt numeric columns
            id_cols = df.select_dtypes(include=['object']).columns.tolist()
            value_cols = df.select_dtypes(include=['number']).columns.tolist()

            if not id_cols or not value_cols:
                return None

            tidy_df = df.melt(
                id_vars=id_cols,
                value_vars=value_cols,
                var_name='variable',
                value_name='value'
            )

            return tidy_df.to_json(orient='records')

        except Exception as e:
            print(f"Tidy conversion failed: {e}")
            return None
