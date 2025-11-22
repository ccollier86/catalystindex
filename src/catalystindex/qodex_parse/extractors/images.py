"""
Image extraction with base64 export for storage and retrieval

Extracts images from PDFs and exports as:
- base64 encoded data (for inline storage)
- Image files (for file-based storage)
- OCR text (optional)
"""

import base64
from pathlib import Path
from typing import List, Optional, Literal
from io import BytesIO
from PIL import Image
from ..core.schemas import ImageMetadata, Bbox


class ImageExtractor:
    """
    Extract images from PDFs with multiple export options.

    Features:
    - Extract images from PyMuPDF or pdfplumber
    - Export as base64 or file
    - Optional OCR for image text
    - Bounding box tracking
    """

    def __init__(
        self,
        export_format: Literal["base64", "file", "both"] = "base64",
        output_dir: Optional[Path] = None,
        extract_ocr: bool = False,
    ):
        """
        Args:
            export_format: How to export images
            output_dir: Directory for file export (required if export_format includes "file")
            extract_ocr: Whether to run OCR on images
        """
        self.export_format = export_format
        self.output_dir = Path(output_dir) if output_dir else None
        self.extract_ocr = extract_ocr

        if "file" in export_format and not self.output_dir:
            raise ValueError("output_dir required when export_format includes 'file'")

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_images_from_page(
        self,
        pdf_path: str,
        page_num: int,
        doc_id: str,
    ) -> List[ImageMetadata]:
        """
        Extract all images from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            doc_id: Document ID for naming exported files

        Returns:
            List of ImageMetadata objects
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            page = doc[page_num]

            images = []
            image_list = page.get_images()

            for img_idx, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    # Get image data
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Load with PIL for processing
                    pil_image = Image.open(BytesIO(image_bytes))

                    # Convert CMYK to RGB immediately (before any export)
                    if pil_image.mode == 'CMYK':
                        pil_image = pil_image.convert('RGB')

                    width, height = pil_image.size

                    # Get bounding box (if available)
                    bbox = self._get_image_bbox(page, xref)

                    # Export image
                    base64_data = None
                    file_path = None

                    if self.export_format in ["base64", "both"]:
                        base64_data = self._to_base64(pil_image)

                    if self.export_format in ["file", "both"]:
                        file_path = self._save_image_file(
                            pil_image,
                            doc_id,
                            page_num,
                            img_idx,
                        )

                    # Extract OCR text if requested
                    ocr_text = None
                    if self.extract_ocr:
                        ocr_text = self._extract_ocr_text(pil_image)

                    # Create metadata
                    meta = ImageMetadata(
                        width=width,
                        height=height,
                        format=image_ext.upper(),
                        base64_data=base64_data,
                        file_path=str(file_path) if file_path else None,
                        caption=None,
                        alt_text=None,
                        ocr_text=ocr_text,
                    )

                    images.append(meta)

                except Exception as e:
                    print(f"Failed to extract image {img_idx} from page {page_num}: {e}")
                    continue

            doc.close()
            return images

        except Exception as e:
            print(f"Image extraction failed for page {page_num}: {e}")
            return []

    def _to_base64(self, pil_image: Image.Image) -> str:
        """Convert PIL image to base64 string (always as PNG)"""
        # Ensure RGB mode (should already be converted, but double-check)
        if pil_image.mode == 'CMYK':
            pil_image = pil_image.convert('RGB')

        # Convert to PNG bytes
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    def _save_image_file(
        self,
        pil_image: Image.Image,
        doc_id: str,
        page_num: int,
        img_idx: int,
    ) -> Path:
        """Save image to file (always as PNG)"""
        # Should already be RGB, but double-check
        if pil_image.mode == 'CMYK':
            pil_image = pil_image.convert('RGB')

        filename = f"{doc_id}_p{page_num}_img{img_idx}.png"
        filepath = self.output_dir / filename
        pil_image.save(filepath, format='PNG')
        return filepath

    def _get_image_bbox(self, page, xref: int) -> Optional[Bbox]:
        """
        Get bounding box for an image.

        Note: This is a simplified version - PyMuPDF doesn't always provide
        accurate image bboxes. May need more sophisticated detection.
        """
        try:
            # Get image rectangles
            img_rects = page.get_image_rects(xref)
            if img_rects:
                rect = img_rects[0]
                return Bbox(
                    x0=rect.x0,
                    y0=rect.y0,
                    x1=rect.x1,
                    y1=rect.y1,
                    page=page.number,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                )
        except:
            pass

        return None

    def _extract_ocr_text(self, pil_image: Image.Image) -> Optional[str]:
        """
        Extract text from image using OCR.

        Requires tesseract to be installed.
        """
        try:
            import pytesseract
            text = pytesseract.image_to_string(pil_image)
            return text.strip() if text else None
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return None


class ImageEnricher:
    """
    Enrich image metadata with AI-generated descriptions.

    Optional: Use vision models to generate captions and alt text.
    """

    @staticmethod
    def generate_caption(
        base64_data: str,
        openai_api_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate caption using GPT-4 Vision.

        Args:
            base64_data: Base64-encoded image data
            openai_api_key: OpenAI API key

        Returns:
            Generated caption, or None if failed
        """
        if not openai_api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in 1-2 sentences. Focus on what's relevant for document understanding."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_data
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Caption generation failed: {e}")
            return None

    @staticmethod
    def generate_alt_text(
        base64_data: str,
        openai_api_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate accessibility alt text using GPT-4 Vision.

        Args:
            base64_data: Base64-encoded image data
            openai_api_key: OpenAI API key

        Returns:
            Generated alt text, or None if failed
        """
        if not openai_api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Generate concise alt text for this image (1 sentence, for accessibility)."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_data
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Alt text generation failed: {e}")
            return None
