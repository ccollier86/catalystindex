"""
Simple Docling Test - Convert DOCX to Markdown

Tests Docling's conversion capabilities to see what structure/metadata we get.
"""

import sys
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("ERROR: docling not installed")
    print("Install with: pip install docling")
    sys.exit(1)


def convert_docx_to_markdown(docx_path: str, output_dir: str = "docling_output"):
    """
    Convert a DOCX file to Markdown using Docling.

    Args:
        docx_path: Path to DOCX file
        output_dir: Directory to save output (default: docling_output)
    """
    docx_file = Path(docx_path)

    if not docx_file.exists():
        print(f"ERROR: File not found: {docx_path}")
        return

    if not docx_file.suffix.lower() == '.docx':
        print(f"WARNING: File doesn't appear to be DOCX: {docx_path}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nConverting: {docx_file.name}")
    print("=" * 80)

    # Initialize Docling converter
    converter = DocumentConverter()

    # Convert document
    result = converter.convert(str(docx_file))

    # Save as Markdown
    md_filename = docx_file.stem + ".md"
    md_path = output_path / md_filename

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(result.document.export_to_markdown())

    print(f"✓ Markdown saved to: {md_path}")

    # Also save the structured JSON if available
    json_filename = docx_file.stem + "_structure.json"
    json_path = output_path / json_filename

    try:
        import json
        # Try to get document structure
        doc_dict = result.document.export_to_dict()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ Structure JSON saved to: {json_path}")
    except Exception as e:
        print(f"⚠ Could not save structure JSON: {e}")

    # Print summary
    print("\nDocument Info:")
    print(f"  - Format: {result.input.format if hasattr(result, 'input') else 'N/A'}")

    # Try to get page count
    try:
        if hasattr(result.document, 'pages'):
            print(f"  - Pages: {len(result.document.pages)}")
    except:
        pass

    # Show first 500 chars of markdown
    markdown_content = result.document.export_to_markdown()
    print(f"\nFirst 500 characters of Markdown:")
    print("-" * 80)
    print(markdown_content[:500])
    print("-" * 80)

    return result


def main():
    """Test Docling on multiple DOCX files"""

    # Check for command line args
    if len(sys.argv) < 2:
        print("Usage: python test_docling.py <docx_file1> [docx_file2] ...")
        print("\nExample:")
        print("  python test_docling.py sample.docx")
        print("  python test_docling.py *.docx")
        return

    # Process each file
    docx_files = sys.argv[1:]

    print(f"\nDocling DOCX → Markdown Test")
    print(f"Processing {len(docx_files)} file(s)")

    for docx_path in docx_files:
        try:
            convert_docx_to_markdown(docx_path)
        except Exception as e:
            print(f"\n✗ ERROR processing {docx_path}:")
            print(f"  {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print("Done! Check the 'docling_output' directory for results.")


if __name__ == "__main__":
    main()
