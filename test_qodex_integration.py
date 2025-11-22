#!/usr/bin/env python3
"""
Test script to verify Qodex-Parse integration with catalyst_index.

This script tests:
1. QodexParserAdapter imports correctly
2. Parser is registered in the registry
3. Basic parsing works with a sample PDF
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)

    try:
        from catalystindex.parsers.qodex_adapter import (
            QodexParserAdapter,
            QODEX_PARSER_METADATA,
        )
        print("✓ QodexParserAdapter imports successfully")
        print(f"✓ Parser metadata: {QODEX_PARSER_METADATA.name}")
        print(f"  - Content types: {', '.join(QODEX_PARSER_METADATA.content_types)}")
        print(f"  - Description: {QODEX_PARSER_METADATA.description}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registry():
    """Test that Qodex is registered as the PDF parser."""
    print("\n" + "=" * 60)
    print("TEST 2: Parser Registry")
    print("=" * 60)

    try:
        from catalystindex.parsers.registry import default_registry

        registry = default_registry()

        # Check that 'pdf' is registered
        pdf_parser = registry.resolve("pdf")
        print(f"✓ PDF parser registered: {type(pdf_parser).__name__}")

        # Verify it's QodexParserAdapter
        from catalystindex.parsers.qodex_adapter import QodexParserAdapter
        if isinstance(pdf_parser, QodexParserAdapter):
            print("✓ PDF parser is QodexParserAdapter (correct!)")
        else:
            print(f"✗ PDF parser is {type(pdf_parser).__name__} (expected QodexParserAdapter)")
            return False

        # Check metadata
        metadata = registry.metadata("pdf")
        if metadata:
            print(f"✓ Metadata: {metadata.name} - {metadata.description[:50]}...")

        # List all parsers
        print("\nRegistered parsers:")
        for name, meta in registry.list_parsers().items():
            print(f"  - {name}: {meta.description[:60]}...")

        return True

    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parsing():
    """Test basic parsing functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Basic Parsing")
    print("=" * 60)

    try:
        from catalystindex.parsers.qodex_adapter import QodexParserAdapter

        # Create adapter
        adapter = QodexParserAdapter(mode="basic")
        print("✓ QodexParserAdapter instantiated")

        # Check if test PDF exists
        test_pdf = Path("ccbhc-criteria-2022.pdf")
        if not test_pdf.exists():
            print(f"⚠ Test PDF not found: {test_pdf}")
            print("  Skipping parsing test (this is OK for unit tests)")
            return True

        # Parse test PDF
        print(f"Parsing test PDF: {test_pdf}")
        with open(test_pdf, "rb") as f:
            pdf_bytes = f.read()

        sections = list(adapter.parse(
            pdf_bytes,
            document_title="CCBHC Criteria 2022",
            content_type="application/pdf"
        ))

        print(f"✓ Parsed {len(sections)} sections")

        if sections:
            # Show first section
            first = sections[0]
            print(f"\nFirst section preview:")
            print(f"  - Slug: {first.section_slug}")
            print(f"  - Title: {first.title[:50]}...")
            print(f"  - Text length: {len(first.text)} chars")
            print(f"  - Pages: {first.start_page}-{first.end_page}")
            print(f"  - Parser: {first.metadata.get('parser')}")
            print(f"  - Chunk type: {first.metadata.get('chunk_type')}")

            # Check for spatial metadata
            if "bbox" in first.metadata:
                print(f"  - Bbox: ✓ (spatial metadata present)")

            # Check for table metadata
            table_sections = [s for s in sections if s.metadata.get("chunk_type") == "table"]
            print(f"\n✓ Found {len(table_sections)} table chunks")

        return True

    except Exception as e:
        print(f"✗ Parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("QODEX-PARSE INTEGRATION TEST")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Registry", test_registry()))
    results.append(("Parsing", test_parsing()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
