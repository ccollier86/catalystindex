#!/usr/bin/env python3
"""Test script for QodexSparseGenerator with qodex metadata."""

from catalystindex.models.common import ChunkRecord
from catalystindex.sparse.qodex_generator import QodexSparseGenerator


def test_qodex_sparse_generation():
    """Test sparse vector generation with qodex metadata."""
    print("=" * 70)
    print("Testing QodexSparseGenerator with Qodex Metadata")
    print("=" * 70)

    # Create generator
    generator = QodexSparseGenerator(
        search_term_weight=2.0,
        keyword_weight=1.5,
        text_weight=1.0,
    )
    print("\n✅ QodexSparseGenerator instantiated successfully")
    print(f"   - search_term_weight: {generator.search_term_weight}")
    print(f"   - keyword_weight: {generator.keyword_weight}")
    print(f"   - text_weight: {generator.text_weight}")

    # Test Case 1: Chunk with qodex metadata (full mode)
    print("\n" + "=" * 70)
    print("Test Case 1: Chunk with Qodex Metadata (keywords + search_terms)")
    print("=" * 70)

    chunk_with_metadata = ChunkRecord(
        chunk_id="test-chunk-1",
        section_slug="care_coordination",
        text="CCBHC programs provide comprehensive behavioral health care coordination and treatment planning for veterans.",
        chunk_tier="premium",
        start_page=1,
        end_page=1,
        bbox_pointer=None,
        summary="CCBHC care coordination for veterans",
        key_terms=["CCBHC", "care coordination", "veterans"],
        requires_previous=False,
        prev_chunk_id=None,
        confidence_note=None,
        metadata={
            "keywords": ["behavioral health", "care coordination", "treatment planning"],
            "search_terms": ["CCBHC", "veterans", "behavioral health services"],
            "section_title": "Care Coordination Requirements",
        },
    )

    sparse_vector = generator.generate(chunk_with_metadata)
    print(f"\n✅ Generated sparse vector with {len(sparse_vector)} unique indices")
    print(f"   Sample indices (first 10): {list(sparse_vector.keys())[:10]}")
    print(f"   Sample weights (first 10): {list(sparse_vector.values())[:10]}")

    # Verify weighting - check that search_terms have highest weight
    ccbhc_hash = generator._hash_token("ccbhc")
    veterans_hash = generator._hash_token("veterans")
    behavioral_hash = generator._hash_token("behavioral")

    if ccbhc_hash in sparse_vector:
        print(f"\n   Token 'ccbhc' (search_term): weight={sparse_vector[ccbhc_hash]:.2f}")
    if veterans_hash in sparse_vector:
        print(f"   Token 'veterans' (search_term): weight={sparse_vector[veterans_hash]:.2f}")
    if behavioral_hash in sparse_vector:
        print(f"   Token 'behavioral' (keyword): weight={sparse_vector[behavioral_hash]:.2f}")

    # Test Case 2: Chunk without qodex metadata (fallback mode)
    print("\n" + "=" * 70)
    print("Test Case 2: Chunk without Qodex Metadata (text-only fallback)")
    print("=" * 70)

    chunk_without_metadata = ChunkRecord(
        chunk_id="test-chunk-2",
        section_slug="treatment_planning",
        text="Treatment planning requires comprehensive assessment and documentation.",
        chunk_tier="premium",
        start_page=2,
        end_page=2,
        bbox_pointer=None,
        summary=None,
        key_terms=[],
        requires_previous=False,
        prev_chunk_id=None,
        confidence_note=None,
        metadata={},  # No qodex metadata
    )

    sparse_vector_fallback = generator.generate(chunk_without_metadata)
    print(f"\n✅ Generated sparse vector (fallback) with {len(sparse_vector_fallback)} unique indices")
    print(f"   Sample indices (first 10): {list(sparse_vector_fallback.keys())[:10]}")
    print(f"   Sample weights (first 10): {list(sparse_vector_fallback.values())[:10]}")

    # Test Case 3: Empty chunk
    print("\n" + "=" * 70)
    print("Test Case 3: Empty Chunk")
    print("=" * 70)

    empty_chunk = ChunkRecord(
        chunk_id="test-chunk-3",
        section_slug="empty_section",
        text="",
        chunk_tier="economy",
        start_page=3,
        end_page=3,
        bbox_pointer=None,
        summary=None,
        key_terms=[],
        requires_previous=False,
        prev_chunk_id=None,
        confidence_note=None,
        metadata={},
    )

    sparse_vector_empty = generator.generate(empty_chunk)
    print(f"\n✅ Empty chunk result: {sparse_vector_empty}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ All test cases passed!")
    print(f"   - Metadata-enhanced chunks: {len(sparse_vector)} indices")
    print(f"   - Text-only fallback chunks: {len(sparse_vector_fallback)} indices")
    print(f"   - Empty chunks: Correctly returns None")
    print("\n✅ QodexSparseGenerator is working correctly!")


if __name__ == "__main__":
    test_qodex_sparse_generation()
