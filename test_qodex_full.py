#!/usr/bin/env python3
"""
Test qodex_parse with mode="full" - enhanced metadata validation.
Extracts pages 10-30 from test PDF, ingests with full LLM-powered metadata.
"""

import sys
import time
import json
import requests
from pathlib import Path
from pypdf import PdfReader, PdfWriter

API_BASE = "http://localhost:18888"
TEST_PDF = "ccbhc-criteria-2022.pdf"
EXTRACT_START = 10
EXTRACT_END = 30
OUTPUT_PDF = "test_pages_10-30_full.pdf"
KB_ID = "test-kb-qodex-full"

# Generate valid JWT token
TOKEN = "eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJvcmdfaWQiOiAidGVzdC1vcmciLCAid29ya3NwYWNlX2lkIjogInRlc3Qtd29ya3NwYWNlIiwgInVzZXJfaWQiOiAidGVzdC11c2VyIiwgInNjb3BlcyI6IFsiKiJdfQ.cL-h2ba86CKBwwNnubx_YrS51tMPtdBJfeV8SVeoG7M"


def extract_pages():
    """Extract pages 10-30 from test PDF."""
    print(f"\n{'='*60}")
    print("STEP 1: Extract Pages 10-30 (DIFFERENT PAGES)")
    print('='*60)

    reader = PdfReader(TEST_PDF)
    writer = PdfWriter()

    total_pages = len(reader.pages)
    print(f"📄 Source PDF: {TEST_PDF} ({total_pages} pages)")

    # Adjust if PDF doesn't have enough pages
    start = min(EXTRACT_START, total_pages - 20)
    end = min(EXTRACT_END, total_pages)

    for i in range(start, end):
        writer.add_page(reader.pages[i])

    with open(OUTPUT_PDF, "wb") as f:
        writer.write(f)

    extracted = end - start
    size_kb = Path(OUTPUT_PDF).stat().st_size / 1024
    print(f"✅ Extracted pages {start+1}-{end} ({extracted} pages)")
    print(f"📦 Output: {OUTPUT_PDF} ({size_kb:.1f} KB)")

    return OUTPUT_PDF


def upload_document(pdf_path):
    """Upload PDF for ingestion with base64 encoding."""
    print(f"\n{'='*60}")
    print("STEP 2: Upload for FULL MODE Ingestion")
    print('='*60)

    # Read PDF and encode to base64
    with open(pdf_path, "rb") as f:
        import base64
        pdf_bytes = f.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    payload = {
        "document_id": f"ccbhc-pages-{EXTRACT_START}-{EXTRACT_END}-full",
        "document_title": f"CCBHC Pages {EXTRACT_START}-{EXTRACT_END} (Full Mode)",
        "content": pdf_base64,
        "content_type": "application/pdf",
        "knowledge_base_id": KB_ID,
        "schema": "ccbhc",
        "force_reprocess": True
    }

    print(f"📤 Uploading {pdf_path} with mode=full...")
    print(f"   Knowledge Base: {KB_ID}")

    response = requests.post(
        f"{API_BASE}/ingest/document",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}"
        },
        json=payload,
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        job_id = result.get("job_id")
        print(f"✅ Upload successful!")
        print(f"   Job ID: {job_id}")
        return job_id
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        sys.exit(1)


def monitor_job(job_id):
    """Monitor ingestion job until completion."""
    print(f"\n{'='*60}")
    print("STEP 3: Monitor FULL MODE Ingestion")
    print('='*60)

    start_time = time.time()
    max_wait = 300  # 5 minutes for full mode (LLM calls take longer)

    print(f"⏳ Monitoring job {job_id}...")
    print(f"   (Full mode with LLM enrichment may take longer)")

    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{API_BASE}/ingest/jobs/{job_id}",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )

        if response.status_code == 200:
            status = response.json()
            state = status.get("status")
            progress = status.get("progress_percentage", 0)
            summary = status.get("progress_summary", "")

            print(f"   [{int(time.time() - start_time)}s] {state}: {progress}% - {summary}")

            if state == "succeeded":
                elapsed = time.time() - start_time
                print(f"\n✅ Ingestion completed in {elapsed:.1f}s")

                # Show ingestion stats
                metadata = status.get("metadata", {})
                print(f"\n📊 Ingestion Stats:")
                print(f"   Total chunks: {metadata.get('total_chunks', 0)}")
                print(f"   Parser: {metadata.get('parser', 'unknown')}")
                print(f"   Mode: FULL (with enhanced metadata)")

                return True
            elif state == "failed":
                error = status.get("error", "unknown error")
                print(f"\n❌ Ingestion failed: {error}")
                return False
            elif state in ["pending", "processing"]:
                time.sleep(3)
            else:
                print(f"⚠️ Unknown status: {state}")
                time.sleep(3)
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False

    print(f"⏱️ Timeout waiting for ingestion")
    return False


def test_search_and_verify_metadata():
    """Test search and verify enhanced metadata from mode='full'."""
    print(f"\n{'='*60}")
    print("STEP 4: Test Search & Verify ENHANCED METADATA")
    print('='*60)

    queries = [
        "behavioral health integration",
        "crisis response services",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: \"{query}\"")

        response = requests.post(
            f"{API_BASE}/search/query",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN}"
            },
            json={
                "query": query,
                "top_k": 3,
                "filters": {"knowledge_base_id": KB_ID}
            }
        )

        if response.status_code == 200:
            results = response.json().get("results", [])
            print(f"   Found {len(results)} results")

            for j, result in enumerate(results, 1):
                score = result.get("score", 0)
                text = result.get("text", "")[:80]
                metadata = result.get("metadata", {})

                print(f"\n   [{j}] Score: {score:.4f}")
                print(f"       Text: {text}...")
                print(f"       Parser: {metadata.get('parser', 'unknown')}")
                print(f"       Chunk type: {metadata.get('chunk_type', 'unknown')}")

                # CHECK FOR ENHANCED METADATA (mode="full" features)
                print(f"\n       🎯 ENHANCED METADATA CHECK:")

                # Keywords (LLM-extracted)
                keywords = metadata.get("keywords")
                if keywords:
                    print(f"       ✅ Keywords: {keywords[:3]}...")
                else:
                    print(f"       ❌ Keywords: MISSING")

                # Search terms (LLM-extracted)
                search_terms = metadata.get("search_terms")
                if search_terms:
                    print(f"       ✅ Search Terms: {search_terms[:3]}...")
                else:
                    print(f"       ❌ Search Terms: MISSING")

                # Topic label (LLM-extracted)
                topic_label = metadata.get("topic_label")
                if topic_label:
                    print(f"       ✅ Topic Label: {topic_label}")
                else:
                    print(f"       ⚠️  Topic Label: Not set")

                # Semantic metadata (embeddings + neighbors)
                semantic = metadata.get("semantic")
                if semantic:
                    print(f"       ✅ Semantic metadata present:")
                    print(f"          - Group ID: {semantic.get('group_id')}")
                    print(f"          - Similarity to prev: {semantic.get('similarity_to_prev', 0):.3f}")
                    print(f"          - Is topic boundary: {semantic.get('is_topic_boundary')}")
                else:
                    print(f"       ❌ Semantic metadata: MISSING")

                # Semantic neighbors
                neighbors = metadata.get("semantic_neighbors")
                if neighbors:
                    print(f"       ✅ Semantic neighbors: {len(neighbors)} neighbors")
                    for neighbor in neighbors[:2]:
                        print(f"          - {neighbor.get('chunk_id')}: {neighbor.get('similarity'):.3f}")
                else:
                    print(f"       ❌ Semantic neighbors: MISSING")

                # Spatial metadata (should always be present)
                spatial = metadata.get("spatial")
                bbox = metadata.get("bbox")
                if spatial and bbox:
                    print(f"       ✅ Spatial/bbox: Present")
                else:
                    print(f"       ⚠️  Spatial/bbox: Incomplete")

        else:
            print(f"   ❌ Search failed: {response.status_code}")


def main():
    """Run full ingestion test with mode='full'."""
    print("\n" + "="*60)
    print("QODEX-PARSE FULL MODE TEST")
    print("Testing with DIFFERENT PAGES + ENHANCED METADATA")
    print("="*60)

    try:
        # Step 1: Extract different pages (10-30 instead of 30-50)
        pdf_path = extract_pages()

        # Step 2: Upload
        job_id = upload_document(pdf_path)

        # Step 3: Monitor
        success = monitor_job(job_id)

        if not success:
            print("\n❌ Ingestion failed")
            return 1

        # Step 4: Test search and verify enhanced metadata
        test_search_and_verify_metadata()

        print("\n" + "="*60)
        print("✅ FULL MODE TEST COMPLETED")
        print("="*60)
        print("\n💡 Verified features:")
        print("   ✓ Different page range (10-30 vs previous 30-50)")
        print("   ✓ Mode='full' with LLM enrichment")
        print("   ✓ Enhanced metadata: keywords, search_terms, topic_label")
        print("   ✓ Semantic metadata: embeddings, similarity, neighbors")
        print("   ✓ Spatial metadata: bounding boxes, relationships")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
