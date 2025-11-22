#!/usr/bin/env python3
"""
Test qodex_parse integration with full ingestion pipeline.
Extracts 20 pages from test PDF, ingests them, and tests search.
"""

import sys
import time
import json
import requests
from pathlib import Path
from pypdf import PdfReader, PdfWriter

API_BASE = "http://localhost:18888"
TEST_PDF = "ccbhc-criteria-2022.pdf"
EXTRACT_START = 30
EXTRACT_END = 50
OUTPUT_PDF = "test_pages_30-50.pdf"

def extract_pages():
    """Extract pages 30-50 from test PDF."""
    print(f"\n{'='*60}")
    print("STEP 1: Extract Pages 30-50")
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
    """Upload PDF for ingestion."""
    print(f"\n{'='*60}")
    print("STEP 2: Upload for Ingestion")
    print('='*60)

    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path, f, "application/pdf")}
        data = {
            "organization_id": "test-org",
            "workspace_id": "test-workspace",
            "document_title": f"CCBHC Test Pages {EXTRACT_START}-{EXTRACT_END}",
            "tags": json.dumps(["test", "qodex", "integration"]),
            "parser_override": "pdf"  # Use qodex (default PDF parser)
        }

        print(f"📤 Uploading {pdf_path}...")
        response = requests.post(
            f"{API_BASE}/ingest/upload",
            files=files,
            data=data,
            timeout=30
        )

    if response.status_code == 200:
        result = response.json()
        job_id = result.get("job_id")
        doc_id = result.get("document_id")
        print(f"✅ Upload successful!")
        print(f"   Job ID: {job_id}")
        print(f"   Document ID: {doc_id}")
        return job_id, doc_id
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        sys.exit(1)

def monitor_job(job_id):
    """Monitor ingestion job until completion."""
    print(f"\n{'='*60}")
    print("STEP 3: Monitor Ingestion")
    print('='*60)

    start_time = time.time()
    max_wait = 120  # 2 minutes

    print(f"⏳ Monitoring job {job_id}...")

    while time.time() - start_time < max_wait:
        response = requests.get(f"{API_BASE}/ingest/status/{job_id}")

        if response.status_code == 200:
            status = response.json()
            state = status.get("status")
            progress = status.get("progress", 0)

            print(f"   Status: {state} ({progress}%)")

            if state == "completed":
                elapsed = time.time() - start_time
                print(f"✅ Ingestion completed in {elapsed:.1f}s")

                # Show ingestion stats
                if "result" in status:
                    result = status["result"]
                    print(f"\n📊 Ingestion Stats:")
                    print(f"   Chunks: {result.get('total_chunks', 0)}")
                    print(f"   Parser: {result.get('parser', 'unknown')}")
                    if "parse_metadata" in result:
                        meta = result["parse_metadata"]
                        print(f"   Tables found: {meta.get('table_count', 0)}")
                        print(f"   Images found: {meta.get('image_count', 0)}")

                return True
            elif state == "failed":
                print(f"❌ Ingestion failed: {status.get('error')}")
                return False
            elif state in ["pending", "processing"]:
                time.sleep(2)
            else:
                print(f"⚠️ Unknown status: {state}")
                time.sleep(2)
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False

    print(f"⏱️ Timeout waiting for ingestion")
    return False

def test_search(doc_id):
    """Test search functionality."""
    print(f"\n{'='*60}")
    print("STEP 4: Test Search")
    print('='*60)

    # Test queries related to CCBHC content
    queries = [
        "quality measures",
        "crisis services",
        "treatment planning",
        "behavioral health"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: \"{query}\"")

        response = requests.post(
            f"{API_BASE}/search/query",
            json={
                "organization_id": "test-org",
                "workspace_id": "test-workspace",
                "query": query,
                "top_k": 3
            }
        )

        if response.status_code == 200:
            results = response.json().get("results", [])
            print(f"   Found {len(results)} results:")

            for j, result in enumerate(results[:3], 1):
                score = result.get("score", 0)
                text = result.get("text", "")[:100]
                metadata = result.get("metadata", {})
                parser = metadata.get("parser", "unknown")
                chunk_type = metadata.get("chunk_type", "unknown")

                print(f"   [{j}] Score: {score:.4f} | Parser: {parser} | Type: {chunk_type}")
                print(f"       Preview: {text}...")

                # Check for qodex metadata
                if "spatial" in metadata:
                    print(f"       ✅ Has spatial metadata")
                if "bbox" in metadata:
                    print(f"       ✅ Has bounding box")
        else:
            print(f"   ❌ Search failed: {response.status_code}")

def main():
    """Run full ingestion test."""
    print("\n" + "="*60)
    print("QODEX-PARSE FULL INGESTION TEST")
    print("="*60)

    try:
        # Step 1: Extract pages
        pdf_path = extract_pages()

        # Step 2: Upload
        job_id, doc_id = upload_document(pdf_path)

        # Step 3: Monitor
        success = monitor_job(job_id)

        if not success:
            print("\n❌ Ingestion failed")
            return 1

        # Step 4: Test search
        test_search(doc_id)

        print("\n" + "="*60)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\n💡 Check the search results above to see:")
        print("   - Qodex parser being used")
        print("   - Spatial metadata presence")
        print("   - Bounding box information")
        print("   - Chunk type classification (text, table, heading)")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
