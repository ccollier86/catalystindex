#!/usr/bin/env python3
"""
Test fullstack ingestion performance using PROPER multipart file upload.

This tests the production workflow:
- File uploaded via multipart/form-data
- Server uploads original to S3
- Server processes file (parse → chunk → enrich → embed → vector store)
- Artifacts stored to S3
- Returns job status with timing and metadata

NO BASE64 ENCODING - uses multipart upload as intended!
"""

import json
import time
from pathlib import Path

import requests

# Configuration
API_BASE = "http://localhost:18888"
TOKEN = "eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJvcmdfaWQiOiAidGVzdC1vcmciLCAid29ya3NwYWNlX2lkIjogInRlc3Qtd29ya3NwYWNlIiwgInVzZXJfaWQiOiAidGVzdC11c2VyIiwgInNjb3BlcyI6IFsiKiJdfQ.cL-h2ba86CKBwwNnubx_YrS51tMPtdBJfeV8SVeoG7M"
PDF_PATH = Path("/Volumes/code-bank/active/catalyst_index/fullstack-web-dev.pdf")


def main():
    print("=" * 60)
    print("🚀 FULLSTACK INGESTION TEST (MULTIPART UPLOAD)")
    print("=" * 60)
    print(f"Document: {PDF_PATH.name}")
    print(f"Size: {PDF_PATH.stat().st_size / 1024:.1f} KB")
    print(f"Method: multipart/form-data (NO base64)")
    print()

    overall_start = time.time()

    # Prepare multipart upload
    print("📤 Uploading file via multipart/form-data...")
    upload_start = time.time()

    headers = {"Authorization": f"Bearer {TOKEN}"}

    with open(PDF_PATH, "rb") as pdf_file:
        files = {"file": (PDF_PATH.name, pdf_file, "application/pdf")}
        data = {
            "document_id": "fullstack-web-dev-test",
            "document_title": "Fullstack Web Development Guide",
            "knowledge_base_id": "test-kb",
            "schema": "technical",
        }

        response = requests.post(
            f"{API_BASE}/ingest/upload", headers=headers, files=files, data=data
        )

    if not response.ok:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        return

    result = response.json()
    upload_duration = time.time() - upload_start

    print(f"✅ File uploaded in {upload_duration:.2f} seconds")
    print()

    # Extract job_id
    job_id = result.get("job_id")
    if not job_id:
        print("❌ No job_id returned")
        print(json.dumps(result, indent=2))
        return

    print(f"📊 Job ID: {job_id}")
    print("⏳ Monitoring job progress...")
    print()

    # Monitor job progress
    iteration = 0
    final_status = None

    for i in range(120):  # Max 4 minutes (120 * 2 seconds)
        time.sleep(2)
        iteration = (i + 1) * 2

        job_response = requests.get(
            f"{API_BASE}/ingest/jobs/{job_id}", headers={"Authorization": f"Bearer {TOKEN}"}
        )

        if not job_response.ok:
            print(f"⚠️  Failed to fetch job status: {job_response.status_code}")
            continue

        job_status = job_response.json()

        status = job_status.get("status", "unknown")

        # Get progress from first document
        documents = job_status.get("documents", [])
        if documents:
            doc = documents[0]
            progress_dict = doc.get("progress", {})

            # Find latest stage
            current_stage = None
            for stage in ["uploaded", "embedded", "enriched", "chunked", "parsed", "acquired"]:
                if stage in progress_dict:
                    stage_info = progress_dict[stage]
                    if isinstance(stage_info, dict) and stage_info.get("status") in ("running", "succeeded"):
                        current_stage = stage
                        break

            progress_str = f"{status}"
            if current_stage:
                progress_str = f"{status} - {current_stage}"

            print(f"  [{iteration:3d} sec] {progress_str}")
        else:
            print(f"  [{iteration:3d} sec] {status}")

        if status in ("succeeded", "failed"):
            final_status = job_status
            break

    overall_duration = time.time() - overall_start

    print()
    print("=" * 60)
    print(f"⏱️  TOTAL INGESTION TIME: {overall_duration:.1f} SECONDS")
    print("=" * 60)
    print()

    if not final_status:
        print("⚠️  Job did not complete within timeout")
        return

    # Show full results
    print("=" * 60)
    print("📋 FULL JOB RESULT:")
    print("=" * 60)
    print(json.dumps(final_status, indent=2))
    print()

    # Extract and show LLM-generated policy
    print("=" * 60)
    print("🧠 LLM-GENERATED POLICY:")
    print("=" * 60)

    documents = final_status.get("documents", [])
    if documents:
        doc = documents[0]
        metadata = doc.get("metadata", {})
        print(f"Policy Name: {metadata.get('advisor_policy', 'N/A')}")
        print(f"Confidence: {metadata.get('orchestrator_confidence', 'N/A')}")
        print(f"Notes: {metadata.get('orchestrator_notes', 'N/A')}")
        print(f"Tags: {metadata.get('orchestrator_tags', {})}")
        print(f"Overrides: {metadata.get('orchestrator_overrides', {})}")
        print()

        # Show S3 storage info
        print("=" * 60)
        print("📦 S3 STORAGE INFO:")
        print("=" * 60)
        print(f"S3 Bucket: {metadata.get('s3_bucket', 'N/A')}")
        print(f"S3 Key: {metadata.get('s3_key', 'N/A')}")
        print(f"Uploaded Via: {metadata.get('uploaded_via', 'N/A')}")
        print(f"Artifact URI: {doc.get('artifact', {}).get('uri', 'N/A')}")
        print()

        # Show sample chunks with full metadata
        print("=" * 60)
        print("📦 SAMPLE CHUNKS WITH LLM METADATA:")
        print("=" * 60)
        chunks = doc.get("chunks", [])
        print(f"Total chunks: {len(chunks)}")
        print()

        # Show first 3 chunks with full metadata
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"--- CHUNK {i} ---")
            print(f"ID: {chunk.get('chunk_id')}")
            print(f"Tier: {chunk.get('chunk_tier')}")
            text = chunk.get("text", "")
            print(f"Text (first 150 chars): {text[:150]}...")

            chunk_metadata = chunk.get("metadata", {})
            print(f"Key Terms: {chunk_metadata.get('key_terms', [])}")
            print(f"Likely Question: {chunk_metadata.get('likely_question', 'N/A')}")
            print(f"Token Count: {chunk_metadata.get('token_count', 'N/A')}")
            print(f"Chunk Mode: {chunk_metadata.get('chunk_mode', 'N/A')}")
            print(f"Page ID: {chunk_metadata.get('page_id', 'N/A')}")
            print()

    print("=" * 60)
    print("✨ TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
