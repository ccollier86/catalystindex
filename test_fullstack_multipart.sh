#!/bin/bash
#
# Test fullstack ingestion using PROPER multipart file upload
#
# This tests the production workflow:
# - File uploaded via multipart/form-data (NO BASE64)
# - Server uploads original to S3
# - Server processes file (parse → chunk → enrich → embed)
# - Artifacts stored to S3
#

PDF_FILE="fullstack-web-dev.pdf"
TOKEN="eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJvcmdfaWQiOiAidGVzdC1vcmciLCAid29ya3NwYWNlX2lkIjogInRlc3Qtd29ya3NwYWNlIiwgInVzZXJfaWQiOiAidGVzdC11c2VyIiwgInNjb3BlcyI6IFsiKiJdfQ.cL-h2ba86CKBwwNnubx_YrS51tMPtdBJfeV8SVeoG7M"

echo "========================================="
echo "🚀 FULLSTACK INGESTION TEST (MULTIPART)"
echo "========================================="
echo "Document: $PDF_FILE"
echo "Size: $(du -h "$PDF_FILE" | cut -f1)"
echo "Method: multipart/form-data (NO base64)"
echo ""

OVERALL_START=$(date +%s)

# Upload file via multipart/form-data
echo "📤 Uploading file via multipart/form-data..."
UPLOAD_START=$(date +%s)

RESPONSE=$(curl -s -X POST http://localhost:18888/ingest/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@$PDF_FILE" \
  -F "document_id=fullstack-web-dev-test" \
  -F "document_title=Fullstack Web Development Guide" \
  -F "knowledge_base_id=test-kb" \
  -F "schema=technical")

UPLOAD_END=$(date +%s)
UPLOAD_DURATION=$((UPLOAD_END - UPLOAD_START))

echo "✅ File uploaded in ${UPLOAD_DURATION} seconds"
echo ""

# Extract job_id
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null)

if [ -z "$JOB_ID" ]; then
  echo "❌ No job_id returned"
  echo "$RESPONSE" | python3 -m json.tool
  exit 1
fi

echo "📊 Job ID: $JOB_ID"
echo "⏳ Monitoring job progress..."
echo ""

# Monitor job progress
ITERATION=0
for i in {1..120}; do
  sleep 2
  ITERATION=$((i * 2))

  JOB_STATUS=$(curl -s http://localhost:18888/ingest/jobs/$JOB_ID \
    -H "Authorization: Bearer $TOKEN")

  STATUS=$(echo "$JOB_STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null)

  # Extract progress info
  DOCUMENTS=$(echo "$JOB_STATUS" | python3 -c "import sys, json; docs = json.load(sys.stdin).get('documents', []); print(len(docs))" 2>/dev/null)

  if [ "$DOCUMENTS" -gt 0 ]; then
    # Get current stage from first document
    CURRENT_STAGE=$(echo "$JOB_STATUS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
docs = data.get('documents', [])
if docs:
    progress = docs[0].get('progress', {})
    for stage in ['uploaded', 'embedded', 'enriched', 'chunked', 'parsed', 'acquired']:
        if stage in progress:
            stage_info = progress[stage]
            if isinstance(stage_info, dict) and stage_info.get('status') in ('running', 'succeeded'):
                print(stage)
                break
" 2>/dev/null)

    if [ -n "$CURRENT_STAGE" ]; then
      echo "  [$ITERATION sec] $STATUS - $CURRENT_STAGE"
    else
      echo "  [$ITERATION sec] $STATUS"
    fi
  else
    echo "  [$ITERATION sec] $STATUS"
  fi

  if echo "$STATUS" | grep -q "succeeded\|failed"; then
    FINAL_STATUS_JSON="$JOB_STATUS"
    break
  fi
done

OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))

echo ""
echo "========================================="
echo "⏱️  TOTAL INGESTION TIME: ${TOTAL_DURATION} SECONDS"
echo "========================================="
echo ""

# Show full results
echo "========================================="
echo "📋 FULL JOB RESULT:"
echo "========================================="
echo "$FINAL_STATUS_JSON" | python3 -m json.tool
echo ""

# Extract and show policy
echo "========================================="
echo "🧠 LLM-GENERATED POLICY:"
echo "========================================="
echo "$FINAL_STATUS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
docs = data.get('documents', [])
if docs:
    doc = docs[0]
    metadata = doc.get('metadata', {})
    print(f\"Policy Name: {metadata.get('advisor_policy', 'N/A')}\")
    print(f\"Confidence: {metadata.get('orchestrator_confidence', 'N/A')}\")
    print(f\"Notes: {metadata.get('orchestrator_notes', 'N/A')}\")
    print(f\"Tags: {metadata.get('orchestrator_tags', {})}\")
    print(f\"Overrides: {metadata.get('orchestrator_overrides', {})}\")
"
echo ""

# Show S3 storage info
echo "========================================="
echo "📦 S3 STORAGE INFO:"
echo "========================================="
echo "$FINAL_STATUS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
docs = data.get('documents', [])
if docs:
    doc = docs[0]
    metadata = doc.get('metadata', {})
    print(f\"S3 Bucket: {metadata.get('s3_bucket', 'N/A')}\")
    print(f\"S3 Key: {metadata.get('s3_key', 'N/A')}\")
    print(f\"Uploaded Via: {metadata.get('uploaded_via', 'N/A')}\")
    artifact = doc.get('artifact', {})
    print(f\"Artifact URI: {artifact.get('uri', 'N/A')}\")
"
echo ""

# Show sample chunks
echo "========================================="
echo "📦 SAMPLE CHUNKS WITH LLM METADATA:"
echo "========================================="
echo "$FINAL_STATUS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
docs = data.get('documents', [])
if docs:
    chunks = docs[0].get('chunks', [])
    print(f\"Total chunks: {len(chunks)}\")
    print(\"\")

    # Show first 3 chunks
    for i, chunk in enumerate(chunks[:3], 1):
        print(f\"--- CHUNK {i} ---\")
        print(f\"ID: {chunk.get('chunk_id')}\")
        print(f\"Tier: {chunk.get('chunk_tier')}\")
        text = chunk.get('text', '')
        print(f\"Text (first 150 chars): {text[:150]}...\")

        metadata = chunk.get('metadata', {})
        print(f\"Key Terms: {metadata.get('key_terms', [])}\")
        print(f\"Likely Question: {metadata.get('likely_question', 'N/A')}\")
        print(f\"Token Count: {metadata.get('token_count', 'N/A')}\")
        print(f\"Chunk Mode: {metadata.get('chunk_mode', 'N/A')}\")
        print(f\"Page ID: {metadata.get('page_id', 'N/A')}\")
        print(\"\")
"

echo "========================================="
echo "✨ TEST COMPLETE"
echo "========================================="
