#!/bin/bash

# Encode PDF to base64
PDF_BASE64=$(base64 -i fullstack-web-dev.pdf | tr -d '\n')

# Create JSON payload
cat > /tmp/fullstack_payload.json <<EOF
{
  "document_id": "fullstack-web-dev-test",
  "document_title": "Fullstack Web Development Guide",
  "content": "$PDF_BASE64",
  "content_type": "application/pdf",
  "knowledge_base_id": "test-kb",
  "schema": "technical"
}
EOF

# Generate valid JWT token with proper scopes
TOKEN="eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJvcmdfaWQiOiAidGVzdC1vcmciLCAid29ya3NwYWNlX2lkIjogInRlc3Qtd29ya3NwYWNlIiwgInVzZXJfaWQiOiAidGVzdC11c2VyIiwgInNjb3BlcyI6IFsiKiJdfQ.cL-h2ba86CKBwwNnubx_YrS51tMPtdBJfeV8SVeoG7M"

# Submit ingestion request
echo "========================================="
echo "🚀 STARTING INGESTION TEST"
echo "========================================="
echo "Document: Fullstack Web Development Guide"
echo ""

OVERALL_START=$(date +%s)

RESPONSE=$(curl -s -X POST http://localhost:18888/ingest/document \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d @/tmp/fullstack_payload.json)

SUBMIT_END=$(date +%s)
SUBMIT_DURATION=$((SUBMIT_END - OVERALL_START))

echo "✅ Request submitted in ${SUBMIT_DURATION} seconds"
echo ""

# Extract job_id
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null)

if [ -n "$JOB_ID" ]; then
  echo "📊 Job ID: $JOB_ID"
  echo "⏳ Monitoring job progress..."
  echo ""

  ITERATION=0
  for i in {1..120}; do
    sleep 2
    ITERATION=$((i * 2))

    JOB_STATUS=$(curl -s http://localhost:18888/ingest/jobs/$JOB_ID \
      -H "Authorization: Bearer $TOKEN")

    STATUS=$(echo "$JOB_STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null)
    PROGRESS=$(echo "$JOB_STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('progress_percentage', 0))" 2>/dev/null)
    SUMMARY=$(echo "$JOB_STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('progress_summary', 'N/A'))" 2>/dev/null)

    echo "  [$ITERATION sec] $STATUS: $PROGRESS% - $SUMMARY"

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
doc = data.get('document', {})
metadata = doc.get('metadata', {})
print(f\"Policy Name: {metadata.get('advisor_policy', 'N/A')}\")
print(f\"Confidence: {metadata.get('orchestrator_confidence', 'N/A')}\")
print(f\"Notes: {metadata.get('orchestrator_notes', 'N/A')}\")
print(f\"Tags: {metadata.get('orchestrator_tags', {})}\")
print(f\"Overrides: {metadata.get('orchestrator_overrides', {})}\")
"
  echo ""

  # Show sample chunks with full metadata
  echo "========================================="
  echo "📦 SAMPLE CHUNKS WITH LLM METADATA:"
  echo "========================================="
  echo "$FINAL_STATUS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
chunks = data.get('document', {}).get('chunks', [])
print(f\"Total chunks: {len(chunks)}\")
print(\"\")

# Show first 3 chunks with full metadata
for i, chunk in enumerate(chunks[:3], 1):
    print(f\"--- CHUNK {i} ---\")
    print(f\"ID: {chunk.get('chunk_id')}\")
    print(f\"Tier: {chunk.get('chunk_tier')}\")
    print(f\"Text (first 150 chars): {chunk.get('text', '')[:150]}...\")
    print(f\"Key Terms: {chunk.get('key_terms', [])}\")
    metadata = chunk.get('metadata', {})
    print(f\"Likely Question: {metadata.get('likely_question', 'N/A')}\")
    print(f\"Token Count: {metadata.get('token_count', 'N/A')}\")
    print(f\"Chunk Mode: {metadata.get('chunk_mode', 'N/A')}\")
    print(f\"Summary Model: {metadata.get('summary_model', 'N/A')}\")
    print(f\"Page ID: {metadata.get('page_id', 'N/A')}\")
    print(\"\")
"

  echo "========================================="
  echo "✨ TEST COMPLETE"
  echo "========================================="
fi

# Cleanup
rm /tmp/fullstack_payload.json
