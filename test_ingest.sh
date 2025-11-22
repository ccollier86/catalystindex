#!/bin/bash

# Encode PDF to base64
PDF_BASE64=$(base64 -i ccbhc-criteria-2022.pdf | tr -d '\n')

# Create JSON payload
cat > /tmp/ingest_payload.json <<EOF
{
  "document_id": "ccbhc-2022-test",
  "document_title": "CCBHC Criteria 2022",
  "content": "$PDF_BASE64",
  "content_type": "application/pdf",
  "knowledge_base_id": "test-kb",
  "schema": "ccbhc",
  "force_reprocess": true
}
EOF

# Generate valid JWT token with proper scopes
TOKEN="eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJvcmdfaWQiOiAidGVzdC1vcmciLCAid29ya3NwYWNlX2lkIjogInRlc3Qtd29ya3NwYWNlIiwgInVzZXJfaWQiOiAidGVzdC11c2VyIiwgInNjb3BlcyI6IFsiKiJdfQ.cL-h2ba86CKBwwNnubx_YrS51tMPtdBJfeV8SVeoG7M"

# Submit ingestion request
echo "Submitting ingestion request..."
echo "Document: CCBHC Criteria 2022"
START_TIME=$(date +%s)

RESPONSE=$(curl -s -X POST http://localhost:18888/ingest/document \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d @/tmp/ingest_payload.json)

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo "\n✅ Request submitted in ${DURATION} seconds"

# Extract job_id
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null)

if [ -n "$JOB_ID" ]; then
  echo "\n📊 Job ID: $JOB_ID"
  echo "Monitoring job progress..."

  for i in {1..60}; do
    sleep 2
    STATUS=$(curl -s http://localhost:18888/ingest/jobs/$JOB_ID \
      -H "Authorization: Bearer $TOKEN" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data.get('status')}: {data.get('progress_percentage', 0)}% - {data.get('progress_summary', 'N/A')}\")" 2>/dev/null)
    echo "  [$i] $STATUS"

    if echo "$STATUS" | grep -q "succeeded\|failed"; then
      break
    fi
  done
fi

# Cleanup
rm /tmp/ingest_payload.json
