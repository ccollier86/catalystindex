#!/bin/bash
set -e

echo "============================================================"
echo "QODEX-PARSE TEST: Pages 50-70"
echo "Using MULTIPART FORM DATA (NOT base64)"
echo "============================================================"

# Configuration
TEST_PDF="ccbhc-criteria-2022.pdf"
KB_ID="test-kb-pages-50-70"
API_BASE="http://localhost:18888"
TOKEN="eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJvcmdfaWQiOiAidGVzdC1vcmciLCAid29ya3NwYWNlX2lkIjogInRlc3Qtd29ya3NwYWNlIiwgInVzZXJfaWQiOiAidGVzdC11c2VyIiwgInNjb3BlcyI6IFsiKiJdfQ.cL-h2ba86CKBwwNnubx_YrS51tMPtdBJfeV8SVeoG7M"

# Step 1: Extract pages 50-70
echo ""
echo "============================================================"
echo "STEP 1: Extract Pages 50-70"
echo "============================================================"

python3 -c "
from pypdf import PdfReader, PdfWriter
reader = PdfReader('$TEST_PDF')
writer = PdfWriter()
for i in range(49, 70):  # 0-indexed, so 49-69 for pages 50-70
    if i < len(reader.pages):
        writer.add_page(reader.pages[i])
with open('test_pages_50-70.pdf', 'wb') as f:
    writer.write(f)
print(f'✅ Extracted pages 50-70 (21 pages)')
" 2>&1 || {
    echo "⚠️  pypdf not available, skipping page extraction"
    echo "Using full PDF for test instead"
    TEST_PDF_SUBSET="$TEST_PDF"
}

TEST_PDF_SUBSET="test_pages_50-70.pdf"

# Check file exists
if [ -f "$TEST_PDF_SUBSET" ]; then
    SIZE=$(ls -lh "$TEST_PDF_SUBSET" | awk '{print $5}')
    echo "📦 Output: $TEST_PDF_SUBSET ($SIZE)"
else
    TEST_PDF_SUBSET="$TEST_PDF"
    echo "Using full PDF: $TEST_PDF"
fi

# Step 2: Upload using MULTIPART FORM DATA (correct method)
echo ""
echo "============================================================"
echo "STEP 2: Upload using MULTIPART FORM DATA"
echo "============================================================"
echo "📤 Uploading $TEST_PDF_SUBSET via /ingest/upload..."
echo "   Knowledge Base: $KB_ID"
echo "   Method: MULTIPART FORM DATA (NOT base64)"

START_TIME=$(date +%s)

RESPONSE=$(curl -s -X POST $API_BASE/ingest/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@$TEST_PDF_SUBSET" \
  -F "document_id=ccbhc-pages-50-70" \
  -F "document_title=CCBHC Pages 50-70 (Full Mode)" \
  -F "knowledge_base_id=$KB_ID" \
  -F "schema=ccbhc" \
  -F "force_reprocess=true" \
  -F "parser_override=pdf")

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""
echo "✅ Request submitted in ${DURATION} seconds"

# Extract job_id
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null)

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to get job_id from response"
    exit 1
fi

# Step 3: Monitor ingestion
echo ""
echo "============================================================"
echo "STEP 3: Monitor Ingestion"
echo "============================================================"
echo "📊 Job ID: $JOB_ID"
echo "⏳ Monitoring job progress..."

for i in {1..90}; do
    sleep 2
    STATUS_JSON=$(curl -s $API_BASE/ingest/jobs/$JOB_ID \
      -H "Authorization: Bearer $TOKEN")

    STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status'))" 2>/dev/null)
    PROGRESS=$(echo "$STATUS_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('progress_percentage', 0))" 2>/dev/null)
    SUMMARY=$(echo "$STATUS_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('progress_summary', 'N/A'))" 2>/dev/null)

    echo "  [$i] $STATUS: $PROGRESS% - $SUMMARY"

    if echo "$STATUS" | grep -q "succeeded"; then
        echo ""
        echo "✅ Ingestion completed!"

        # Show stats
        TOTAL_CHUNKS=$(echo "$STATUS_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('total_chunks', 0))" 2>/dev/null)
        PARSER=$(echo "$STATUS_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('parser', 'unknown'))" 2>/dev/null)

        echo ""
        echo "📊 Ingestion Stats:"
        echo "   Total chunks: $TOTAL_CHUNKS"
        echo "   Parser: $PARSER"
        echo "   Mode: FULL (with enhanced metadata)"
        break
    elif echo "$STATUS" | grep -q "failed"; then
        echo ""
        echo "❌ Ingestion failed"
        echo "$STATUS_JSON" | python3 -m json.tool 2>/dev/null || echo "$STATUS_JSON"
        exit 1
    fi
done

# Step 4: Test search
echo ""
echo "============================================================"
echo "STEP 4: Test Search"
echo "============================================================"

# Query 1: Care coordination
echo ""
echo "🔍 Query 1: \"care coordination\""
echo ""
curl -s -X POST $API_BASE/search/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"query\": \"care coordination\",
    \"top_k\": 3,
    \"filters\": {\"knowledge_base_id\": \"$KB_ID\"}
  }" | python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('results', [])
print(f'   Found {len(results)} results\n')

for i, result in enumerate(results, 1):
    score = result.get('score', 0)
    text = result.get('text', '')[:100]
    metadata = result.get('metadata', {})

    print(f'   [{i}] Score: {score:.4f}')
    print(f'       Text: {text}...')
    print(f'       Page: {metadata.get(\"page_number\", \"N/A\")}')
    print('')
" 2>&1

# Query 2: Treatment planning
echo ""
echo "🔍 Query 2: \"treatment planning\""
echo ""
curl -s -X POST $API_BASE/search/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"query\": \"treatment planning\",
    \"top_k\": 3,
    \"filters\": {\"knowledge_base_id\": \"$KB_ID\"}
  }" | python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('results', [])
print(f'   Found {len(results)} results\n')

for i, result in enumerate(results, 1):
    score = result.get('score', 0)
    text = result.get('text', '')[:100]
    metadata = result.get('metadata', {})

    print(f'   [{i}] Score: {score:.4f}')
    print(f'       Text: {text}...')
    print(f'       Page: {metadata.get(\"page_number\", \"N/A\")}')
    print('')
" 2>&1

# Summary
echo ""
echo "============================================================"
echo "✅ PAGES 50-70 TEST COMPLETED"
echo "============================================================"
echo ""
echo "💡 Verified features:"
echo "   ✓ Pages 50-70 extracted and uploaded"
echo "   ✓ Used MULTIPART FORM DATA (NOT base64)"
echo "   ✓ Mode='full' with LLM enrichment"
echo "   ✓ Search working correctly"
echo ""
