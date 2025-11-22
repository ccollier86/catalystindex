#!/usr/bin/env python3
"""Test search quality with various queries."""

import json
import urllib.request
import urllib.error

TOKEN = "eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJvcmdfaWQiOiAidGVzdC1vcmciLCAid29ya3NwYWNlX2lkIjogInRlc3Qtd29ya3NwYWNlIiwgInVzZXJfaWQiOiAidGVzdC11c2VyIiwgInNjb3BlcyI6IFsiKiJdfQ.cL-h2ba86CKBwwNnubx_YrS51tMPtdBJfeV8SVeoG7M"
BASE_URL = "http://localhost:18888"

test_queries = [
    "What are the CCBHC staffing requirements?",
    "substance abuse treatment services",
    "crisis intervention 24/7 availability",
    "care coordination requirements",
    "quality reporting measures",
]

def run_search(query):
    """Run a search query and return results."""
    data = json.dumps({
        "query": query,
        "knowledge_base_id": "test-kb",
        "limit": 5,
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{BASE_URL}/search/query",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.read().decode('utf-8')}")
        return {"results": []}

def rate_result(result, query):
    """Rate a search result on various metrics."""
    ratings = {}

    # Relevance (1-10): Does text contain query terms?
    text = result.get("text", "").lower()
    query_terms = query.lower().split()
    relevance = sum(1 for term in query_terms if term in text) / len(query_terms) * 10
    ratings["relevance"] = min(10, int(relevance))

    # Completeness (1-10): Is the text chunk substantial?
    text_length = len(text)
    if text_length > 500:
        completeness = 10
    elif text_length > 300:
        completeness = 8
    elif text_length > 150:
        completeness = 6
    else:
        completeness = 4
    ratings["completeness"] = completeness

    # Metadata Quality (1-10): Does it have OpenParse metadata?
    metadata = result.get("metadata", {})
    has_parser = "parser" in metadata and metadata["parser"] == "openparse"
    has_bbox = "bbox" in metadata
    has_node_type = "node_type" in metadata
    has_page = "start_page" in result or "page_number" in metadata

    metadata_score = (
        (3 if has_parser else 0) +
        (3 if has_bbox else 0) +
        (2 if has_node_type else 0) +
        (2 if has_page else 0)
    )
    ratings["metadata_quality"] = metadata_score

    # Score (1-10): Normalize the similarity score
    score = result.get("score", 0)
    if score > 0.8:
        score_rating = 10
    elif score > 0.6:
        score_rating = 8
    elif score > 0.4:
        score_rating = 6
    elif score > 0.2:
        score_rating = 4
    else:
        score_rating = 2
    ratings["score_quality"] = score_rating

    return ratings

print("=" * 80)
print("SEARCH QUALITY EVALUATION")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 80}")
    print(f"Query {i}: {query}")
    print("=" * 80)

    results = run_search(query)

    retrieved = results.get("tracks", [{}])[0].get("retrieved", 0) if results.get("tracks") else 0
    result_count = len(results.get("results", []))

    print(f"Retrieved: {retrieved} chunks")
    print(f"Returned: {result_count} results")

    if result_count == 0:
        print("\n⚠️  NO RESULTS - Rating: 0/10 across all metrics")
        print("   Issue: Chunks may not be embedded in vector store yet")
        continue

    print(f"\nTop {min(3, result_count)} Results:")
    print("-" * 80)

    all_ratings = []
    for j, result in enumerate(results["results"][:3], 1):
        ratings = rate_result(result, query)
        all_ratings.append(ratings)

        print(f"\nResult #{j}:")
        print(f"  Score: {result.get('score', 0):.4f}")
        print(f"  Chunk ID: {result.get('chunk_id', 'N/A')}")
        print(f"  Parser: {result.get('metadata', {}).get('parser', 'N/A')}")
        print(f"  Node Type: {result.get('metadata', {}).get('node_type', 'N/A')}")
        print(f"  Has BBox: {'bbox' in result.get('metadata', {})}")
        print(f"  Text Preview: {result.get('text', '')[:150]}...")
        print(f"\n  RATINGS:")
        print(f"    Relevance: {ratings['relevance']}/10")
        print(f"    Completeness: {ratings['completeness']}/10")
        print(f"    Metadata Quality: {ratings['metadata_quality']}/10")
        print(f"    Score Quality: {ratings['score_quality']}/10")

    # Calculate average ratings
    if all_ratings:
        avg_relevance = sum(r["relevance"] for r in all_ratings) / len(all_ratings)
        avg_completeness = sum(r["completeness"] for r in all_ratings) / len(all_ratings)
        avg_metadata = sum(r["metadata_quality"] for r in all_ratings) / len(all_ratings)
        avg_score = sum(r["score_quality"] for r in all_ratings) / len(all_ratings)
        overall = (avg_relevance + avg_completeness + avg_metadata + avg_score) / 4

        print(f"\n{'=' * 80}")
        print(f"AVERAGE RATINGS FOR QUERY {i}:")
        print(f"  Relevance: {avg_relevance:.1f}/10")
        print(f"  Completeness: {avg_completeness:.1f}/10")
        print(f"  Metadata Quality: {avg_metadata:.1f}/10")
        print(f"  Score Quality: {avg_score:.1f}/10")
        print(f"  OVERALL: {overall:.1f}/10")

print(f"\n{'=' * 80}")
print("EVALUATION COMPLETE")
print("=" * 80)
