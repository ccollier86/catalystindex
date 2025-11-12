# Vision-Aware Chunking & Reranking Plan

## Objective
Let the classification/ingestion pipeline decide when a document (or specific chunks) should:
1. Be processed with a vision-language model (e.g., for diagrams or tables).
2. Use an alternate reranker (e.g., ColBERT) instead of the default Cohere reranker for those chunks.

## Workflow
1. **Vision heuristic or LLM hint**
   - During policy advising, analyze page content for cues (images, HTML `<img>`, PDF vector graphics).
   - If a section benefits from vision processing, mark the chunk with `metadata["vision_required"] = true`.

2. **Vision chunk generation**
   - For flagged sections, run a vision-language model (e.g., OpenAI GPT-4o or CLIP-based pipeline) to extract extra text chunks or descriptions.
   - Store these “vision chunks” in the same vector store track (`track="vision"`).

3. **Metadata annotations**
   - Tag each chunk with `vision_supported=true/false`, `preferred_reranker=colbert|cohere`, etc.
   - These tags influence search options and reranker selection.

4. **Search integration**
   - When search runs, inspect chunk metadata:
     - If `preferred_reranker=colbert`, route those candidates through the ColBERT pipeline.
     - Combine reranked results with existing tracks using final fusion.

5. **Configuration**
   - Settings for vision model (`CATALYST_VISION__provider`, `__model`, `__api_key`).
   - Enable/disable ColBERT reranker per chunk type.

## Implementation Steps
1. Extend Policy Advisor to flag vision-worthy sections.
2. Implement VisionChunkGenerator that captures images/diagrams and produces descriptive chunks.
3. Add metadata fields and update vector store payloads for reranker hints.
4. Integrate ColBERT (or other reranker) path in `SearchService` when chunk metadata requests it.
5. Document how to enable vision processing and reranker hints.
