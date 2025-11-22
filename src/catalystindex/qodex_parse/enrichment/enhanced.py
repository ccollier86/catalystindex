"""
Enhanced metadata layer - LLM-generated keywords and questions

Uses GPT to generate:
- 3-5 keywords that describe the chunk content
- 1-3 questions a user might ask to find this chunk
"""

from typing import List
import json
from ..core.schemas import QodexChunk, EnhancedMetadata


class EnhancedMetadataBuilder:
    """
    Build enhanced metadata using LLM (GPT) to generate keywords and questions.

    This layer adds human-readable metadata that improves search and discovery.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
    ):
        """
        Args:
            openai_api_key: OpenAI API key for GPT
            model: GPT model to use (gpt-4o-mini for cost efficiency)
        """
        self.openai_api_key = openai_api_key
        self.model = model

    def build_enhanced_layer(
        self,
        chunks: List[QodexChunk],
    ) -> List[QodexChunk]:
        """
        Add enhanced metadata to chunks using GPT.

        Process:
        1. For each chunk, use GPT to generate:
           - 3-5 keywords describing the content
           - 1-3 questions a user might ask to find this chunk
        2. Populate EnhancedMetadata for each chunk

        Args:
            chunks: List of chunks to enhance

        Returns:
            Same chunks with enhanced metadata added
        """
        if not chunks:
            return chunks

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key)

            # Process chunks in batches for efficiency
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                self._process_batch(client, batch)

            return chunks

        except Exception as e:
            print(f"Enhanced metadata generation failed: {e}")
            # If enhancement fails, return chunks without enhanced metadata
            return chunks

    def _process_batch(self, client, chunks: List[QodexChunk]) -> None:
        """Process a batch of chunks to generate enhanced metadata"""

        for chunk in chunks:
            # Skip if text is too short
            if len(chunk.text.split()) < 5:
                continue

            try:
                # Generate keywords and questions using GPT
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert at analyzing document chunks and generating metadata.

For the given text chunk, generate:
1. 3-5 relevant keywords that describe the content
2. 1-3 short questions a user might ask to find this chunk

Return ONLY a JSON object with this exact format:
{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "questions": ["question 1?", "question 2?"]
}

Keep keywords concise (1-3 words each).
Keep questions natural and user-focused."""
                        },
                        {
                            "role": "user",
                            "content": f"Text chunk:\n{chunk.text[:500]}"  # Limit to 500 chars
                        }
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )

                # Parse response
                content = response.choices[0].message.content.strip()

                # Extract JSON (handle markdown code blocks)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                metadata_json = json.loads(content)

                # Create EnhancedMetadata
                chunk.enhanced = EnhancedMetadata(
                    keywords=metadata_json.get("keywords", []),
                    search_terms=metadata_json.get("questions", []),
                )

            except Exception as e:
                print(f"Failed to enhance chunk {chunk.chunk_id}: {e}")
                # Skip this chunk and continue
                continue
