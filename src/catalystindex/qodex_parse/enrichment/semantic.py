"""
Semantic layer builder - adds meaning as metadata, not reorganization

Key insight from semantic chunking article:
- Semantic chunking is for RETRIEVAL (finding relevant content)
- NOT for NAVIGATION (preserving document structure)

Solution: Add semantic grouping as METADATA pointer references
- Preserve spatial structure
- Add semantic similarity scores
- Link semantically related chunks
- Detect topic boundaries
"""

from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ..core.schemas import QodexChunk, SemanticMetadata


class SemanticLayerBuilder:
    """
    Build semantic layer by adding meaning as metadata.

    Does NOT reorganize chunks - preserves spatial structure.
    Adds semantic grouping info as pointer references.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.5,
        topic_boundary_threshold: float = 0.3,
        max_neighbors: int = 5,
    ):
        """
        Args:
            openai_api_key: OpenAI API key for embeddings
            model: Embedding model to use
            similarity_threshold: Minimum similarity for semantic grouping
            topic_boundary_threshold: Max similarity for topic boundaries
            max_neighbors: Maximum semantic neighbors to track
        """
        self.openai_api_key = openai_api_key
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.topic_boundary_threshold = topic_boundary_threshold
        self.max_neighbors = max_neighbors

    def build_semantic_layer(
        self,
        chunks: List[QodexChunk],
    ) -> List[QodexChunk]:
        """
        Add semantic metadata to chunks.

        Process:
        1. Generate embeddings for all chunks
        2. Calculate similarity to adjacent chunks
        3. Detect topic boundaries
        4. Group semantically similar chunks
        5. Find semantic neighbors across document

        Args:
            chunks: List of chunks with spatial graph built

        Returns:
            Same chunks with semantic metadata added
        """
        if not chunks:
            return chunks

        # Generate embeddings
        self._generate_embeddings(chunks)

        # Calculate similarity to adjacent chunks
        self._calculate_adjacent_similarity(chunks)

        # Detect topic boundaries
        self._detect_topic_boundaries(chunks)

        # Build semantic groups
        self._build_semantic_groups(chunks)

        # Find semantic neighbors
        self._find_semantic_neighbors(chunks)

        return chunks

    def _generate_embeddings(self, chunks: List[QodexChunk]) -> None:
        """
        Generate embeddings for all chunks using OpenAI.

        Updates chunk.embedding in place.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key)

            # Collect texts
            texts = [chunk.text for chunk in chunks]

            # Batch embed (max 2048 texts at once for OpenAI)
            batch_size = 2048
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_chunks = chunks[i:i+batch_size]

                response = client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                )

                # Assign embeddings
                for j, embedding_obj in enumerate(response.data):
                    embedding = np.array(embedding_obj.embedding)
                    batch_chunks[j].embedding = embedding

        except Exception as e:
            print(f"Embedding generation failed: {e}")
            # If embedding fails, set dummy embeddings
            for chunk in chunks:
                chunk.embedding = np.zeros(1536)  # Default embedding size

    def _calculate_adjacent_similarity(self, chunks: List[QodexChunk]) -> None:
        """
        Calculate similarity to previous and next chunks.

        Updates semantic.similarity_to_prev and similarity_to_next.
        """
        for i, chunk in enumerate(chunks):
            if chunk.semantic is None:
                chunk.semantic = SemanticMetadata()

            # Similarity to previous
            if i > 0 and chunks[i-1].embedding is not None and chunk.embedding is not None:
                sim = cosine_similarity(
                    chunks[i-1].embedding.reshape(1, -1),
                    chunk.embedding.reshape(1, -1)
                )[0, 0]
                chunk.semantic.similarity_to_prev = float(sim)

            # Similarity to next
            if i < len(chunks) - 1 and chunks[i+1].embedding is not None and chunk.embedding is not None:
                sim = cosine_similarity(
                    chunk.embedding.reshape(1, -1),
                    chunks[i+1].embedding.reshape(1, -1)
                )[0, 0]
                chunk.semantic.similarity_to_next = float(sim)

    def _detect_topic_boundaries(self, chunks: List[QodexChunk]) -> None:
        """
        Detect topic boundaries based on similarity drops.

        A topic boundary occurs when similarity to next chunk is low.
        """
        for chunk in chunks:
            if chunk.semantic is None:
                chunk.semantic = SemanticMetadata()

            # Topic boundary if similarity to next is low
            if chunk.semantic.similarity_to_next is not None:
                if chunk.semantic.similarity_to_next < self.topic_boundary_threshold:
                    chunk.semantic.is_topic_boundary = True
                    chunk.semantic.topic_shift_score = 1.0 - chunk.semantic.similarity_to_next

    def _build_semantic_groups(self, chunks: List[QodexChunk]) -> None:
        """
        Group chunks by semantic similarity.

        Uses topic boundaries to create groups.
        Groups are assigned sequential IDs.
        """
        group_counter = 0
        current_group = f"group_{group_counter}"

        for chunk in chunks:
            if chunk.semantic is None:
                chunk.semantic = SemanticMetadata()

            # Assign current group
            chunk.semantic.group_id = current_group

            # Start new group at topic boundary
            if chunk.semantic.is_topic_boundary:
                group_counter += 1
                current_group = f"group_{group_counter}"

    def _find_semantic_neighbors(self, chunks: List[QodexChunk]) -> None:
        """
        Find semantically similar chunks across the document.

        For each chunk, find the most similar chunks (excluding adjacent ones).
        """
        # Build embedding matrix
        embeddings = np.array([c.embedding for c in chunks if c.embedding is not None])

        if len(embeddings) == 0:
            return

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # For each chunk, find top similar chunks
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                continue

            if chunk.semantic is None:
                chunk.semantic = SemanticMetadata()

            # Get similarity scores
            sims = similarity_matrix[i]

            # Create list of (chunk_idx, similarity) tuples
            candidates = [
                (j, sims[j])
                for j in range(len(chunks))
                if j != i  # Exclude self
                and abs(j - i) > 1  # Exclude adjacent chunks
                and sims[j] >= self.similarity_threshold
            ]

            # Sort by similarity (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Take top N neighbors
            top_neighbors = candidates[:self.max_neighbors]

            # Create neighbor metadata
            chunk.semantic.semantic_neighbors = [
                {
                    "chunk_id": chunks[idx].chunk_id,
                    "similarity": float(sim),
                    "excerpt": chunks[idx].text[:100] + "..." if len(chunks[idx].text) > 100 else chunks[idx].text,
                }
                for idx, sim in top_neighbors
            ]


class SemanticQueryHelper:
    """
    Helper class for semantic queries.

    Provides utilities for finding semantically relevant content.
    """

    def __init__(self, chunks: List[QodexChunk], openai_api_key: str):
        """
        Args:
            chunks: List of chunks with semantic layer
            openai_api_key: OpenAI API key for query embedding
        """
        self.chunks = chunks
        self.openai_api_key = openai_api_key

        # Build embedding matrix
        self.embeddings = np.array([
            c.embedding for c in chunks
            if c.embedding is not None
        ])
        self.chunk_index = {
            c.chunk_id: i for i, c in enumerate(chunks)
            if c.embedding is not None
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_by_type: Optional[List[str]] = None,
    ) -> List[tuple[QodexChunk, float]]:
        """
        Semantic search for relevant chunks.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_by_type: Optional list of chunk types to filter by

        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self._embed_query(query)

        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]

        # Create results
        results = []
        for chunk, sim in zip(self.chunks, similarities):
            # Filter by type if specified
            if filter_by_type and chunk.type.value not in filter_by_type:
                continue

            results.append((chunk, float(sim)))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key)

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
            )

            return np.array(response.data[0].embedding)

        except Exception as e:
            print(f"Query embedding failed: {e}")
            return np.zeros(1536)

    def get_semantic_group(self, group_id: str) -> List[QodexChunk]:
        """Get all chunks in a semantic group"""
        return [
            c for c in self.chunks
            if c.semantic and c.semantic.group_id == group_id
        ]

    def get_topic_boundaries(self) -> List[QodexChunk]:
        """Get all chunks that are topic boundaries"""
        return [
            c for c in self.chunks
            if c.semantic and c.semantic.is_topic_boundary
        ]
