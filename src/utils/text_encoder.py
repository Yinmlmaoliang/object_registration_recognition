"""
Text Encoder for Object Registration and Recognition System.

This module provides a wrapper for SentenceTransformer to encode object
attributes and user queries for text-based retrieval.
"""

import numpy as np
from typing import List, Dict, Optional, Union


class TextEncoder:
    """
    Text encoder using SentenceTransformer for semantic text embeddings.

    Uses the 'multi-qa-mpnet-base-dot-v1' model which provides:
    - 768-dimensional embeddings
    - Better semantic search performance
    - Trained on QA and search data

    Example:
        >>> encoder = TextEncoder()
        >>> # Encode object attributes
        >>> attributes = ["my favorite mug", "blue mug with Mickey Mouse"]
        >>> embedding = encoder.encode_attributes(attributes)
        >>> print(embedding.shape)  # (768,)
        >>>
        >>> # Encode query
        >>> query_emb = encoder.encode_query("Find my Mickey Mouse mug")
        >>> print(query_emb.shape)  # (768,)
    """

    MODEL_NAME = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'

    def __init__(
        self,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """
        Initialize the TextEncoder.

        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            verbose: Whether to print loading progress
        """
        self.device = device
        self.verbose = verbose

        if self.verbose:
            print(f"Loading SentenceTransformer: {self.MODEL_NAME}")

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL_NAME, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        if self.verbose:
            print(f"  Model loaded successfully")
            print(f"  Embedding dimension: {self.embedding_dim}")

    def encode_attributes(
        self,
        attributes: List[str],
        aggregation: str = 'max'
    ) -> np.ndarray:
        """
        Encode multiple attribute descriptions into a single embedding.

        Uses max pooling by default to aggregate multiple attributes,
        which captures the most salient features across all descriptions.

        Args:
            attributes: List of attribute text descriptions
            aggregation: Aggregation strategy ('max', 'mean')

        Returns:
            Aggregated embedding as numpy array of shape (embedding_dim,)
        """
        if not attributes:
            raise ValueError("Attributes list cannot be empty")

        # Encode all attributes
        embeddings = self.model.encode(
            attributes,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Aggregate embeddings
        if aggregation == 'max':
            aggregated = embeddings.max(dim=0).values
        elif aggregation == 'mean':
            aggregated = embeddings.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation}")

        return aggregated.cpu().numpy()

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query text into an embedding.

        Args:
            query: Query text string

        Returns:
            Query embedding as numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.

        Args:
            texts: List of text strings

        Returns:
            Embeddings as numpy array of shape (num_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        template_embeddings: Dict[str, np.ndarray],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Compute cosine similarity between query and template embeddings.

        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            template_embeddings: Dict mapping instance_id to embedding
            top_k: Number of top results to return

        Returns:
            List of dicts with 'instance_id' and 'score', sorted by score descending
        """
        from sentence_transformers import util

        similarities = {}
        query_tensor = self._to_tensor(query_embedding)

        for inst_id, template_emb in template_embeddings.items():
            template_tensor = self._to_tensor(template_emb)
            sim = util.cos_sim(query_tensor, template_tensor).item()
            similarities[inst_id] = sim

        # Sort by similarity (descending)
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = [
            {'instance_id': inst_id, 'score': score}
            for inst_id, score in ranked[:top_k]
        ]

        return results

    def _to_tensor(self, array: np.ndarray):
        """Convert numpy array to tensor."""
        import torch
        if isinstance(array, torch.Tensor):
            return array
        return torch.from_numpy(array).float()

    @property
    def feat_dim(self) -> int:
        """Get the feature dimension (alias for embedding_dim)."""
        return self.embedding_dim