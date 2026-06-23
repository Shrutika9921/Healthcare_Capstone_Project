"""
MediAssist AI — Embedding Service
Provides a singleton HuggingFace embedding model with device detection.
Uses BAAI/bge-small-en-v1.5 for fast, high-quality embeddings.
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings

# Import config
try:
    from backend.config import EMBEDDING_MODEL_NAME
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.config import EMBEDDING_MODEL_NAME

# Singleton instance
_embedding_model = None


def get_embedding_model():
    """
    Returns a singleton instance of the HuggingFace embedding model.

    Automatically detects CUDA availability and uses GPU if possible.
    The model is loaded only once and reused across calls.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    global _embedding_model

    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [INFO] Loading embedding model: {EMBEDDING_MODEL_NAME} (device: {device})")

        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("  [OK] Embedding model loaded successfully")

    return _embedding_model