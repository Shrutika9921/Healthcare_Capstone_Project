"""
MediAssist AI — Retriever
Wraps the ChromaDB vector store as a LangChain retriever
with similarity search and relevance score filtering.
"""

import os

# Import config
try:
    from backend.config import RETRIEVER_TOP_K, RETRIEVER_SCORE_THRESHOLD
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.config import RETRIEVER_TOP_K, RETRIEVER_SCORE_THRESHOLD


def get_retriever(vector_store):
    """
    Create a retriever from the vector store.

    Uses similarity search with score threshold to filter out
    irrelevant results.

    Args:
        vector_store: Chroma vector store instance.

    Returns:
        LangChain retriever instance.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": RETRIEVER_TOP_K,
            "score_threshold": RETRIEVER_SCORE_THRESHOLD
        }
    )

    return retriever


def retrieve_documents(retriever, query):
    """
    Retrieve relevant documents for a given query.

    Args:
        retriever: LangChain retriever instance.
        query: User's question string.

    Returns:
        List of relevant Document objects with metadata.
    """
    docs = retriever.invoke(query)
    return docs
