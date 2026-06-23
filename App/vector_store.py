"""
MediAssist AI — Vector Store
Creates and loads a FAISS vector store for document embeddings.
"""

import os
import shutil
from langchain_community.vectorstores import FAISS

# Import config
try:
    from backend.config import VECTOR_DB_DIR
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.config import VECTOR_DB_DIR


def create_vector_store(chunks, embeddings):
    """
    Create a new FAISS vector store from document chunks.

    If a vector store already exists at the persist directory,
    it will be cleared and recreated.

    Args:
        chunks: List of LangChain Document objects (chunked).
        embeddings: HuggingFace embedding model instance.

    Returns:
        FAISS vector store instance.
    """
    # Clear existing vector store if it exists
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print(f"  [INFO] Clearing existing vector store at {VECTOR_DB_DIR}")
        shutil.rmtree(VECTOR_DB_DIR)
    
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    print(f"  [INFO] Creating FAISS vector store with {len(chunks)} chunks...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    vector_store.save_local(VECTOR_DB_DIR)

    print(f"  [OK] FAISS Vector store created and persisted at {VECTOR_DB_DIR}")
    return vector_store


def load_vector_store(embeddings):
    """
    Load an existing FAISS vector store from disk.

    Args:
        embeddings: HuggingFace embedding model instance.

    Returns:
        FAISS vector store instance, or None if not found.
    """
    if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        print(f"  [ERROR] No vector store found at {VECTOR_DB_DIR}")
        print("  [INFO] Run 'python ingest.py' first to create the vector store.")
        return None

    print(f"  [INFO] Loading FAISS vector store from {VECTOR_DB_DIR}")

    try:
        vector_store = FAISS.load_local(
            folder_path=VECTOR_DB_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Verify the collection has documents (FAISS specific way to count)
        count = vector_store.index.ntotal
        print(f"  [OK] FAISS Vector store loaded with {count} documents")

        return vector_store
    except Exception as e:
        print(f"  [ERROR] Failed to load FAISS index: {e}")
        return None
