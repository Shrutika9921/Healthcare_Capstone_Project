"""
MediAssist AI — Document Ingestion Pipeline
Orchestrates: Load Documents → Chunk → Embed → Store in ChromaDB

Usage:
    python App/ingest.py
"""

import sys
import os
import time

# Ensure App directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import UPLOAD_DIR, VECTOR_DB_DIR
from document_loader import load_documents
from chunking import create_chunks
from embedding_service import get_embedding_model
from vector_store import create_vector_store


def run_ingestion():
    """Run the complete document ingestion pipeline."""

    print("=" * 60)
    print("  MediAssist AI — Document Ingestion Pipeline")
    print("=" * 60)
    start_time = time.time()

    # ─── Step 1: Load Documents ──────────────────────────
    print("\n[STEP 1/4] Loading documents...")
    docs = load_documents(UPLOAD_DIR)

    if not docs:
        print("\n[FAILED] No documents were loaded. Please add files to Data/uploads/")
        print("Supported formats: .pdf, .docx, .txt")
        return False

    print(f"\n  Total documents loaded: {len(docs)}")

    # ─── Step 2: Create Chunks ───────────────────────────
    print("\n[STEP 2/4] Chunking documents...")
    chunks = create_chunks(docs)
    print(f"  Total chunks created: {len(chunks)}")

    # Show a sample chunk for verification
    if chunks:
        sample = chunks[0]
        print(f"\n  Sample chunk preview:")
        print(f"  Source: {sample.metadata.get('source_filename', 'N/A')}")
        print(f"  Page: {sample.metadata.get('page', 'N/A')}")
        print(f"  Content: {sample.page_content[:150]}...")

    # ─── Step 3: Load Embedding Model ────────────────────
    print("\n[STEP 3/4] Loading embedding model...")
    embeddings = get_embedding_model()

    # ─── Step 4: Create Vector Store ─────────────────────
    print("\n[STEP 4/4] Creating vector store in FAISS...")
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vector_store = create_vector_store(chunks, embeddings)

    # ─── Done ────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  [OK] Ingestion complete in {elapsed:.1f} seconds")
    print(f"  [OK] {len(docs)} documents -> {len(chunks)} chunks -> FAISS")
    print(f"  [OK] Vector store saved at: {VECTOR_DB_DIR}")
    print("=" * 60)
    print("\nYou can now run queries with: python App/query.py")

    return True


if __name__ == "__main__":
    run_ingestion()