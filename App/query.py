"""
MediAssist AI — Interactive Query CLI
Load the vector store, build the RAG chain, and query interactively.

Usage:
    python App/query.py
"""

import sys
import os

# Ensure App directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding_service import get_embedding_model
from vector_store import load_vector_store
from retriever import get_retriever, retrieve_documents
from rag_chain import build_rag_chain, ask_question


def main():
    """Interactive query loop for the MediAssist AI RAG pipeline."""

    print("=" * 60)
    print("  MediAssist AI — Healthcare Document Assistant")
    print("=" * 60)

    # ─── Initialize Pipeline ────────────────────────────
    print("\n[INIT] Setting up RAG pipeline...\n")

    # Load embedding model
    print("[1/3] Loading embedding model...")
    embeddings = get_embedding_model()

    # Load vector store
    print("\n[2/3] Loading vector store...")
    vector_store = load_vector_store(embeddings)

    if vector_store is None:
        print("\n[ERROR] Cannot start query mode without a vector store.")
        print("Run 'python App/ingest.py' first to ingest documents.")
        sys.exit(1)

    # Build RAG chain
    print("\n[3/3] Building RAG chain with Groq LLM...")
    retriever = get_retriever(vector_store)
    rag_chain = build_rag_chain(retriever)

    print("\n" + "=" * 60)
    print("  [OK] MediAssist AI is ready!")
    print("  Type your healthcare questions below.")
    print("  Type 'exit' or 'quit' to stop.")
    print("  Type 'sources' to see retrieved documents for last query.")
    print("=" * 60)

    last_query = None

    while True:
        print()
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("\nGoodbye!")
            break

        # Show source documents for the last query
        if query.lower() == "sources" and last_query:
            print("\n--- Retrieved Source Documents ---")
            docs = retrieve_documents(retriever, last_query)
            if docs:
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source_filename", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    print(f"\n[Source {i}] {source} | Page: {page}")
                    print(f"Content: {doc.page_content[:300]}...")
            else:
                print("No relevant documents found for the last query.")
            continue

        # Ask the question
        try:
            last_query = query
            print("\nMediAssist AI: ", end="", flush=True)
            answer = ask_question(rag_chain, query)
            print(answer)
        except Exception as e:
            print(f"\n[ERROR] Failed to process query: {e}")


if __name__ == "__main__":
    main()
