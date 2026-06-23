"""
MediAssist AI — Document Chunking
Splits documents into chunks with metadata for vector storage.
Uses larger chunk sizes suited for healthcare documents.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import config — handle both direct execution and module import
try:
    from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(documents):
    """
    Split documents into chunks with metadata enrichment.

    Each chunk gets a unique chunk_id composed of:
    source_filename + page_number + chunk_index

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects with enriched metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Add chunk_id metadata for traceability
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source_filename", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{source}_page{page}_chunk{idx}"
        chunk.metadata["chunk_index"] = idx

    return chunks