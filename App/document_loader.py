"""
MediAssist AI — Document Loader
Loads PDF, DOCX, and TXT files from the uploads directory.
Adds source metadata to each document for traceability.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_documents(folder_path):
    """
    Load all supported documents from the given folder.

    Supported formats: .pdf, .docx, .txt
    Each document gets metadata with source filename and file type.

    Args:
        folder_path: Path to the folder containing documents.

    Returns:
        List of LangChain Document objects.
    """
    docs = []
    supported_extensions = {".pdf", ".docx", ".txt"}

    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return docs

    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in supported_extensions]

    if not files:
        print(f"[WARNING] No supported documents found in {folder_path}")
        return docs

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
            elif ext == ".docx":
                loader = Docx2txtLoader(filepath)
            elif ext == ".txt":
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                continue

            loaded_docs = loader.load()

            # Enrich metadata for each loaded document
            for doc in loaded_docs:
                doc.metadata["source_filename"] = filename
                doc.metadata["file_type"] = ext.replace(".", "")

            docs.extend(loaded_docs)
            print(f"  [OK] Loaded: {filename} ({len(loaded_docs)} pages/sections)")

        except Exception as e:
            print(f"  [ERROR] Failed to load {filename}: {e}")
            continue

    return docs