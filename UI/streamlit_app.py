"""
MediAssist AI — Streamlit Chat Interface
A premium healthcare document assistant with chat UI,
document upload, and source citation display.

Usage:
    streamlit run UI/streamlit_app.py
"""

import streamlit as st
import sys
import os
import time

# ─── Path Setup ──────────────────────────────────────────
# Add App directory to path for imports
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "App")
sys.path.insert(0, APP_DIR)

from backend.config import UPLOAD_DIR, VECTOR_DB_DIR, GROQ_MODEL_NAME
from document_loader import load_documents
from chunking import create_chunks
from embedding_service import get_embedding_model
from vector_store import create_vector_store, load_vector_store
from retriever import get_retriever, retrieve_documents
from rag_chain import build_rag_chain, ask_question


# ─── Page Configuration ─────────────────────────────────
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Styles ── */
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        font-weight: 300;
    }

    /* ── Sidebar Styles ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: rgba(255, 255, 255, 0.75);
    }

    /* ── Status Cards ── */
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
    }
    .status-card .label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .status-card .value {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .status-online {
        color: #4ade80 !important;
    }
    .status-offline {
        color: #f87171 !important;
    }

    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(10px);
    }

    /* ── Source Expander ── */
    .source-card {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .source-card .source-title {
        color: #667eea;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .source-card .source-text {
        color: rgba(255, 255, 255, 0.65);
        font-size: 0.8rem;
        margin-top: 0.3rem;
        line-height: 1.5;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        background: rgba(102, 126, 234, 0.05);
    }

    /* ── Success/Info Messages ── */
    .stSuccess, .stInfo {
        border-radius: 10px;
    }

    /* ── Metric Styling ── */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    [data-testid="stMetricValue"] {
        color: #667eea !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(102, 126, 234, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0


# ─── Initialize RAG Pipeline ────────────────────────────
@st.cache_resource(show_spinner=False)
def init_rag_pipeline():
    """Initialize the RAG pipeline (cached so it loads only once)."""
    embeddings = get_embedding_model()
    vector_store = load_vector_store(embeddings)

    if vector_store is None:
        return None, None, 0

    retriever = get_retriever(vector_store)
    rag_chain = build_rag_chain(retriever)
    doc_count = vector_store._collection.count()

    return rag_chain, retriever, doc_count


def run_ingestion_pipeline():
    """Run the full ingestion pipeline and reinitialize RAG chain."""
    embeddings = get_embedding_model()

    docs = load_documents(UPLOAD_DIR)
    if not docs:
        return False, 0, 0

    chunks = create_chunks(docs)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    create_vector_store(chunks, embeddings)

    # Clear the cached pipeline so it reloads
    init_rag_pipeline.clear()

    return True, len(docs), len(chunks)


# ─── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 3rem;">🏥</span>
        <h2 style="margin: 0.5rem 0 0 0; color: #667eea;">MediAssist AI</h2>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Healthcare Document Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Status Section ──
    st.markdown("### System Status")

    # Try to initialize pipeline
    if not st.session_state.pipeline_ready:
        with st.spinner("Loading RAG pipeline..."):
            rag_chain, retriever, doc_count = init_rag_pipeline()
            if rag_chain is not None:
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.doc_count = doc_count
                st.session_state.pipeline_ready = True

    if st.session_state.pipeline_ready:
        st.markdown(f"""
        <div class="status-card">
            <div class="label">Pipeline Status</div>
            <div class="value status-online">● Online</div>
        </div>
        <div class="status-card">
            <div class="label">LLM Model</div>
            <div class="value">{GROQ_MODEL_NAME}</div>
        </div>
        <div class="status-card">
            <div class="label">Documents Indexed</div>
            <div class="value">{st.session_state.doc_count} chunks</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <div class="label">Pipeline Status</div>
            <div class="value status-offline">● No Documents</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("Upload documents and click 'Ingest' to get started.")

    st.divider()

    # ── Document Upload ──
    st.markdown("### Upload Documents")

    uploaded_files = st.file_uploader(
        "Drop healthcare documents here",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to uploads/")

    # ── Ingest Button ──
    if st.button("Ingest Documents", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            success, doc_count, chunk_count = run_ingestion_pipeline()

            if success:
                # Reinitialize pipeline
                rag_chain, retriever, total = init_rag_pipeline()
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.doc_count = total
                st.session_state.pipeline_ready = True
                st.success(f"Ingested {doc_count} docs into {chunk_count} chunks!")
                st.rerun()
            else:
                st.error("No documents found in Data/uploads/")

    st.divider()

    # ── Clear Chat ──
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── Sample Queries ──
    st.markdown("### Try These Queries")
    sample_queries = [
        "What is the discharge procedure?",
        "What is the duty roster?",
        "What are the emergency contacts?",
        "What medications were prescribed?",
        "What is the escalation matrix?",
    ]
    for q in sample_queries:
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.08);
            border: 1px solid rgba(102, 126, 234, 0.15);
            border-radius: 8px;
            padding: 0.5rem 0.8rem;
            margin-bottom: 0.4rem;
            color: rgba(255,255,255,0.7);
            font-size: 0.8rem;
            cursor: pointer;
        ">
            {q}
        </div>
        """, unsafe_allow_html=True)


# ─── Main Content Area ──────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 MediAssist AI</h1>
    <p>Ask questions about hospital operations, discharge summaries, and clinical workflows</p>
</div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍⚕️" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📄 View Source Documents", expanded=False):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">Source {i}: {src['filename']} | Page: {src['page']}</div>
                        <div class="source-text">{src['content'][:300]}...</div>
                    </div>
                    """, unsafe_allow_html=True)


# ─── Chat Input ─────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your healthcare documents..."):

    # Check if pipeline is ready
    if not st.session_state.pipeline_ready:
        st.error("Please upload and ingest documents first using the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching documents & generating answer..."):
                try:
                    # Get answer from RAG chain
                    answer = ask_question(st.session_state.rag_chain, prompt)

                    # Get source documents
                    source_docs = retrieve_documents(st.session_state.retriever, prompt)
                    sources = []
                    for doc in source_docs:
                        sources.append({
                            "filename": doc.metadata.get("source_filename", "Unknown"),
                            "page": doc.metadata.get("page", "N/A"),
                            "content": doc.page_content
                        })

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    if sources:
                        with st.expander("📄 View Source Documents", expanded=False):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-title">Source {i}: {src['filename']} | Page: {src['page']}</div>
                                    <div class="source-text">{src['content'][:300]}...</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
