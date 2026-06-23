"""
MediAssist AI — AI Agent
Connects FAISS unstructured retrieval and PostgreSQL MCP tools
using a LangGraph ReAct agent.
"""

import os
import asyncio
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Import config
try:
    from backend.config import GROQ_API_KEY, GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
    from mcp_client import get_mcp_tools
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.config import GROQ_API_KEY, GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
    from mcp_client import get_mcp_tools


# ─── Healthcare Agent System Prompt ──────────────────────
SYSTEM_PROMPT = """You are MediAssist AI, an intelligent healthcare information assistant.
You have access to two types of data sources:
1. Unstructured hospital documents (SOPs, summaries) via the search_hospital_documents tool.
2. Structured patient database (PostgreSQL) via your MCP tools (e.g., get_patient_details, get_patient_appointments).

RULES:
1. If the user asks about general procedures, guidelines, or hospital policies, ALWAYS use the `search_hospital_documents` tool.
2. If the user asks about a specific patient's details, demographics, or appointments, ALWAYS use the MCP tools.
3. You can use BOTH if a question requires combining unstructured rules with patient structured data.
4. Answer ONLY based on the tools' outputs. Do NOT make up medical information or appointments.
5. If the tools return no information, state that clearly.
6. Be precise, professional, and empathetic.
"""

def _format_docs(docs):
    """Format retrieved documents into a single context string with source info."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source {i}: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_llm():
    """Initialize the Groq LLM."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found! Please add it to your .env file.")

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS
    )


def build_rag_chain(retriever):
    """
    Build the LangGraph AI Agent equipped with FAISS and MCP tools.
    
    Args:
        retriever: LangChain retriever instance for FAISS.
        
    Returns:
        Compiled LangGraph Agent Executor.
    """
    llm = get_llm()
    
    # 1. Create the FAISS RAG Search Tool
    def search_docs(query: str) -> str:
        """Search hospital documents (PDFs, DOCX) for procedures, summaries, or general text."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found in the database."
        return _format_docs(docs)
        
    rag_tool = Tool(
        name="search_hospital_documents",
        description="Search unstructured hospital documents (PDFs, SOPs, Summaries) for text-based information.",
        func=search_docs
    )
    
    # 2. Get the PostgreSQL MCP Tools
    try:
        mcp_tools = asyncio.run(get_mcp_tools())
        print(f"  [OK] Successfully loaded {len(mcp_tools)} MCP tools for PostgreSQL.")
    except Exception as e:
        print(f"  [WARNING] Could not connect to MCP Server: {e}")
        mcp_tools = []
        
    # Combine all tools
    tools = [rag_tool] + mcp_tools
    
    # 3. Create the Agent
    agent_executor = create_react_agent(
        llm, 
        tools,
        prompt=SYSTEM_PROMPT
    )
    
    return agent_executor


def ask_question(agent_executor, question):
    """
    Ask a question to the Agent and get an answer.
    
    Args:
        agent_executor: The built LangGraph agent.
        question: User's question string.
        
    Returns:
        Answer string from the LLM.
    """
    response = agent_executor.invoke({"messages": [HumanMessage(content=question)]})
    # The final answer is the last message content
    return response["messages"][-1].content
