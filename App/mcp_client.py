"""
MediAssist AI — MCP Client
Connects to the local MCP server via stdio and converts its tools to LangChain tools.
"""

import sys
import os
import asyncio
from langchain_core.tools import Tool
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Path to the MCP server script
SERVER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server", "server.py")

async def get_mcp_tools():
    """
    Connects to the MCP server and returns a list of LangChain Tools.
    Note: In a real production app, the session would be kept alive.
    For this prototype, we'll recreate the connection per query, or better, 
    keep it alive in Streamlit session state.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT_PATH]
    )
    
    tools = []
    
    # We use a context manager to ensure proper cleanup, but for LangChain tools 
    # that execute async, we might need a persistent session.
    # For simplicity, we will create synchronous wrappers that spin up a temporary client.
    
    def sync_call_tool(tool_name, arguments):
        async def _call():
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    # Result content is usually a list of text objects
                    return "\n".join([c.text for c in result.content if hasattr(c, 'text')])
        return asyncio.run(_call())

    # First, get the list of available tools from the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            
            for mcp_tool in mcp_tools.tools:
                # Create a LangChain tool wrapper for each MCP tool
                tool_name = mcp_tool.name
                tool_desc = mcp_tool.description or "MCP Tool"
                
                # We create a closure to capture the tool name
                def create_func(name):
                    def func(*args, **kwargs):
                        # Combine args and kwargs into a single dict for MCP arguments
                        # Assuming the tool takes a single 'patient_name' argument based on our server
                        arguments = kwargs
                        if args and not kwargs:
                            arguments = {"patient_name": args[0]}
                        return sync_call_tool(name, arguments)
                    return func
                
                tools.append(
                    Tool(
                        name=tool_name,
                        description=tool_desc,
                        func=create_func(tool_name)
                    )
                )
                
    return tools
