# -*- coding: utf-8 -*-
"""FastMCP Server Instance"""

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from app.services.mcp.tools import ask_grok_impl
from app.core.config import setting


def create_mcp_server() -> FastMCP:
    """Create MCP server instance, enable authentication if API key is configured"""
    # Check if API key is configured
    api_key = setting.grok_config.get("api_key")
    
    # If API key is configured, enable static token verification
    auth = None
    if api_key:
        auth = StaticTokenVerifier(
            tokens={
                api_key: {
                    "client_id": "grok2api-client",
                    "scopes": ["read", "write", "admin"]
                }
            },
            required_scopes=["read"]
        )
    
    # Create FastMCP instance
    return FastMCP(
        name="Grok2API-MCP",
        instructions="MCP server providing Grok AI chat capabilities. Use ask_grok tool to interact with Grok AI models.",
        auth=auth
    )


# Create global MCP instance
mcp = create_mcp_server()


# Register ask_grok tool
@mcp.tool
async def ask_grok(
    query: str,
    model: str = "grok-3-fast",
    system_prompt: str = None
) -> str:
    """
    Call Grok AI for conversation, especially suitable when users ask for the latest information, need to use search functions, or want to know about social platform dynamics (such as Twitter(X), Reddit, etc.).

    Args:
        query: User's question or instruction
        model: Grok model name, optional values: grok-3-fast (default), grok-4-fast, grok-4-fast-expert, grok-4-expert, grok-4-heavy
        system_prompt: Optional system prompt, used to set AI role or behavior constraints

    Returns:
        Complete response content from Grok AI, may include text and image links (Markdown format)

    Examples:
        - Simple Q&A: ask_grok("What is Python?")
        - Specific model: ask_grok("Explain quantum computing", model="grok-4-fast")
        - With system prompt: ask_grok("Write a poem", system_prompt="You are a classical poet")
    """
    return await ask_grok_impl(query, model, system_prompt)
