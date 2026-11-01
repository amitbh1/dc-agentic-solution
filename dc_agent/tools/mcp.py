"""MCP servers as role-scoped enrichment tools (for example a runbook or incident search).

Configure with DC_MCP_SERVERS and DC_MCP_TOOLS (see core/config.py). Only the question text is sent
to the tool, as the argument named in DC_MCP_TOOLS[tool].arg; result rows never leave the agent.
"""
from __future__ import annotations

import asyncio

from .registry import ToolRegistry, ToolSpec


async def load_mcp_tools(servers: dict) -> list:
    """servers example: {"runbooks": {"url": "https://mcp.internal/runbooks", "transport": "streamable_http"}}"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return await MultiServerMCPClient(servers).get_tools()


def mcp_spec(tool, roles, arg: str = "query", timeout_s: float = 5.0) -> ToolSpec:
    def call(question: str, **_context):
        # Runs in the registry's worker thread, which has no event loop of its own.
        return asyncio.run(tool.ainvoke({arg: question}))

    return ToolSpec(tool.name, call, frozenset(roles), timeout_s)


async def build_mcp_registry(servers: dict, tool_config: dict) -> ToolRegistry:
    """Register only the tools named in tool_config; any other tool the server offers is ignored."""
    registry = ToolRegistry()
    for tool in await load_mcp_tools(servers):
        cfg = tool_config.get(tool.name)
        if cfg:
            registry.register(mcp_spec(tool, cfg["roles"], cfg.get("arg", "query"), cfg.get("timeout_s", 5.0)))
    return registry
