"""Role-scoped enrichment tools and the MCP loader.

When a registry has tools, the graph adds an `enrich` step after the result check: each tool the
user's role may use is called with a row-free context and may return a short note for the answer.
"""
from .mcp import build_mcp_registry, load_mcp_tools, mcp_spec
from .registry import ToolRegistry, ToolSpec

__all__ = ["ToolRegistry", "ToolSpec", "build_mcp_registry", "load_mcp_tools", "mcp_spec"]
