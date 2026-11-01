from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..bootstrap import build_agent
from ..core.config import Settings
from ..observability import configure_logging
from .middleware import BodyLimit, request_context
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = Settings.from_env()
    tools = None
    if settings.mcp_servers and settings.mcp_tools:
        from ..tools import build_mcp_registry

        tools = await build_mcp_registry(settings.mcp_servers, settings.mcp_tools)
    app.state.bundle = build_agent(settings, tools=tools)
    yield
    app.state.bundle.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Data centre agent", lifespan=lifespan)
    app.middleware("http")(request_context)
    app.add_middleware(BodyLimit)
    app.include_router(router)
    return app


app = create_app()
