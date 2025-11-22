"""FastAPI App Main Entry Point"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.core.logger import logger
from app.core.exception import register_exception_handlers
from app.core.storage import storage_manager
from app.core.config import setting
from app.services.grok.token import token_manager
from app.api.v1.chat import router as chat_router
from app.api.v1.models import router as models_router
from app.api.v1.images import router as images_router
from app.api.admin.manage import router as admin_router

# Import MCP Server (Authentication is configured in server.py)
from app.services.mcp import mcp

# Create MCP FastAPI app instance
# Use streamable HTTP transport, supports efficient bidirectional streaming communication
mcp_app = mcp.http_app(stateless_http=True, transport="streamable-http")

# 2. Define application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup Sequence:
    1. Initialize core services (storage, settings, token_manager)
    2. Start MCP service lifecycle
    
    Shutdown Sequence (LIFO):
    1. Shutdown MCP service lifecycle
    2. Shutdown core services
    """
    # --- Startup Process ---
    # 1. Initialize core services
    await storage_manager.init()

    # Set storage to settings and token manager
    storage = storage_manager.get_storage()
    setting.set_storage(storage)
    token_manager.set_storage(storage)
    
    # Reload configuration and token data
    await setting.reload()
    token_manager._load_data()
    logger.info("[Grok2API] Core services initialization complete")

    # 2. Manage MCP service lifecycle
    mcp_lifespan_context = mcp_app.lifespan(app)
    await mcp_lifespan_context.__aenter__()
    logger.info("[MCP] MCP service initialization complete")

    logger.info("[Grok2API] Application started successfully")
    
    try:
        yield
    finally:
        # --- Shutdown Process ---
        # 1. Exit MCP service lifecycle
        await mcp_lifespan_context.__aexit__(None, None, None)
        logger.info("[MCP] MCP service closed")
        
        # 2. Shutdown core services
        await storage_manager.close()
        logger.info("[Grok2API] Application closed successfully")


# Initialize logger
logger.info("[Grok2API] Application is starting...")

# Create FastAPI application
app = FastAPI(
    title="Grok2API",
    description="Grok API Transformation Service",
    version="1.3.1",
    lifespan=lifespan
)

# Register global exception handlers
register_exception_handlers(app)

# Register routers
app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(images_router)
app.include_router(admin_router)

# Mount static files
app.mount("/static", StaticFiles(directory="app/template"), name="template")

@app.get("/")
async def root():
    """Root path"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/login")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Grok2API",
        "version": "1.0.3"
    }

# Mount MCP server
app.mount("", mcp_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
