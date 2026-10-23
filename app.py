# Citadel FastAPI application.

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn

from citadel.api.abstract_api import (
    FastAPIApplication,
    FileResponse,
    StaticFiles,
)
from citadel.api.helpers import load_rag_pipeline as initialize_rag_pipeline
from citadel.api.services import ApplicationServices
from citadel.api.dependencies import settings
from citadel.api.routes import router
from citadel.infrastructure.database.sqlite_repository import SQLiteRepository

logger = logging.getLogger(__name__)

services = ApplicationServices()


def load_rag_pipeline():
    # Load the embedding, retrieval, generation, and database components.
    components = initialize_rag_pipeline(
        services.as_rag_components()
    )
    services.update_from(components)


@asynccontextmanager
async def lifespan(_app):
    if services.db is None:
        services.db = SQLiteRepository(db_path=settings.database_path)
    pipeline_task = None
    if settings.load_pipeline_in_background:
        pipeline_task = asyncio.create_task(asyncio.to_thread(load_rag_pipeline))
    else:
        load_rag_pipeline()
    try:
        yield
    finally:
        if pipeline_task is not None:
            await pipeline_task
        if services.db is not None:
            services.db.close()


app = FastAPIApplication(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
).application
import citadel.api.routes as api_routes

api_routes.services = services
app.include_router(router)


static_dir = settings.static_dir
app.mount("/css", StaticFiles(directory=static_dir / "css"), name="css")
app.mount("/js", StaticFiles(directory=static_dir / "js"), name="js")
app.mount("/images", StaticFiles(directory=static_dir / "images"), name="images")


@app.get("/")
def serve_ui():
    # Serve the main study interface.
    return FileResponse(static_dir / "index.html")

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)