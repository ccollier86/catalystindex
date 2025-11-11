from __future__ import annotations

import logging

from fastapi import FastAPI

from .api.routes import router
from .config.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="Catalyst Index RAG", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()


__all__ = ["create_app", "app"]
