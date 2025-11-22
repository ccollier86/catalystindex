"""Catalyst Index FastAPI application with Hydra configuration.

This module provides the main entry point for the Catalyst Index API server
with composable Hydra configuration management.

Usage:
    # Development (default: env=dev)
    python -m catalystindex.main

    # Production
    python -m catalystindex.main env=prod

    # Production with S3 artifacts
    python -m catalystindex.main env=prod storage=s3

    # Custom overrides
    python -m catalystindex.main env=staging storage.qdrant.host=qdrant-staging

Key Features:
    - Composable YAML configuration with Hydra
    - CLI overrides for any config value
    - Type-safe config validation via Pydantic
    - Support for environment-specific configs (dev/staging/prod)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import uvicorn
from fastapi import FastAPI
from omegaconf import DictConfig, OmegaConf

from .api.routes import router
from .config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create FastAPI application instance.

    Returns:
        FastAPI application instance

    Note:
        This function uses get_settings() which returns either:
        - Environment variable-based settings (default)
        - Hydra-injected settings (if set_settings() was called)
    """
    settings = get_settings()
    app = FastAPI(title="Catalyst Index RAG", version="0.1.0")
    app.include_router(router)
    return app


# Create app instance for ASGI servers (uvicorn, gunicorn, etc.)
app = create_app()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for Hydra-enabled Catalyst Index API server.

    Args:
        cfg: Hydra configuration (DictConfig) composed from YAML files + CLI overrides

    The function:
    1. Loads composed Hydra configuration
    2. Converts to Pydantic Settings for validation
    3. Injects settings into application context
    4. Starts Uvicorn server with configured parameters
    """
    # Import here to avoid circular dependencies
    from catalystindex.config.hydra_adapter import hydra_to_pydantic
    from catalystindex.config.settings import set_settings

    # Log configuration info
    logger.info("=" * 80)
    logger.info("Catalyst Index API Server")
    logger.info("=" * 80)
    logger.info(f"Environment: {cfg.environment}")
    logger.info(f"Config path: {Path('configs').absolute()}")

    # Show key configuration
    logger.info("Configuration:")
    logger.info(f"  - Environment: {cfg.environment}")
    logger.info(f"  - Vector Backend: {cfg.storage.vector_backend}")
    logger.info(f"  - Embeddings Provider: {cfg.embeddings.provider} ({cfg.embeddings.model})")
    logger.info(f"  - Reranker: {cfg.reranker.provider} (enabled: {cfg.reranker.enabled})")
    logger.info(f"  - Semantic Cache: {cfg.semantic_cache.enabled}")
    logger.info(f"  - API: {cfg.api.host}:{cfg.api.port} (workers: {cfg.api.workers})")

    # Debug: Show full config in dev mode
    if cfg.environment == "dev":
        logger.debug("Full Configuration:")
        logger.debug(OmegaConf.to_yaml(cfg))

    try:
        # Convert Hydra config to Pydantic Settings
        logger.info("Converting Hydra config to Pydantic Settings...")
        settings = hydra_to_pydantic(cfg)

        # Inject settings into application context
        logger.info("Injecting settings into application context...")
        set_settings(settings)

        # Create FastAPI application
        logger.info("Creating FastAPI application...")
        app = create_app()

        logger.info("=" * 80)
        logger.info(f"Starting server at http://{cfg.api.host}:{cfg.api.port}")
        logger.info("=" * 80)

        # Start Uvicorn server with configured parameters
        uvicorn.run(
            app,
            host=cfg.api.host,
            port=cfg.api.port,
            reload=cfg.api.reload,
            workers=cfg.api.workers if not cfg.api.reload else 1,  # reload incompatible with workers
            log_level="info",
            access_log=True,
        )

    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


__all__ = ["create_app", "app", "main"]


if __name__ == "__main__":
    main()
