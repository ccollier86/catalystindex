#!/usr/bin/env python3
"""Catalyst Index RQ worker entry point with Hydra configuration.

This module provides the entry point for starting Catalyst Index RQ workers
with composable Hydra configuration management.

Usage:
    # Development (default: env=dev)
    python scripts/run_worker.py

    # Production
    python scripts/run_worker.py env=prod

    # Production with custom worker settings
    python scripts/run_worker.py env=prod workers.max_active_docs=16

    # Custom queue and database
    python scripts/run_worker.py \
        workers.queue_name=priority \
        db.postgres_dsn=postgresql://user:pass@db:5432/jobs

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
from omegaconf import DictConfig, OmegaConf

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for Catalyst Index RQ worker.

    Args:
        cfg: Hydra configuration (DictConfig) composed from YAML files + CLI overrides

    The function:
    1. Loads composed Hydra configuration
    2. Converts to Pydantic Settings for validation
    3. Injects settings into application context
    4. Starts RQ worker with configured parameters
    """
    # Import here to avoid circular dependencies
    from catalystindex.config.hydra_adapter import hydra_to_pydantic
    from catalystindex.config.settings import set_settings

    # Log configuration info
    logger.info("=" * 80)
    logger.info("Catalyst Index RQ Worker")
    logger.info("=" * 80)
    logger.info(f"Environment: {cfg.environment}")
    logger.info(f"Config path: {Path('../configs').absolute()}")

    # Check if worker is enabled
    if not cfg.jobs.worker.enabled:
        logger.error("Worker is disabled in configuration!")
        logger.error("Set CATALYST_JOBS__worker__enabled=true or use workers=default config")
        sys.exit(1)

    # Show key configuration
    logger.info("Worker Configuration:")
    logger.info(f"  - Queue: {cfg.jobs.worker.queue_name}")
    logger.info(f"  - Max Active Docs: {cfg.jobs.worker.max_active_docs}")
    logger.info(f"  - Max Queue Length: {cfg.jobs.worker.max_queue_length}")
    logger.info(f"  - Default Timeout: {cfg.jobs.worker.default_timeout}s")
    logger.info(f"  - Max Retries: {cfg.jobs.worker.max_retries}")
    logger.info(f"  - LLM Batch Size: {cfg.jobs.worker.llm_batch_size}")
    logger.info(f"  - LLM Max Workers: {cfg.jobs.worker.llm_max_workers}")

    logger.info("\nStorage Configuration:")
    logger.info(f"  - Vector Backend: {cfg.storage.vector_backend}")
    logger.info(f"  - Embeddings: {cfg.embeddings.provider} ({cfg.embeddings.model})")
    logger.info(f"  - Redis URL: {cfg.jobs.store.redis_url}")
    logger.info(f"  - Postgres DSN: {cfg.jobs.store.postgres_dsn[:50]}...")

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

        # Import RQ worker components (after settings injection)
        logger.info("Importing RQ worker components...")
        import redis
        from rq import Worker

        # Connect to Redis
        logger.info(f"Connecting to Redis at {settings.jobs.store.redis_url}...")
        redis_conn = redis.from_url(settings.jobs.store.redis_url)

        # Test connection
        redis_conn.ping()
        logger.info("Redis connection successful!")

        # Create worker
        queue_name = settings.jobs.worker.queue_name
        logger.info(f"Creating worker for queue: {queue_name}")

        worker = Worker(
            queues=[queue_name],
            connection=redis_conn,
            name=f"catalyst-worker-{queue_name}",
        )

        logger.info("=" * 80)
        logger.info(f"Starting RQ worker for queue: {queue_name}")
        logger.info("=" * 80)

        # Start worker (blocking)
        worker.work(with_scheduler=False)

    except KeyboardInterrupt:
        logger.info("\nShutting down worker gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start worker: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
