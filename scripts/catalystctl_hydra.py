#!/usr/bin/env python3
"""Hydra-enabled CLI configuration tool for Catalyst Index.

This module provides a CLI tool for inspecting, validating, and testing
Hydra configurations before deploying them. It supports multiple commands
for different configuration management tasks.

Usage:
    # Show current configuration (dev default)
    python scripts/catalystctl_hydra.py show

    # Show production configuration
    python scripts/catalystctl_hydra.py show env=prod

    # Show production with S3 storage
    python scripts/catalystctl_hydra.py show env=prod storage=s3

    # Validate configuration
    python scripts/catalystctl_hydra.py validate env=prod

    # Export configuration to JSON
    python scripts/catalystctl_hydra.py export --format json env=prod

    # Export configuration to YAML
    python scripts/catalystctl_hydra.py export --format yaml env=prod

    # Test configuration with overrides
    python scripts/catalystctl_hydra.py test storage.qdrant.host=custom-host

Key Features:
    - Display composed configurations
    - Validate Hydra → Pydantic conversion
    - Export configs to JSON/YAML
    - Test configuration overrides
    - Preview what settings will be used at runtime
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import hydra
from omegaconf import DictConfig, OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Catalyst Index configuration management CLI (Hydra mode)."""
    pass


@cli.command()
@click.argument("overrides", nargs=-1)
def show(overrides: tuple[str, ...]) -> None:
    """Display current composed configuration.

    Args:
        overrides: Hydra CLI overrides (e.g., env=prod storage=s3)

    Examples:
        catalystctl_hydra.py show
        catalystctl_hydra.py show env=prod
        catalystctl_hydra.py show env=prod storage=s3
    """
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).parent.parent / "configs"
    config_dir = config_dir.absolute()

    logger.info("=" * 80)
    logger.info("Catalyst Index Configuration (Hydra Mode)")
    logger.info("=" * 80)
    logger.info(f"Config directory: {config_dir}")
    logger.info(f"Overrides: {list(overrides) if overrides else 'None'}")
    logger.info("=" * 80)

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=list(overrides))

            # Display as YAML (more readable)
            print(OmegaConf.to_yaml(cfg))

            # Show key info
            print("\n" + "=" * 80)
            print("Key Configuration Summary:")
            print("=" * 80)
            print(f"Environment: {cfg.environment}")
            print(f"Vector Backend: {cfg.storage.vector_backend}")
            print(f"Embeddings: {cfg.embeddings.provider} ({cfg.embeddings.model})")
            print(f"Reranker: {cfg.reranker.provider} (enabled: {cfg.reranker.enabled})")
            print(f"Semantic Cache: {cfg.semantic_cache.enabled}")
            print(f"API: {cfg.api.host}:{cfg.api.port}")
            print(f"Worker Enabled: {cfg.jobs.worker.enabled}")
            if cfg.jobs.worker.enabled:
                print(f"Worker Queue: {cfg.jobs.worker.queue_name}")

    except Exception as e:
        logger.error(f"Failed to compose configuration: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("overrides", nargs=-1)
def validate(overrides: tuple[str, ...]) -> None:
    """Validate configuration by converting to Pydantic Settings.

    This command tests the full Hydra → Pydantic conversion pipeline
    to ensure the configuration is valid and can be used at runtime.

    Args:
        overrides: Hydra CLI overrides (e.g., env=prod storage=s3)

    Examples:
        catalystctl_hydra.py validate
        catalystctl_hydra.py validate env=prod
    """
    from hydra import compose, initialize_config_dir

    from catalystindex.config.hydra_adapter import hydra_to_pydantic

    config_dir = Path(__file__).parent.parent / "configs"
    config_dir = config_dir.absolute()

    logger.info("=" * 80)
    logger.info("Validating Catalyst Index Configuration")
    logger.info("=" * 80)
    logger.info(f"Config directory: {config_dir}")
    logger.info(f"Overrides: {list(overrides) if overrides else 'None'}")

    try:
        # Step 1: Compose Hydra config
        logger.info("Step 1: Composing Hydra configuration...")
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=list(overrides))
        logger.info("✅ Hydra configuration composed successfully")

        # Step 2: Convert to Pydantic
        logger.info("Step 2: Converting to Pydantic Settings...")
        settings = hydra_to_pydantic(cfg)
        logger.info("✅ Pydantic Settings created successfully")

        # Step 3: Validate all fields
        logger.info("Step 3: Validating Pydantic Settings...")
        settings_dict = settings.model_dump()
        logger.info("✅ All settings validated successfully")

        # Show summary
        print("\n" + "=" * 80)
        print("Validation Summary")
        print("=" * 80)
        print(f"Environment: {settings.environment}")
        print(f"Storage Backend: {settings.storage.vector_backend}")
        print(f"Embeddings Provider: {settings.embeddings.provider}")
        print(f"API Host: {settings.api.host}:{settings.api.port}")
        print(f"Worker Enabled: {settings.jobs.worker.enabled}")
        print("\n✅ Configuration is valid and ready for use!")

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["json", "yaml"]),
    default="yaml",
    help="Output format (json or yaml)",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: stdout)")
@click.argument("overrides", nargs=-1)
def export(format: str, output: Optional[str], overrides: tuple[str, ...]) -> None:
    """Export composed configuration to JSON or YAML.

    Args:
        format: Output format (json or yaml)
        output: Output file path (if not provided, prints to stdout)
        overrides: Hydra CLI overrides

    Examples:
        catalystctl_hydra.py export --format json env=prod
        catalystctl_hydra.py export --format yaml -o config.yaml env=prod storage=s3
    """
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).parent.parent / "configs"
    config_dir = config_dir.absolute()

    logger.info(f"Exporting configuration to {format.upper()}...")

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=list(overrides))

            # Convert to desired format
            if format == "json":
                # Convert to plain dict then to JSON
                config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                output_str = json.dumps(config_dict, indent=2)
            else:  # yaml
                output_str = OmegaConf.to_yaml(cfg)

            # Output to file or stdout
            if output:
                output_path = Path(output)
                output_path.write_text(output_str)
                logger.info(f"✅ Configuration exported to: {output_path}")
            else:
                print(output_str)

    except Exception as e:
        logger.error(f"Failed to export configuration: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("overrides", nargs=-1)
def test(overrides: tuple[str, ...]) -> None:
    """Test configuration with overrides (dry run).

    This command shows what the final configuration will be with your
    overrides, without actually starting any services. Useful for
    testing configuration changes before deployment.

    Args:
        overrides: Hydra CLI overrides to test

    Examples:
        catalystctl_hydra.py test storage.qdrant.host=custom-host
        catalystctl_hydra.py test env=prod workers.max_active_docs=16
    """
    from hydra import compose, initialize_config_dir

    from catalystindex.config.hydra_adapter import hydra_to_pydantic

    config_dir = Path(__file__).parent.parent / "configs"
    config_dir = config_dir.absolute()

    logger.info("=" * 80)
    logger.info("Testing Configuration Overrides (Dry Run)")
    logger.info("=" * 80)
    logger.info(f"Overrides: {list(overrides) if overrides else 'None'}")

    if not overrides:
        logger.warning("No overrides provided. Use: catalystctl test KEY=VALUE")
        sys.exit(1)

    try:
        # Compose with overrides
        logger.info("Composing configuration with overrides...")
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=list(overrides))

        # Convert to Pydantic for validation
        logger.info("Validating configuration...")
        settings = hydra_to_pydantic(cfg)

        # Show affected sections
        print("\n" + "=" * 80)
        print("Configuration Test Results")
        print("=" * 80)

        # Parse overrides to show affected sections
        for override in overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                print(f"\n Override: {key} = {value}")

                # Try to show the resolved value
                try:
                    # Navigate nested keys
                    parts = key.split(".")
                    current = cfg
                    for part in parts:
                        current = getattr(current, part, None) or current.get(part)
                    print(f"   Resolved: {current}")
                except Exception as e:
                    print(f"   Could not resolve: {e}")

        # Show full YAML for reference
        print("\n" + "=" * 80)
        print("Full Composed Configuration:")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))

        logger.info("✅ Configuration test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
