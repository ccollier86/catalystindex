"""Hydra ↔ Pydantic adapter for backward compatibility.

This module provides bridge functions to convert between Hydra's DictConfig
and our existing Pydantic Settings models. This enables gradual migration
from environment variables to Hydra configuration.

Usage:
    # Convert Hydra config to Pydantic Settings
    from hydra import compose, initialize
    from catalystindex.config.hydra_adapter import hydra_to_pydantic

    with initialize(config_path="../../../configs"):
        hydra_cfg = compose(config_name="config", overrides=["env=prod"])
        pydantic_settings = hydra_to_pydantic(hydra_cfg)

    # Convert Pydantic Settings to Hydra DictConfig
    from catalystindex.config.hydra_adapter import pydantic_to_hydra

    settings = get_settings()
    hydra_cfg = pydantic_to_hydra(settings)
"""

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from .settings import Settings


def hydra_to_pydantic(hydra_config: DictConfig) -> Settings:
    """Convert Hydra DictConfig to Pydantic Settings.

    Args:
        hydra_config: Hydra configuration object

    Returns:
        Pydantic Settings instance

    Example:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../../../configs"):
        ...     cfg = compose(config_name="config")
        ...     settings = hydra_to_pydantic(cfg)
        >>> settings.environment
        'dev'
    """
    # Convert DictConfig to plain dict (resolves interpolations)
    config_dict = OmegaConf.to_container(hydra_config, resolve=True, throw_on_missing=True)

    # Build nested dict matching Pydantic Settings structure
    pydantic_dict = _build_pydantic_dict(config_dict)

    # Create Pydantic Settings from dict
    return Settings(**pydantic_dict)


def pydantic_to_hydra(settings: Settings) -> DictConfig:
    """Convert Pydantic Settings to Hydra DictConfig.

    Args:
        settings: Pydantic Settings instance

    Returns:
        Hydra DictConfig object

    Example:
        >>> settings = get_settings()
        >>> hydra_cfg = pydantic_to_hydra(settings)
        >>> hydra_cfg.environment
        'dev'
    """
    # Convert Pydantic model to dict
    settings_dict = settings.model_dump()

    # Build nested dict matching Hydra structure
    hydra_dict = _build_hydra_dict(settings_dict)

    # Create DictConfig from dict
    return OmegaConf.create(hydra_dict)


def _build_pydantic_dict(hydra_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build dict structure matching Pydantic Settings from Hydra dict.

    Transforms Hydra's flat/nested structure to match Pydantic field names.

    Args:
        hydra_dict: Dictionary from Hydra config

    Returns:
        Dictionary matching Pydantic Settings structure
    """
    pydantic_dict: Dict[str, Any] = {}

    # Map Hydra keys to Pydantic Settings structure
    # Most keys map directly, but some require transformation

    # Direct mappings
    if "environment" in hydra_dict:
        pydantic_dict["environment"] = hydra_dict["environment"]

    # Security settings
    if "security" in hydra_dict:
        pydantic_dict["security"] = hydra_dict["security"]

    # Storage settings (flatten nested structure)
    if "storage" in hydra_dict:
        storage = hydra_dict["storage"]
        pydantic_dict["storage"] = {
            "vector_backend": storage.get("vector_backend", "qdrant"),
            "vector_dimension": storage.get("vector_dimension", 3072),
            "term_index_backend": storage.get("term_index_backend", "redis"),
            "qdrant": storage.get("qdrant", {}),
            "redis": storage.get("redis", {}),
            "artifacts": storage.get("artifacts", {}),
        }

    # Embeddings settings
    if "embeddings" in hydra_dict:
        pydantic_dict["embeddings"] = hydra_dict["embeddings"]

    # Reranker settings
    if "reranker" in hydra_dict:
        pydantic_dict["reranker"] = hydra_dict["reranker"]

    # Semantic cache settings
    if "semantic_cache" in hydra_dict:
        pydantic_dict["semantic_cache"] = hydra_dict["semantic_cache"]

    # Jobs settings
    if "jobs" in hydra_dict:
        pydantic_dict["jobs"] = hydra_dict["jobs"]

    # Features settings
    if "features" in hydra_dict:
        pydantic_dict["features"] = hydra_dict["features"]

    # Metrics settings
    if "metrics" in hydra_dict:
        pydantic_dict["metrics"] = hydra_dict["metrics"]

    # Acquisition settings
    if "acquisition" in hydra_dict:
        pydantic_dict["acquisition"] = hydra_dict["acquisition"]

    # Policy advisor settings
    if "policy_advisor" in hydra_dict:
        pydantic_dict["policy_advisor"] = hydra_dict["policy_advisor"]

    # Policy synthesis settings
    if "policy_synthesis" in hydra_dict:
        pydantic_dict["policy_synthesis"] = hydra_dict["policy_synthesis"]

    return pydantic_dict


def _build_hydra_dict(settings_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build dict structure matching Hydra config from Pydantic dict.

    Transforms Pydantic Settings structure to match Hydra config layout.

    Args:
        settings_dict: Dictionary from Pydantic Settings

    Returns:
        Dictionary matching Hydra config structure
    """
    # For now, Pydantic and Hydra structures are similar enough
    # that we can mostly pass through. Future optimization can add
    # specific transformations if needed.
    return settings_dict


def merge_with_env_vars(hydra_config: DictConfig) -> DictConfig:
    """Merge Hydra config with environment variables.

    This allows environment variables to override Hydra config values,
    maintaining backward compatibility during migration.

    Args:
        hydra_config: Hydra configuration

    Returns:
        Merged configuration with env var overrides

    Example:
        >>> import os
        >>> os.environ["CATALYST_SECURITY__jwt_secret"] = "prod-secret"
        >>> cfg = merge_with_env_vars(hydra_config)
        >>> cfg.security.jwt_secret
        'prod-secret'
    """
    # Environment variables are already resolved via ${oc.env:VAR} interpolation
    # in the Hydra configs, so this is mostly a pass-through.
    # This function exists for future extensibility.
    return hydra_config


__all__ = [
    "hydra_to_pydantic",
    "pydantic_to_hydra",
    "merge_with_env_vars",
]
