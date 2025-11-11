from __future__ import annotations

from typing import Dict

from .base import ParserAdapter, PlainTextParser


class ParserRegistry:
    """Registry that maps parser names to adapters."""

    def __init__(self) -> None:
        self._parsers: Dict[str, ParserAdapter] = {}

    def register(self, name: str, adapter: ParserAdapter) -> None:
        self._parsers[name] = adapter

    def resolve(self, name: str) -> ParserAdapter:
        try:
            return self._parsers[name]
        except KeyError as exc:
            raise KeyError(f"Parser '{name}' is not registered") from exc


def default_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register("plain_text", PlainTextParser())
    return registry


__all__ = ["ParserRegistry", "default_registry"]
