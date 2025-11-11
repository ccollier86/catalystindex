from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


class TermIndex:
    """In-memory alias index for query expansion and feedback."""

    def __init__(self) -> None:
        self._aliases: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._document_terms: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    def update(self, document_id: str, chunk_id: str, aliases: Iterable[str]) -> None:
        key = f"{document_id}:{chunk_id}"
        for rank, alias in enumerate(alias for alias in aliases if alias):
            normalized = alias.lower().strip()
            if not normalized:
                continue
            weight = max(0.1, 1.0 - rank * 0.1)
            existing = self._aliases[normalized].get(key, 0.0)
            if weight > existing:
                self._aliases[normalized][key] = weight
            if normalized not in self._document_terms[document_id][chunk_id]:
                self._document_terms[document_id][chunk_id].append(normalized)

    def expand_query(self, query: str, *, limit: int = 5) -> List[str]:
        if not query.strip():
            return []
        tokens = {token for token in _tokenize(query) if token}
        if not tokens:
            return []
        scored: List[Tuple[str, float]] = []
        for alias, mapping in self._aliases.items():
            if alias in tokens:
                continue
            if not any(token in alias for token in tokens):
                continue
            total_score = sum(mapping.values())
            scored.append((alias, total_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [alias for alias, _ in scored[:limit]]

    def record_feedback(self, query: str, chunk_ids: Iterable[str], *, positive: bool) -> None:
        adjustment = 0.2 if positive else -0.3
        tokens = {token for token in _tokenize(query)}
        for alias, mapping in self._aliases.items():
            if not tokens.intersection(alias.split()):
                continue
            for chunk_key in list(mapping):
                if any(chunk_key.endswith(chunk_id) for chunk_id in chunk_ids):
                    mapping[chunk_key] = max(0.05, mapping[chunk_key] + adjustment)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]{3,}", text.lower())


__all__ = ["TermIndex"]

