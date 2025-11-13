from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

try:  # pragma: no cover - optional dependency import guard
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass(slots=True)
class PolicyAdvice:
    policy_name: Optional[str]
    confidence: Optional[float]
    tags: Dict[str, str]
    notes: Optional[str]


class PolicyAdvisor:
    """LLM-backed advisor for selecting ingestion policies."""

    def __init__(
        self,
        *,
        enabled: bool,
        provider: str,
        model: str,
        api_key: str | None,
        base_url: str | None,
        sample_chars: int,
    ) -> None:
        self._enabled = enabled and provider.lower() == "openai"
        self._sample_chars = sample_chars
        self._model = model
        if not self._enabled:
            self._client = None
            return
        if OpenAI is None:
            raise RuntimeError("OpenAI client not installed; install the 'openai' extra")
        if not api_key:
            raise RuntimeError("Policy advisor requires OPENAI_API_KEY or CATALYST_POLICY_ADVISOR__api_key")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def enabled(self) -> bool:
        return bool(self._client)

    def advise(self, *, title: str, schema: str | None, content: str) -> PolicyAdvice:
        if not self.enabled:
            return PolicyAdvice(policy_name=None, confidence=None, tags={}, notes=None)
        prompt = (
            "You are assisting a retrieval system by labeling documents with the best ingestion recipe. "
            "Known recipes: dsm5, treatment_planner, summary, general. If none match, return 'general'.\n"
            f"Title: {title}\n"
            f"Provided schema hint: {schema or 'none'}\n"
            "Document preview:\n"
            f"{content[: self._sample_chars]}\n"
            "Respond strictly in JSON with fields policy_name, confidence (0-1), tags (object), notes."
        )
        response = self._client.responses.create(  # type: ignore[union-attr]
            model=self._model,
            input=[{"role": "user", "content": prompt}],
        )
        text = response.output[0].content[0].text  # type: ignore[attr-defined]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return PolicyAdvice(policy_name=None, confidence=None, tags={}, notes=text)
        return PolicyAdvice(
            policy_name=data.get("policy_name"),
            confidence=data.get("confidence"),
            tags={k: str(v) for k, v in (data.get("tags", {}) or {}).items()},
            notes=data.get("notes"),
        )


__all__ = ["PolicyAdvisor", "PolicyAdvice"]
