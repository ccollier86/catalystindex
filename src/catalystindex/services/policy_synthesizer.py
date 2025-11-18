from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass(slots=True)
class PolicySynthesisResult:
    chunk_modes: List[str] | None = None
    window_size: int | None = None
    window_overlap: int | None = None
    max_chunk_tokens: int | None = None
    highlight_phrases: List[str] | None = None
    notes: str | None = None
    confidence: float | None = None

    def to_overrides(self) -> Dict[str, object]:
        overrides: Dict[str, object] = {}
        if self.chunk_modes:
            overrides["chunk_modes"] = tuple(self.chunk_modes)
        for key, value in (
            ("window_size", self.window_size),
            ("window_overlap", self.window_overlap),
            ("max_chunk_tokens", self.max_chunk_tokens),
        ):
            if isinstance(value, int) and value > 0:
                overrides[key] = value
        return overrides


class PolicySynthesizer:
    def synthesize(
        self,
        *,
        title: str,
        content: str,
        existing_policy: str,
        advisor_tags: Dict[str, object] | None = None,
    ) -> PolicySynthesisResult | None:
        raise NotImplementedError


class LLMPolicySynthesizer(PolicySynthesizer):
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
            raise RuntimeError("OpenAI client not installed; install the 'openai' extra to enable policy synthesis")
        if not api_key:
            raise RuntimeError("Policy synthesis requires OPENAI_API_KEY or CATALYST_POLICY_SYNTHESIS__api_key")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def enabled(self) -> bool:
        return bool(self._client)

    def synthesize(
        self,
        *,
        title: str,
        content: str,
        existing_policy: str,
        advisor_tags: Dict[str, object] | None = None,
    ) -> PolicySynthesisResult | None:
        if not self.enabled:
            return None
        preview = content[: self._sample_chars]
        tags_text = json.dumps(advisor_tags or {})
        prompt = (
            "You are configuring an ingestion pipeline for clinical documents. "
            "Given the sample text, propose chunking parameters in JSON with fields: chunk_modes (list), window_size, window_overlap, max_chunk_tokens, highlight_phrases (list), notes, confidence (0-1)."
            "Existing policy: "
            f"{existing_policy}\nAdvisor tags: {tags_text}\n"
            f"Title: {title}\nSample:\n{preview}\n"
            "Respond with JSON only."
        )
        response = self._client.responses.create(  # type: ignore[union-attr]
            model=self._model,
            input=[{"role": "user", "content": prompt}],
        )
        text = response.output[0].content[0].text  # type: ignore[attr-defined]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return PolicySynthesisResult(
            chunk_modes=[str(value).strip() for value in (data.get("chunk_modes") or []) if value],
            window_size=_coerce_int(data.get("window_size")),
            window_overlap=_coerce_int(data.get("window_overlap")),
            max_chunk_tokens=_coerce_int(data.get("max_chunk_tokens")),
            highlight_phrases=[str(value).strip() for value in (data.get("highlight_phrases") or []) if value],
            notes=data.get("notes"),
            confidence=_coerce_float(data.get("confidence")),
        )


def _coerce_int(value) -> int | None:
    try:
        number = int(value)
        return number if number > 0 else None
    except Exception:
        return None


def _coerce_float(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


__all__ = ["PolicySynthesisResult", "PolicySynthesizer", "LLMPolicySynthesizer"]

