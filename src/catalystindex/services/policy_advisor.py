from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    parser_hint: Optional[str] = None
    chunk_overrides: Dict[str, object] = field(default_factory=dict)


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
            "You are assisting a retrieval system by selecting the best ingestion policy template for documents.\n\n"
            "Available policy templates:\n"
            "- ccbhc: Certified Community Behavioral Health Clinic criteria, compliance, staffing/service standards\n"
            "- dsm5: Diagnostic and Statistical Manual diagnostic criteria, mental health disorders\n"
            "- treatment_planner: Treatment plans, goals, objectives, interventions, outcomes\n"
            "- clinical_note: Clinical documentation, progress notes, patient records\n"
            "- legal_document: Legal texts, statutes, regulations, case law, contracts\n"
            "- research_paper: Academic papers, research studies, scientific publications\n"
            "- policy_manual: Policy documents, procedures, compliance manuals, standards\n"
            "- technical_spec: Technical specifications, API docs, protocol definitions\n"
            "- default: General purpose for documents that don't fit other categories\n\n"
            f"Document Title: {title}\n"
            f"Schema Hint (if provided): {schema or 'none'}\n\n"
            "Document Preview:\n"
            f"{content[: self._sample_chars]}\n\n"
            "Analyze the document structure, content type, and intended use. Select the best matching template "
            "or suggest 'default' if no template fits well.\n\n"
            "Respond strictly in JSON with fields:\n"
            "- policy_name: string (template name from above)\n"
            "- parser_hint: string or null (pdf, docx, html, plain_text)\n"
            "- confidence: float 0-1 (how well template matches)\n"
            "- tags: object (document characteristics: {\"has_tables\": true, \"has_criteria\": false, etc})\n"
            "- notes: string (brief explanation of selection)\n"
            "- chunk_overrides: object (optional adjustments: {\"window_size\": 450, \"max_chunk_tokens\": 500})\n"
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
        overrides = data.get("chunk_overrides")
        overrides_dict = overrides if isinstance(overrides, dict) else {}
        return PolicyAdvice(
            policy_name=data.get("policy_name"),
            confidence=data.get("confidence"),
            tags={k: str(v) for k, v in (data.get("tags", {}) or {}).items()},
            notes=data.get("notes"),
            parser_hint=data.get("parser_hint"),
            chunk_overrides=overrides_dict,
        )


__all__ = ["PolicyAdvisor", "PolicyAdvice"]
