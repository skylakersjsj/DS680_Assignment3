"""ADA extraction + scoring logic."""
from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils import PATHS, ensure_dirs, save_json

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are performing Automated Document Analysis (ADA) for AI ethics.

    Indicator: {indicator_id}
    Document title: {title}
    Model: {model}

    Task:
    1. Extract ALL claims relevant to this indicator.
    2. For each claim return:
       - summary (your own words)
       - evidence_span (verbatim quote from the document)
       - page_or_section
       - confidence (0-1)
       - subtags (optional list)

    Indicator-specific guidance:
    {indicator_instruction}

    Return ONLY a JSON object in this structure:
    {{
      "indicator_id": "...",
      "model_name": "...",
      "doc_title": "...",
      "claims": [
        {{
          "summary": "...",
          "evidence_span": "...",
          "page_or_section": "...",
          "confidence": 0.92,
          "subtags": []
        }}
      ]
    }}

    Document excerpt:
    {document_text}
    """
)

DEEPSEEK_PROMPT = textwrap.dedent(
    """
    You are scoring ADA claims for indicator {indicator_id} ({indicator_name}).
    Provided claims JSON:
    {claims_json}

    Score rubric:
    2 = Strong evidence with clear, explicit support
    1 = Partial evidence, vague or incomplete
    0 = No evidence or only generic statements

    Respond with a JSON object:
    {{
      "indicator_id": "...",
      "model_name": "...",
      "score": 0,
      "rationale": "..."
    }}
    """
)


INDICATOR_NAMES = {
    "L4_SafetyTradeoffs": "Safety objectives & trade-offs articulated",
    "L4_RedTeamingUpdates": "Red-teaming informs alignment updates",
    "L4_DomainFineTunes": "Domain-specific fine-tunes with safety",
    "L4_RollbackMechanisms": "Model update & rollback mechanisms",
    "L4_PedagogyEvidence": "Pedagogy evidence documented",
}

INDICATOR_PROMPT_EXTRAS = {
    "L4_SafetyTradeoffs": "Focus on passages discussing explicit trade-offs, value priorities, or balancing safety vs capability.",
    "L4_RedTeamingUpdates": "Capture claims about red-teaming, adversarial testing, and how findings led to mitigations or updates.",
    "L4_DomainFineTunes": "Find statements about domain- or industry-specific fine-tunes, including safety constraints during tuning.",
    "L4_RollbackMechanisms": "Locate details on version rollback, incident response, escalation, or fail-safe procedures.",
    "L4_PedagogyEvidence": "Extract evidence describing pedagogy, instructional design, or educational evaluation for the model.",
}


@dataclass
class LLMClient:
    force_offline: bool = False
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    extractor_provider: str = os.getenv("ADA_EXTRACT_PROVIDER", "anthropic")

    def __post_init__(self) -> None:
        requested_offline = self.force_offline or os.getenv("ADA_OFFLINE") == "1"

        provider = self.extractor_provider.lower().strip()
        if provider not in {"anthropic", "openai"}:
            provider = "anthropic"
        self.extractor_provider = provider

        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self._openai_key = os.getenv("OPENAI_API_KEY")
        self._deepseek_key = os.getenv("DEEPSEEK_API_KEY")

        if provider == "openai":
            has_extraction_key = bool(self._openai_key)
        else:
            has_extraction_key = bool(self._anthropic_key)

        self.extraction_offline = requested_offline or not has_extraction_key
        self.scoring_offline = requested_offline or not self._deepseek_key

        if self.extraction_offline:
            logger.warning(
                "ADA extraction running in offline stub mode (provider=%s).", self.extractor_provider
            )
        if self.scoring_offline:
            logger.warning("DeepSeek scoring running in offline stub mode.")

    # ---------- HTTP helpers ----------
    def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                resp_data = response.read().decode("utf-8")
                return json.loads(resp_data)
        except HTTPError as exc:  # pragma: no cover - network guard
            logger.error("HTTP error %s: %s", exc.code, exc.read())
            raise
        except URLError as exc:  # pragma: no cover
            logger.error("URL error: %s", exc)
            raise

    # ---------- Claude ----------
    def call_claude(self, prompt: str) -> str:
        if self.extraction_offline:
            return prompt  # used downstream for offline claim generation

        api_key = self._anthropic_key
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        payload = {
            "model": self.anthropic_model,
            "max_tokens": 1200,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        response = self._post_json("https://api.anthropic.com/v1/messages", payload, headers)
        content = response.get("content", [])
        if not content:
            raise RuntimeError("Claude response missing content")
        return content[0].get("text", "")

    # ---------- DeepSeek ----------
    def _call_openai(self, prompt: str) -> str:
        if self.extraction_offline:
            return prompt

        api_key = self._openai_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": "You are an ADA extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        response = self._post_json("https://api.openai.com/v1/chat/completions", payload, headers)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI returned no choices")
        return choices[0]["message"].get("content", "")

    def call_deepseek(self, prompt: str) -> str:
        if self.scoring_offline:
            return prompt

        api_key = self._deepseek_key
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")

        payload = {
            "model": self.deepseek_model,
            "messages": [
                {"role": "system", "content": "You score ADA evidence."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        response = self._post_json("https://api.deepseek.com/chat/completions", payload, headers)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("DeepSeek returned no choices")
        return choices[0]["message"].get("content", "")

    def call_extraction_model(self, prompt: str) -> str:
        if self.extractor_provider == "openai":
            return self._call_openai(prompt)
        return self.call_claude(prompt)


# ---------- ADA Extraction ----------

def _offline_claims(indicator_id: str, document_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Generate deterministic placeholder claims for offline mode."""

    lines = [line.strip() for line in document_text.splitlines() if line.strip()]
    trimmed = lines[:3] or [document_text[:200]]
    claims = []
    for idx, line in enumerate(trimmed, start=1):
        claims.append(
            {
                "summary": f"Heuristic claim {idx} for {indicator_id}",
                "evidence_span": line[:280],
                "page_or_section": f"auto-{idx}",
                "confidence": 0.4 + (0.2 * idx),
                "subtags": [],
            }
        )
    return {
        "indicator_id": indicator_id,
        "model_name": meta.get("model", "unknown"),
        "doc_title": meta.get("title", "unknown"),
        "claims": claims,
    }


def ada_extract_for_doc(
    indicator_id: str,
    combined_text: str,
    meta: Dict[str, Any],
    client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    client = client or LLMClient()
    prompt = PROMPT_TEMPLATE.format(
        indicator_id=indicator_id,
        title=meta.get("title", ""),
        model=meta.get("model", ""),
        indicator_instruction=INDICATOR_PROMPT_EXTRAS.get(
            indicator_id, "Prioritize concrete mechanisms, citations, and explicit evidence."
        ),
        document_text=combined_text,
    )

    if client.extraction_offline:
        return _offline_claims(indicator_id, combined_text, meta)

    response_text = client.call_extraction_model(prompt)
    try:
        parsed = json.loads(_clean_llm_json(response_text))
    except json.JSONDecodeError as exc:
        logger.error("Extraction model response not valid JSON: %s", response_text)
        raise RuntimeError("Invalid JSON returned from extraction model") from exc
    return parsed


# ---------- Scoring ----------

def _offline_score(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    claims_count = len(extraction_result.get("claims", []))
    if claims_count >= 3:
        score = 2
    elif claims_count:
        score = 1
    else:
        score = 0
    return {
        "indicator_id": extraction_result.get("indicator_id"),
        "model_name": extraction_result.get("model_name"),
        "score": score,
        "rationale": "Offline heuristic scoring based on claim count.",
    }


def score_with_deepseek(
    extraction_result: Dict[str, Any],
    indicator_id: str,
    client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    client = client or LLMClient()
    if client.scoring_offline:
        return _offline_score(extraction_result)

    prompt = DEEPSEEK_PROMPT.format(
        indicator_id=indicator_id,
        indicator_name=INDICATOR_NAMES.get(indicator_id, indicator_id),
        claims_json=json.dumps(extraction_result, indent=2),
    )
    response_text = client.call_deepseek(prompt)
    try:
        parsed = json.loads(_clean_llm_json(response_text))
    except json.JSONDecodeError as exc:
        logger.error("DeepSeek returned invalid JSON: %s", response_text)
        raise RuntimeError("Invalid JSON from DeepSeek") from exc
    return parsed


# ---------- Persistence Helpers ----------

def save_extraction(result: Dict[str, Any]) -> Path:
    model = result.get("model_name", "unknown")
    indicator = result.get("indicator_id", "unknown")
    filename = f"{model}_{indicator}.json".replace("/", "-")
    path = PATHS.extracted_dir / filename
    save_json(result, path)
    return path


def save_score(result: Dict[str, Any]) -> Path:
    model = result.get("model_name", "unknown")
    indicator = result.get("indicator_id", "unknown")
    filename = f"{model}_{indicator}_score.json".replace("/", "-")
    path = PATHS.scored_dir / filename
    save_json(result, path)
    return path


ensure_dirs([PATHS.extracted_dir, PATHS.scored_dir])
def _clean_llm_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped, count=1)
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")].strip()
    return stripped
