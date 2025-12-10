"""Two-stage relevance filtering for ADA pipeline."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from utils import load_pdf_text

logger = logging.getLogger(__name__)


KEYWORDS: Dict[str, List[str]] = {
    "L4_SafetyTradeoffs": [
        "safety",
        "risk",
        "trade-off",
        "tradeoff",
        "balance",
        "utility",
        "value",
        "alignment",
    ],
    "L4_RedTeamingUpdates": [
        "red team",
        "red-teaming",
        "adversarial",
        "jailbreak",
        "attack",
        "alignment update",
        "mitigation",
    ],
    "L4_DomainFineTunes": [
        "fine-tune",
        "finetune",
        "domain",
        "vertical",
        "specialized",
        "safety constraint",
    ],
    "L4_RollbackMechanisms": [
        "rollback",
        "roll back",
        "revert",
        "version",
        "incident",
        "suspension",
        "kill switch",
    ],
    "L4_PedagogyEvidence": [
        "pedagogy",
        "instructional",
        "curriculum",
        "learning",
        "education",
        "teaching",
    ],
}


@dataclass
class RelevanceResult:
    indicator_id: str
    combined_text: str
    selected_pages: List[int]
    page_scores: List[int]


def page_relevance_score(text: str, keywords: List[str]) -> int:
    if not text:
        return 0
    lowered = text.lower()
    score = 0
    for keyword in keywords:
        count = lowered.count(keyword.lower())
        score += count
    return score


def extract_relevant_pages(pdf_path: str, indicator_id: str, top_k: int = 5) -> RelevanceResult:
    pages = load_pdf_text(pdf_path)
    keywords = KEYWORDS.get(indicator_id, [])
    page_scores = [page_relevance_score(page, keywords) for page in pages]

    ranked = sorted(enumerate(page_scores, start=1), key=lambda item: item[1], reverse=True)
    top_ranked = ranked[:top_k] if top_k else ranked
    selected_indices = [idx for idx, _ in top_ranked]

    combined_text = "\n\n".join(pages[i - 1] for i in selected_indices if 0 < i <= len(pages))

    logger.info(
        "Selected %s/%s pages for %s", len(selected_indices), len(pages), indicator_id
    )

    return RelevanceResult(
        indicator_id=indicator_id,
        combined_text=combined_text,
        selected_pages=selected_indices,
        page_scores=page_scores,
    )


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def refine_relevant_text(raw_text: str, indicator_id: str) -> str:
    keywords = [kw.lower() for kw in KEYWORDS.get(indicator_id, [])]
    if not keywords:
        return raw_text

    sentences = _split_sentences(raw_text)
    filtered = [
        sentence
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in keywords)
    ]

    if not filtered:
        logger.debug("No refinement hits for %s; returning original text", indicator_id)
        return raw_text

    refined_text = " ".join(filtered)
    logger.debug("Refined %s sentences down to %s", len(sentences), len(filtered))
    return refined_text
