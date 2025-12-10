"""Entry point for the ADA Assignment 3 pipeline."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Set

from ada_extract import (
    LLMClient,
    ada_extract_for_doc,
    save_extraction,
    save_score,
    score_with_deepseek,
)
from relevance_filter import extract_relevant_pages, refine_relevant_text
from utils import PATHS, ensure_dirs, read_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ada_pipeline")

ALLOWED_DOC_EXTS = {".pdf", ".txt", ".md", ".html", ".htm"}

INDICATOR_DIRS = {
    "L4_SafetyTradeoffs": PATHS.docs_dir / "L4_SafetyTradeoffs",
    "L4_RedTeamingUpdates": PATHS.docs_dir / "L4_RedTeamingUpdates",
    "L4_DomainFineTunes": PATHS.docs_dir / "L4_DomainFineTunes",
    "L4_RollbackMechanisms": PATHS.docs_dir / "L4_RollbackMechanisms",
    "L4_PedagogyEvidence": PATHS.docs_dir / "L4_PedagogyEvidence",
}


def discover_indicator_docs() -> List[Dict]:
    discovered: List[Dict] = []
    for indicator, folder in INDICATOR_DIRS.items():
        if not folder.exists():
            continue
        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in ALLOWED_DOC_EXTS:
                continue
            rel_path = file_path.relative_to(PATHS.root)
            model_name = file_path.stem
            discovered.append(
                {
                    "path": str(rel_path),
                    "model": model_name,
                    "title": file_path.stem.replace("_", " "),
                    "url": "",
                    "indicators": [indicator],
                }
            )
    return discovered


def load_docs_registry(path: str = "data/docs_meta.json") -> List[Dict]:
    meta_path = Path(path)
    docs: List[Dict] = []
    existing_paths: Set[str] = set()

    if meta_path.exists():
        data = read_json(meta_path)
        docs = data.get("docs", [])
        existing_paths = {doc.get("path", "") for doc in docs if doc.get("path")}
        if not docs:
            logger.warning("No documents configured in %s", meta_path)
    else:
        logger.warning("Registry file not found: %s; relying solely on indicator folders.", meta_path)

    auto_docs = discover_indicator_docs()
    for doc in auto_docs:
        if doc["path"] in existing_paths:
            continue
        docs.append(doc)

    if not docs:
        logger.warning(
            "No documents discovered. Place files under %s or populate %s.",
            PATHS.docs_dir,
            meta_path,
        )
    return docs


def aggregate_scores(rows: List[Dict[str, str]]) -> None:
    if not rows:
        logger.warning("No scores to aggregate")
        return

    csv_path = PATHS.report_dir / "aggregated_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "indicator", "score", "rationale"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote CSV aggregation to %s", csv_path)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("matplotlib not installed: %s", exc)
        return

    indicator_labels = sorted({row["indicator"] for row in rows})
    model_labels = sorted({row["model"] for row in rows})

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / max(1, len(model_labels))

    for idx, model in enumerate(model_labels):
        model_scores = []
        for indicator in indicator_labels:
            match = next(
                (row for row in rows if row["model"] == model and row["indicator"] == indicator),
                None,
            )
            model_scores.append(int(match["score"]) if match else 0)
        positions = [i + idx * bar_width for i in range(len(indicator_labels))]
        ax.bar(positions, model_scores, width=bar_width, label=model)

    ax.set_xticks(
        [i + bar_width * (len(model_labels) - 1) / 2 for i in range(len(indicator_labels))]
    )
    ax.set_xticklabels(indicator_labels, rotation=20, ha="right")
    ax.set_ylim(0, 2.2)
    ax.set_ylabel("Score (0-2)")
    ax.set_title("ADA Indicator Scores")
    ax.legend()
    fig.tight_layout()

    fig_path = PATHS.figures_dir / "scores_bar_chart.png"
    fig.savefig(fig_path, dpi=200)
    logger.info("Saved bar chart to %s", fig_path)


def write_analysis(rows: List[Dict[str, str]]) -> None:
    analysis_path = PATHS.report_dir / "analysis.md"
    if not rows:
        analysis_path.write_text("No scores generated.\n", encoding="utf-8")
        return

    lines = ["# ADA Indicator Analysis", ""]
    lines.append("| Model | Indicator | Score | Rationale |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        rationale = row["rationale"].replace("|", "/")
        lines.append(
            f"| {row['model']} | {row['indicator']} | {row['score']} | {rationale} |"
        )
    analysis_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Updated analysis summary at %s", analysis_path)


def run_ada_pipeline() -> None:
    docs = load_docs_registry()
    ensure_dirs([PATHS.extracted_dir, PATHS.scored_dir, PATHS.report_dir, PATHS.figures_dir])
    client = LLMClient()

    aggregated_rows: List[Dict[str, str]] = []

    for doc in docs:
        doc_path = PATHS.root / doc["path"]
        if not doc_path.exists():
            logger.warning("Document missing: %s", doc_path)
            continue

        indicators = doc.get("indicators", [])
        for indicator in indicators:
            logger.info("Processing %s for indicator %s", doc_path.name, indicator)
            relevance = extract_relevant_pages(str(doc_path), indicator_id=indicator)
            refined_text = refine_relevant_text(relevance.combined_text, indicator)

            extraction_result = ada_extract_for_doc(
                indicator_id=indicator,
                combined_text=refined_text,
                meta=doc,
                client=client,
            )
            extraction_result.setdefault("metadata", {})["selected_pages"] = relevance.selected_pages
            extraction_result["metadata"]["page_scores"] = relevance.page_scores
            extraction_path = save_extraction(extraction_result)
            logger.debug("Extraction saved to %s", extraction_path)

            score_result = score_with_deepseek(extraction_result, indicator_id=indicator, client=client)
            score_path = save_score(score_result)
            logger.debug("Score saved to %s", score_path)

            aggregated_rows.append(
                {
                    "model": score_result.get("model_name", doc.get("model", "")),
                    "indicator": score_result.get("indicator_id", indicator),
                    "score": str(score_result.get("score", 0)),
                    "rationale": score_result.get("rationale", ""),
                }
            )

    aggregate_scores(aggregated_rows)
    write_analysis(aggregated_rows)


if __name__ == "__main__":
    run_ada_pipeline()
