"""Utility helpers for the ADA assignment pipeline."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from html.parser import HTMLParser

logger = logging.getLogger(__name__)


@dataclass
class Paths:
    """Centralized project paths used across modules."""

    root: Path = Path(__file__).resolve().parent
    data_dir: Path = root / "data"
    docs_dir: Path = root / "docs"
    results_dir: Path = root / "results"
    extracted_dir: Path = results_dir / "extracted"
    scored_dir: Path = results_dir / "scored"
    report_dir: Path = root / "report"
    figures_dir: Path = root / "figures"


PATHS = Paths()


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", path)


def save_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Saved JSON to %s", destination)


def read_json(source: Path) -> dict:
    with source.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def _strip_html(html_text: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html_text)
    return stripper.get_text()


def _read_file_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()
    if path.suffix.lower() in {".html", ".htm"}:
        return _strip_html(content)
    return content


def load_pdf_text(path: str | os.PathLike[str]) -> List[str]:
    """Return a list of page-level texts from a PDF/HTML/TXT document."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".html", ".htm"}:
        logger.warning("Unsupported extension %s, attempting to read as text", suffix)

    if suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "PyPDF2 is required to read PDF files. Run `pip install PyPDF2`."
            ) from exc

        reader = PdfReader(str(file_path))
        pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(text.strip())
            logger.debug("Loaded PDF page %s from %s", idx + 1, file_path.name)
        return pages

    # Non PDF formats fall back to simple text splitting.
    content = _read_file_text(file_path)
    paragraphs = [block.strip() for block in content.split("\n\n") if block.strip()]
    return paragraphs or [content]


def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """Simple chunker that keeps chunks within LLM context limits."""

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + max_chars, len(text))
        chunk = text[cursor:end]
        chunks.append(chunk)
        cursor = end
    return chunks


ensure_dirs(
    [
        PATHS.data_dir,
        PATHS.docs_dir,
        PATHS.results_dir,
        PATHS.extracted_dir,
        PATHS.scored_dir,
        PATHS.report_dir,
        PATHS.figures_dir,
    ]
)
