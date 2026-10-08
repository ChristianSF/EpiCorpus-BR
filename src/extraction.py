"""
PDF extraction pipeline -- scenario B2 from the PROPOR 2026 paper.

Converts each SIREVA-SUS PDF to a list of narrative sentences, discarding
table content and keeping only prose text.  Output is saved as JSONL in
data/extracted/, one file per report.

Scenario B2 definition (from the PROPOR paper):
    Docling text-only export with explicit markdown-table removal and
    caption preservation.  This yields the cleanest narrative sentences
    for downstream NER annotation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.config import DATA_EXTRACTED, DATA_RAW

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Regex that detects a Markdown table row: starts with | and has >= 2 pipes.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|")
# Regex that detects a Markdown table separator row: only |, -, :, spaces.
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|")
# Caption patterns that precede tables and should be retained.
_CAPTION_RE = re.compile(
    r"^\s*(tabela\s*\d+[.:]|table\s*\d+[.:]|figura\s*\d+[.:]|figure\s*\d+[.:])",
    re.IGNORECASE,
)
# Sentence boundary: period / ! / ? followed by whitespace and an uppercase letter
# or digit that starts the next sentence.  Avoids splitting on common
# abbreviations by requiring the following token to start with [A-Z0-9].
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s{1,3}(?=[A-Z\d])")
# Table-of-contents noise: three or more consecutive dots (filling dots).
_TOC_LINE_RE = re.compile(r"\.{3,}")
# Lines that are mostly non-alphabetic (e.g. page numbers, codes, dashes).
_MOSTLY_NONALPHA_RE = re.compile(r"^[^a-zA-ZÀ-ÿ]{0,3}$")


def _is_table_line(line: str) -> bool:
    """Return True if the line belongs to a Markdown table body or header."""
    return bool(_TABLE_ROW_RE.match(line)) and line.count("|") >= 2


def _is_sep_line(line: str) -> bool:
    """Return True if the line is a Markdown table separator (--|--|--)."""
    stripped = line.strip()
    return (
        stripped.startswith("|")
        and bool(_TABLE_SEP_RE.match(line))
        and set(stripped.replace("|", "").replace(":", "").strip()) <= {"-", " "}
    )


def _remove_tables(text: str, keep_caption: bool = True) -> str:
    """Remove Markdown table blocks from Docling export text.

    Args:
        text: Raw text from ``DoclingDocument.export_to_text()``.
        keep_caption: When True, preserve the caption line that appears
            immediately before each table block (e.g. "Tabela 1:").

    Returns:
        Cleaned text with table blocks replaced by blank lines.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if _is_table_line(line):
            # Look back to optionally keep the caption.
            if keep_caption and out and _CAPTION_RE.match(out[-1]):
                pass  # caption stays in out
            elif out and out[-1].strip() == "":
                pass  # blank separator stays
            # Skip all consecutive table lines.
            while i < len(lines) and (
                _is_table_line(lines[i]) or _is_sep_line(lines[i])
            ):
                i += 1
            # Add blank line after the removed block.
            if out and out[-1].strip() != "":
                out.append("")
        else:
            out.append(line)
            i += 1

    return "\n".join(out)


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into individual sentences.

    Uses a conservative regex boundary that requires the next token to
    start with an uppercase letter or digit.  Short fragments (< 15 chars)
    are merged back into the previous sentence to avoid noisy annotations.

    Args:
        text: A prose paragraph extracted from the report.

    Returns:
        List of sentence strings.
    """
    raw = _SENTENCE_BOUNDARY_RE.split(text.strip())
    sentences: list[str] = []

    for fragment in raw:
        fragment = fragment.strip()
        if not fragment:
            continue
        # Merge very short fragments (likely split mid-abbreviation) into
        # the previous sentence.
        if sentences and len(fragment) < 15:
            sentences[-1] = sentences[-1].rstrip() + " " + fragment
        else:
            sentences.append(fragment)

    return sentences


def _is_narrative(text: str) -> bool:
    """Return True if the text looks like genuine narrative prose.

    Rejects table-of-contents lines (filling dots), very short fragments,
    and lines that contain no alphabetic content.

    Args:
        text: Candidate sentence string.

    Returns:
        True for narrative text, False for structural noise.
    """
    stripped = text.strip()
    if len(stripped) < 15:
        return False
    # TOC lines with filling dots: "Introducao  ........ 4"
    if _TOC_LINE_RE.search(stripped):
        return False
    # Mostly non-alphabetic (page numbers, dividers, etc.)
    if _MOSTLY_NONALPHA_RE.match(stripped):
        return False
    # Must contain at least one word of 3+ letters.
    if not re.search(r"[a-zA-ZÀ-ÿ]{3,}", stripped):
        return False
    return True


def _iter_paragraphs(clean_text: str) -> list[tuple[int, str]]:
    """Split cleaned full-text into (approximate_page, paragraph) pairs.

    Because Docling's ``export_to_text()`` does not embed page markers in
    the flat string, page numbers are approximated by tracking occurrences
    of the form ``\\n\\n`` (paragraph breaks).  A more accurate page
    attribution requires iterating the document items directly -- see
    ``extract_narrative_text`` which uses the structured approach when
    provenance data is available.

    Args:
        clean_text: Text after table removal.

    Returns:
        List of (page_approx, paragraph_text) tuples.  page_approx is
        always 0 in the flat-text path; the structured path fills it
        correctly.
    """
    paragraphs = re.split(r"\n{2,}", clean_text)
    return [(0, p.strip()) for p in paragraphs if p.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_narrative_text(
    pdf_path: Path, report_name: str | None = None
) -> list[dict[str, Any]]:
    """Extract narrative sentences from a single SIREVA-SUS PDF.

    Applies scenario B2: Docling text-only export followed by explicit
    Markdown-table removal.  Iterates the structured document items to
    obtain accurate per-item page provenance when available, then falls
    back to the flat-text approximation.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        report_name: Override for the ``relatorio`` / ``sentenca_id`` slug.
            Defaults to ``pdf_path.stem``.

    Returns:
        List of sentence records.  Each record is a dict with keys:

        - ``relatorio`` (str): Report stem, e.g. ``"sireva_2024"``.
        - ``pagina`` (int): Page number (1-based), or 0 if unavailable.
        - ``sentenca_id`` (str): Unique identifier, e.g. ``"sireva_2024_s0012"``.
        - ``texto`` (str): Sentence text.
    """
    pdf_path = Path(pdf_path)
    report_name = report_name or pdf_path.stem
    logger.info("Converting %s ...", pdf_path.name)

    # Configure a lightweight pipeline: no OCR (PDFs are text-based),
    # no ML table structure analysis.  This avoids downloading large
    # model weights and is sufficient for the SIREVA-SUS reports.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path))
    doc = result.document

    sentences: list[dict[str, Any]] = []
    sent_counter = 0

    # ------------------------------------------------------------------
    # Structured path: iterate document items to get page provenance.
    # TextItem objects carry .text and .prov[].page_no.
    # Table items are skipped entirely.
    # ------------------------------------------------------------------
    try:
        from docling_core.types.doc import DocItemLabel, TextItem  # type: ignore[import]

        _TABLE_LABELS = {
            DocItemLabel.TABLE,
        }

        for item, _level in doc.iterate_items():
            # Skip non-text items (tables, figures, etc.).
            if not isinstance(item, TextItem):
                continue
            if hasattr(item, "label") and item.label in _TABLE_LABELS:
                continue

            raw_text: str = item.text or ""
            raw_text = raw_text.strip()
            if not raw_text:
                continue

            # Get page number from provenance (first provenance entry).
            page: int = 0
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0]
                page = getattr(prov, "page_no", 0)

            for sent_text in _split_sentences(raw_text):
                if not _is_narrative(sent_text):
                    continue
                sent_counter += 1
                sentences.append(
                    {
                        "relatorio": report_name,
                        "pagina": page,
                        "sentenca_id": f"{report_name}_s{sent_counter:04d}",
                        "texto": sent_text,
                    }
                )

        if sentences:
            logger.info(
                "  Structured path: %d sentences from %s", len(sentences), report_name
            )
            return sentences

        # If structured path yielded nothing, fall through to flat-text.
        logger.warning(
            "  Structured path returned 0 sentences for %s -- falling back to flat text.",
            report_name,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  Structured Docling iteration failed (%s) -- using flat-text fallback.",
            exc,
        )

    # ------------------------------------------------------------------
    # Flat-text fallback: export_to_text() + table removal.
    # Page numbers are unavailable in this path (set to 0).
    # ------------------------------------------------------------------
    full_text: str = doc.export_to_text()
    clean_text: str = _remove_tables(full_text, keep_caption=True)

    for _page, paragraph in _iter_paragraphs(clean_text):
        for sent_text in _split_sentences(paragraph):
            if not _is_narrative(sent_text):
                continue
            sent_counter += 1
            sentences.append(
                {
                    "relatorio": report_name,
                    "pagina": 0,
                    "sentenca_id": f"{report_name}_s{sent_counter:04d}",
                    "texto": sent_text,
                }
            )

    logger.info(
        "  Flat-text path: %d sentences from %s", len(sentences), report_name
    )
    return sentences


def extract_and_save(
    pdf_path: Path,
    output_dir: Path = DATA_EXTRACTED,
    report_name: str | None = None,
) -> Path:
    """Extract sentences from a PDF and save as JSONL.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory where the JSONL file will be written.
            Defaults to ``DATA_EXTRACTED`` from config.
        report_name: Override for the report identifier used in ``relatorio``
            and ``sentenca_id`` fields.  Defaults to ``pdf_path.stem``.

    Returns:
        Path to the written JSONL file.
    """
    sentences = extract_narrative_text(pdf_path, report_name=report_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = report_name if report_name else pdf_path.stem
    out_path = output_dir / f"{stem}.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for record in sentences:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d sentences to %s", len(sentences), out_path)
    return out_path


def extract_all(
    raw_dir: Path = DATA_RAW,
    output_dir: Path = DATA_EXTRACTED,
    glob: str = "sireva_*.pdf",
) -> list[Path]:
    """Run the extraction pipeline over all PDFs matching a glob pattern.

    Args:
        raw_dir: Directory containing the source PDFs.
        output_dir: Directory for the JSONL outputs.
        glob: Glob pattern to select PDFs inside ``raw_dir``.

    Returns:
        List of paths to the written JSONL files, in sorted order.
    """
    pdf_paths = sorted(raw_dir.glob(glob))
    if not pdf_paths:
        logger.warning("No PDFs matched '%s' in %s", glob, raw_dir)
        return []

    logger.info("Found %d PDFs in %s", len(pdf_paths), raw_dir)
    outputs: list[Path] = []

    for pdf_path in pdf_paths:
        try:
            out = extract_and_save(pdf_path, output_dir)
            outputs.append(out)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process %s: %s", pdf_path.name, exc)

    logger.info("Extraction complete: %d/%d files processed.", len(outputs), len(pdf_paths))
    return outputs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    extract_all()
