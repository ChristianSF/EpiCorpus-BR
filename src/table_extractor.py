"""
Table extraction pipeline using pdfplumber.

Extracts tabular data from SIREVA-SUS PDFs and converts each data row into
a synthetic "sentence" suitable for NER annotation.  Only well-structured
tables (no rotated-text garbling) are included.

Output: data/extracted_tables/<report>.jsonl
Each record has the same schema as data/extracted/:
    relatorio, pagina, sentenca_id, texto, tipo="tabela"
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber

from src.config import DATA_RAW, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_TABLES: Path = PROJECT_ROOT / "data" / "extracted_tables"

# Minimum fraction of non-empty cells in a row to be considered a data row.
_MIN_FILL = 0.15
# Maximum fraction of single-character cells allowed in a table (garbling detector).
_MAX_SINGLE_CHAR = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_cell(cell: str | None) -> str:
    """Strip whitespace and normalise newlines inside a cell."""
    if not cell:
        return ""
    return re.sub(r"\s+", " ", cell).strip()


def _is_garbled(table: list[list[str | None]]) -> bool:
    """Return True if the table has too many single-character cells (rotated text)."""
    all_cells = [_clean_cell(c) for row in table for c in row if _clean_cell(c)]
    if not all_cells:
        return True
    single = sum(1 for c in all_cells if len(c) == 1)
    return single / len(all_cells) > _MAX_SINGLE_CHAR


def _nonempty_col_indices(table: list[list[str | None]]) -> list[int]:
    """Return indices of columns that have at least one non-empty cell."""
    if not table:
        return []
    ncols = max(len(row) for row in table)
    return [
        i for i in range(ncols)
        if any(_clean_cell(row[i]) if i < len(row) else "" for row in table)
    ]


def _split_header_data(
    rows: list[list[str]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Split table rows into header rows and data rows.

    Header rows are those where the majority of non-empty cells are
    non-numeric (text labels).  The first row with predominantly numeric
    content marks the start of data rows.
    """
    header: list[list[str]] = []
    data: list[list[str]] = []

    for row in rows:
        nonempty = [c for c in row if c]
        if not nonempty:
            continue
        numeric = sum(1 for c in nonempty if re.match(r"^[\d,.\-+%]+$", c))
        if numeric / len(nonempty) > 0.5 or data:
            data.append(row)
        else:
            header.append(row)

    return header, data


def _collapse_headers(header_rows: list[list[str]]) -> list[str]:
    """Merge multi-row headers into a single list of column labels."""
    if not header_rows:
        return []
    ncols = max(len(r) for r in header_rows)
    labels: list[str] = []
    for ci in range(ncols):
        parts = []
        for row in header_rows:
            val = row[ci] if ci < len(row) else ""
            if val and val not in parts:
                parts.append(val)
        labels.append(" ".join(parts))
    return labels


def _row_to_sentence(
    header_labels: list[str],
    data_row: list[str],
    table_caption: str,
) -> str:
    """Convert a data row into a readable synthetic sentence.

    Format: "[caption] | col1: val1 | col2: val2 | ..."
    Empty or duplicate values are skipped.
    """
    parts: list[str] = []
    if table_caption:
        parts.append(table_caption)

    for label, value in zip(header_labels, data_row):
        if not value:
            continue
        if label:
            parts.append(f"{label}: {value}")
        else:
            parts.append(value)

    return " | ".join(parts)


def _extract_tables_from_page(
    page: Any,
    report_name: str,
    sent_counter: int,
    page_no: int,
    caption_map: dict[int, str],
) -> tuple[list[dict[str, Any]], int]:
    """Extract synthetic sentences from all valid tables on a page."""
    records: list[dict[str, Any]] = []
    raw_tables = page.extract_tables()

    for table in raw_tables:
        if _is_garbled(table):
            logger.debug("Skipping garbled table on page %d", page_no)
            continue

        col_indices = _nonempty_col_indices(table)
        if not col_indices:
            continue

        # Filter to non-empty columns only
        filtered: list[list[str]] = [
            [_clean_cell(row[i]) if i < len(row) else "" for i in col_indices]
            for row in table
        ]

        header_rows, data_rows = _split_header_data(filtered)
        if not data_rows:
            continue

        header_labels = _collapse_headers(header_rows)
        caption = caption_map.get(page_no, "")

        for row in data_rows:
            nonempty = [c for c in row if c]
            if len(nonempty) < 2:
                continue
            # Skip subtotal/total aggregate rows
            first = row[0] if row else ""
            if re.match(r"^(subtotal|total)\b", first, re.IGNORECASE):
                continue

            text = _row_to_sentence(header_labels, row, caption)
            if len(text) < 10:
                continue

            sent_counter += 1
            records.append(
                {
                    "relatorio": report_name,
                    "pagina": page_no,
                    "sentenca_id": f"{report_name}_t{sent_counter:04d}",
                    "texto": text,
                    "tipo": "tabela",
                }
            )

    return records, sent_counter


def _build_caption_map(pdf: Any) -> dict[int, str]:
    """Build a page -> caption string map from text preceding tables."""
    caption_map: dict[int, str] = {}
    caption_re = re.compile(
        r"(tabela\s*\d+[\.\:]?\s*[^\n]{5,80})", re.IGNORECASE
    )
    for page in pdf.pages:
        text = page.extract_text() or ""
        m = caption_re.search(text)
        if m:
            caption = re.sub(r"\s+", " ", m.group(1)).strip()
            caption_map[page.page_number] = caption
    return caption_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_tables(
    pdf_path: Path,
    report_name: str | None = None,
) -> list[dict[str, Any]]:
    """Extract table rows as synthetic sentences from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        report_name: Override for the report slug (defaults to pdf_path.stem).

    Returns:
        List of sentence records with tipo="tabela".
    """
    pdf_path = Path(pdf_path)
    report_name = report_name or pdf_path.stem
    logger.info("Extracting tables from %s ...", pdf_path.name)

    records: list[dict[str, Any]] = []
    sent_counter = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        caption_map = _build_caption_map(pdf)
        for page in pdf.pages:
            page_records, sent_counter = _extract_tables_from_page(
                page, report_name, sent_counter, page.page_number, caption_map
            )
            records.extend(page_records)

    logger.info("  %d table rows extracted from %s", len(records), report_name)
    return records


def extract_tables_and_save(
    pdf_path: Path,
    output_dir: Path = DATA_TABLES,
    report_name: str | None = None,
) -> Path:
    """Extract table rows from a PDF and save as JSONL.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory for JSONL output.
        report_name: Override for report slug.

    Returns:
        Path to the written JSONL file.
    """
    records = extract_tables(pdf_path, report_name=report_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = report_name or Path(pdf_path).stem
    out_path = output_dir / f"{stem}.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Saved %d rows to %s", len(records), out_path)
    return out_path


def extract_all_tables(
    raw_dir: Path = DATA_RAW,
    output_dir: Path = DATA_TABLES,
    glob: str = "sireva_*.pdf",
) -> list[Path]:
    """Run table extraction over all PDFs matching a glob pattern.

    Args:
        raw_dir: Directory containing source PDFs.
        output_dir: Directory for JSONL outputs.
        glob: Glob pattern to select PDFs.

    Returns:
        List of paths to written JSONL files.
    """
    pdf_paths = sorted(raw_dir.glob(glob))
    if not pdf_paths:
        logger.warning("No PDFs matched '%s' in %s", glob, raw_dir)
        return []

    logger.info("Found %d PDFs in %s", len(pdf_paths), raw_dir)
    outputs: list[Path] = []

    for pdf_path in pdf_paths:
        try:
            out = extract_tables_and_save(pdf_path, output_dir)
            outputs.append(out)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_path.name, exc)

    logger.info(
        "Table extraction complete: %d/%d files processed.",
        len(outputs),
        len(pdf_paths),
    )
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    extract_all_tables()
