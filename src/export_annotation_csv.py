"""
Export annotation CSVs for human review.

Generates two CSV files ready to send to an annotator:
  annotation/review/annotation_texto.csv   -- 180 narrative sentences
  annotation/review/annotation_tabelas.csv -- 100 table rows

Column schema (both files):
  id                  -- sentence/row identifier
  relatorio           -- source document slug
  pagina              -- PDF page number
  texto               -- full sentence / table row text
  entidades_sugeridas -- silver spans in human-readable form, e.g.
                         "pneumococo → PATOGENO; 2024 → PERIODO_TEMPORAL"
                         "(nenhuma)" when there are no silver entities
  entidades_corrigidas -- [ANNOTATOR FILLS THIS]
                          Accept as-is → write "ok"
                          Correct/add  → write same format as entidades_sugeridas
                          Reject all   → write "(nenhuma)"
  status              -- [ANNOTATOR FILLS THIS]: ok | corrigido | erro
  notas               -- [ANNOTATOR FILLS THIS]: free-text remarks

Usage:
    python -m src.export_annotation_csv           # both files
    python -m src.export_annotation_csv --texto    # text only
    python -m src.export_annotation_csv --tabelas  # tables only
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.config import ANNOTATION_SAMPLES, PROJECT_ROOT

logger = logging.getLogger(__name__)

REVIEW_DIR: Path = PROJECT_ROOT / "annotation" / "review"
TEXT_REVIEW_CSV: Path = REVIEW_DIR / "gold_review.csv"
TABLE_CANDIDATES_JSONL: Path = ANNOTATION_SAMPLES / "gold_table_candidates.jsonl"
OUT_TEXT_CSV: Path = REVIEW_DIR / "annotation_texto.csv"
OUT_TABLE_CSV: Path = REVIEW_DIR / "annotation_tabelas.csv"

_FIELDNAMES = ["id", "relatorio", "pagina", "texto",
               "entidades_sugeridas", "entidades_corrigidas", "status", "notas"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spans_to_readable(spans_raw: str | list[dict[str, str]]) -> str:
    """Convert a JSON spans list to a human-readable string.

    Args:
        spans_raw: Either a JSON string or a list of span dicts.

    Returns:
        Human-readable string like "pneumococo → PATOGENO; 2024 → PERIODO_TEMPORAL",
        or "(nenhuma)" if there are no spans.
    """
    if isinstance(spans_raw, str):
        raw = spans_raw.strip()
        if not raw or raw == "[]":
            return "(nenhuma)"
        try:
            spans: list[dict[str, str]] = json.loads(raw)
        except json.JSONDecodeError:
            return spans_raw  # leave as-is if unparseable
    else:
        spans = spans_raw

    if not spans:
        return "(nenhuma)"

    parts = [f"{s.get('texto', '')} → {s.get('tipo', '')}" for s in spans]
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Text CSV (from gold_review.csv)
# ---------------------------------------------------------------------------


def export_text_csv(
    source_csv: Path = TEXT_REVIEW_CSV,
    out_csv: Path = OUT_TEXT_CSV,
) -> int:
    """Export annotation CSV for narrative text sentences.

    Reads the existing gold_review.csv (which contains silver spans as JSON)
    and produces a human-readable CSV for the annotator.

    Args:
        source_csv: Source review CSV with JSON spans.
        out_csv: Destination annotation CSV.

    Returns:
        Number of rows written.
    """
    if not source_csv.exists():
        logger.error("Source CSV not found: %s", source_csv)
        return 0

    rows: list[dict[str, Any]] = []
    with source_csv.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row.get("sentenca_id", ""),
                "relatorio": row.get("relatorio", ""),
                "pagina": row.get("pagina", ""),
                "texto": row.get("texto", ""),
                "entidades_sugeridas": _spans_to_readable(row.get("spans_silver", "[]")),
                "entidades_corrigidas": "",
                "status": "",
                "notas": row.get("notas", ""),
            })

    logger.info("annotation_texto.csv: %d linhas → %s", len(rows), out_csv)
    return len(rows)


# ---------------------------------------------------------------------------
# Tables CSV (from gold_table_candidates.jsonl)
# ---------------------------------------------------------------------------


def export_table_csv(
    source_jsonl: Path = TABLE_CANDIDATES_JSONL,
    out_csv: Path = OUT_TABLE_CSV,
) -> int:
    """Export annotation CSV for table rows.

    Reads the sampled table candidates (silver-annotated JSONL) and produces
    a human-readable CSV for the annotator.

    Args:
        source_jsonl: JSONL file with silver-annotated table rows.
        out_csv: Destination annotation CSV.

    Returns:
        Number of rows written.
    """
    if not source_jsonl.exists():
        logger.error("Source JSONL not found: %s", source_jsonl)
        return 0

    records: list[dict[str, Any]] = []
    with source_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "id": rec.get("sentenca_id", ""),
                "relatorio": rec.get("relatorio", ""),
                "pagina": rec.get("pagina", ""),
                "texto": rec.get("texto", ""),
                "entidades_sugeridas": _spans_to_readable(rec.get("spans", [])),
                "entidades_corrigidas": "",
                "status": "",
                "notas": "",
            })

    logger.info("annotation_tabelas.csv: %d linhas → %s", len(records), out_csv)
    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Export annotation CSVs for human review.")
    parser.add_argument("--texto", action="store_true", help="Export text CSV only.")
    parser.add_argument("--tabelas", action="store_true", help="Export tables CSV only.")
    args = parser.parse_args()

    do_text = args.texto or (not args.texto and not args.tabelas)
    do_tables = args.tabelas or (not args.texto and not args.tabelas)

    if do_text:
        n = export_text_csv()
        print(f"Texto:   {n} sentencas -> {OUT_TEXT_CSV}")

    if do_tables:
        n = export_table_csv()
        print(f"Tabelas: {n} linhas    -> {OUT_TABLE_CSV}")
