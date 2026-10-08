"""
Gold standard builder: converts the reviewed CSV back to JSONL.

Workflow:
  1. Open  annotation/review/gold_review.csv  in Excel / LibreOffice Calc.
  2. For each sentence:
       - If the silver annotation is correct, leave  status = "ok".
       - If you corrected the spans, set  status = "corrigido"  and
         edit the  spans_correto  column (JSON array).
       - If the sentence is unusable, set  status = "erro"  (it will be
         excluded from the gold set).
  3. Save the CSV (keep UTF-8 encoding).
  4. Run:
         python -m src.gold_builder

Output:
  data/gold/gold_sample.jsonl   -- gold JSONL, same schema as silver.
  data/gold/build_report.json   -- summary of accepted / corrected / skipped.

JSON format for the  spans_correto  column:
  [{"texto": "Streptococcus pneumoniae", "tipo": "PATOGENO"}, ...]
  Empty list []  means no entities in this sentence.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from src.config import DATA_GOLD, PROJECT_ROOT, VALID_ENTITY_NAMES

logger = logging.getLogger(__name__)

REVIEW_CSV: Path = PROJECT_ROOT / "annotation" / "review" / "gold_review.csv"
GOLD_JSONL: Path = DATA_GOLD / "gold_sample.jsonl"
BUILD_REPORT: Path = DATA_GOLD / "build_report.json"

# Valid status values that are accepted into the gold set.
_ACCEPTED_STATUSES = {"ok", "corrigido"}
_SKIP_STATUSES = {"erro", "skip", "ignorar"}


# ---------------------------------------------------------------------------
# Tokeniser (identical to silver_annotator.tokenize)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Split text into whitespace tokens with character offsets."""
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


# ---------------------------------------------------------------------------
# Span → BIO conversion
# ---------------------------------------------------------------------------


def _normalize_ws(s: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", s).strip()


def _find_span_in_text(
    span_text: str,
    text: str,
    text_norm: str,
) -> tuple[int, int] | None:
    """Locate span_text inside text, tolerating extra internal spaces.

    Tries three strategies in order:
      1. Exact (case-insensitive) match on the raw text.
      2. Exact match after collapsing whitespace in both strings.
      3. Regex match where each space in the span can match ``\\s+``.

    Args:
        span_text: The span surface form.
        text: Original sentence text.
        text_norm: Pre-normalised version of text (collapsed spaces).

    Returns:
        (start, end) character offsets into the *original* text, or None.
    """
    span_lower = span_text.lower()
    text_lower = text.lower()

    # Strategy 1: exact match on raw text.
    idx = text_lower.find(span_lower)
    if idx != -1:
        return idx, idx + len(span_text)

    # Strategy 2: match on normalised text, then map back to original offsets.
    span_norm = _normalize_ws(span_text).lower()
    text_norm_lower = text_norm.lower()
    idx_norm = text_norm_lower.find(span_norm)
    if idx_norm != -1:
        # Map normalised offset → original offset by advancing through text.
        orig_pos = 0
        norm_pos = 0
        orig_start = 0
        for ni in range(idx_norm):
            while orig_pos < len(text) and text_norm_lower[norm_pos] != text_lower[orig_pos]:
                orig_pos += 1
            orig_pos += 1
            norm_pos += 1
        orig_start = orig_pos
        orig_end = orig_start
        for ni in range(len(span_norm)):
            while orig_end < len(text) and text_norm_lower[idx_norm + ni] != text_lower[orig_end]:
                orig_end += 1
            orig_end += 1
        # Safety: fall through to strategy 3 if mapping went wrong.
        if orig_end <= len(text):
            return orig_start, orig_end

    # Strategy 3: regex with \s+ between words.
    words = re.split(r"\s+", span_text.strip())
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.start(), m.end()

    # Strategy 4: allow optional spaces around punctuation (., -, /) and
    # between words.  Handles PDF artifacts like "sp ." and "1999- 2018".
    # Split the span on punctuation chars to avoid re.escape mangling them.
    _PUNCT_RE = re.compile(r"([.\-/])")
    parts = _PUNCT_RE.split(span_text.strip())
    pat_pieces: list[str] = []
    for part in parts:
        if _PUNCT_RE.fullmatch(part):
            pat_pieces.append(r"\s*" + re.escape(part) + r"\s*")
        else:
            escaped = re.escape(part)
            escaped = re.sub(r"\\ ", r"\\s+", escaped)
            pat_pieces.append(escaped)
    punct_pattern = "".join(pat_pieces)
    m = re.search(punct_pattern, text, re.IGNORECASE)
    if m:
        return m.start(), m.end()

    return None


def _spans_to_bio(
    text: str,
    spans: list[dict[str, str]],
) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    """Convert span annotations to BIO tags aligned to whitespace tokens.

    Span matching is case-insensitive and tolerates extra internal spaces
    introduced by PDF extraction.  A span that cannot be found is logged
    as a warning and skipped.

    Args:
        text: Sentence text.
        spans: List of {"texto": str, "tipo": str} dicts.

    Returns:
        (tokens, offsets, bio_tags)
    """
    tokens, offsets = _tokenize(text)
    bio_tags = ["O"] * len(tokens)
    text_norm = _normalize_ws(text)

    # Sort spans longest-first so that overlapping spans are handled
    # deterministically (longer wins).
    sorted_spans = sorted(spans, key=lambda s: -len(s.get("texto", "")))

    # Build a char-level label array: index → entity_type | None
    char_label: list[str | None] = [None] * len(text)

    for span in sorted_spans:
        span_text = span.get("texto", "").strip()
        entity_type = span.get("tipo", "").strip().upper()

        if not span_text:
            continue
        if entity_type not in VALID_ENTITY_NAMES:
            logger.warning("Unknown entity type '%s' -- skipping span.", entity_type)
            continue

        result = _find_span_in_text(span_text, text, text_norm)
        if result is None:
            logger.warning(
                "Span text not found in sentence:\n  span=%r\n  sentence=%r",
                span_text,
                text,
            )
            continue

        start, end = result
        # Mark chars only if not already taken (longest wins).
        if all(char_label[i] is None for i in range(start, end)):
            for i in range(start, end):
                char_label[i] = entity_type

    # Assign BIO tags to tokens using char offsets.
    for i, (tok_start, tok_end) in enumerate(offsets):
        # Get the entity type covering the majority of this token's chars.
        labels_in_token = [char_label[j] for j in range(tok_start, tok_end) if char_label[j]]
        if not labels_in_token:
            bio_tags[i] = "O"
            continue

        entity_type = max(set(labels_in_token), key=labels_in_token.count)

        # Determine B- vs I- by checking the previous token.
        if i == 0 or bio_tags[i - 1] == "O":
            bio_tags[i] = f"B-{entity_type}"
        else:
            prev_type = bio_tags[i - 1].split("-", 1)[-1]
            if prev_type == entity_type:
                bio_tags[i] = f"I-{entity_type}"
            else:
                bio_tags[i] = f"B-{entity_type}"

    return tokens, offsets, bio_tags


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------


def _parse_spans(raw: str) -> list[dict[str, str]]:
    """Parse the spans column value (JSON string) into a list of dicts.

    Tolerates empty strings and whitespace-only values.

    Args:
        raw: Raw string from the CSV cell.

    Returns:
        List of span dicts, empty list if raw is empty or invalid.
    """
    raw = raw.strip()
    if not raw or raw in ("[]", ""):
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        logger.warning("Could not parse spans JSON: %r", raw)
    return []


def _read_review_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Read the review CSV into a list of row dicts.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of row dicts with keys matching the CSV columns.
    """
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    logger.info("Read %d rows from %s", len(rows), csv_path)
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_gold(
    review_csv: Path = REVIEW_CSV,
    gold_jsonl: Path = GOLD_JSONL,
    report_path: Path = BUILD_REPORT,
) -> list[dict[str, Any]]:
    """Build the gold JSONL from the reviewed CSV.

    Reads each row in the review CSV, applies the corrected span annotations
    to generate BIO tags, and writes the resulting records to gold_jsonl.

    Rows with ``status == "erro"`` (or other skip values) are excluded.
    Rows with ``status == "ok"`` use ``spans_silver`` as-is.
    Rows with ``status == "corrigido"`` use ``spans_correto``.

    Args:
        review_csv: Path to the filled review CSV.
        gold_jsonl: Destination JSONL file.
        report_path: Destination for the build summary JSON.

    Returns:
        List of gold records written to disk.
    """
    rows = _read_review_csv(review_csv)

    gold_records: list[dict[str, Any]] = []
    summary = {
        "total_rows": len(rows),
        "accepted_ok": 0,
        "accepted_corrigido": 0,
        "skipped_erro": 0,
        "skipped_pendente": 0,
        "warnings": 0,
    }

    for row in rows:
        status = row.get("status", "pendente").strip().lower()

        if status in _SKIP_STATUSES:
            summary["skipped_erro"] += 1
            continue

        if status == "pendente":
            # Treat unreviewed rows as accepted (using silver spans).
            logger.warning(
                "Row %s still 'pendente' -- using silver spans.", row.get("sentenca_id")
            )
            summary["skipped_pendente"] += 1
            spans_raw = row.get("spans_silver", "[]")
        elif status == "corrigido":
            spans_raw = row.get("spans_correto", row.get("spans_silver", "[]"))
            summary["accepted_corrigido"] += 1
        else:  # "ok"
            spans_raw = row.get("spans_silver", "[]")
            summary["accepted_ok"] += 1

        spans = _parse_spans(spans_raw)
        texto = row.get("texto", "").strip()

        if not texto:
            logger.warning("Empty text for %s -- skipping.", row.get("sentenca_id"))
            summary["warnings"] += 1
            continue

        tokens, _offsets, bio_tags = _spans_to_bio(texto, spans)

        # Reconstruct final spans from bio_tags for consistency.
        final_spans = _bio_to_spans(tokens, bio_tags)

        gold_records.append(
            {
                "relatorio": row.get("relatorio", ""),
                "pagina": int(row.get("pagina", 0) or 0),
                "sentenca_id": row.get("sentenca_id", ""),
                "texto": texto,
                "tokens": tokens,
                "bio_tags": bio_tags,
                "spans": final_spans,
            }
        )

    # Write JSONL.
    gold_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with gold_jsonl.open("w", encoding="utf-8") as fh:
        for record in gold_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary["gold_sentences"] = len(gold_records)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    logger.info(
        "Gold built: %d sentences (%d ok, %d corrigido, %d erros, %d pendentes).",
        len(gold_records),
        summary["accepted_ok"],
        summary["accepted_corrigido"],
        summary["skipped_erro"],
        summary["skipped_pendente"],
    )
    logger.info("Written to %s", gold_jsonl)
    return gold_records


def _bio_to_spans(
    tokens: list[str], bio_tags: list[str]
) -> list[dict[str, str]]:
    """Reconstruct entity spans from BIO-tagged token list.

    Args:
        tokens: Whitespace tokens.
        bio_tags: Corresponding BIO tags.

    Returns:
        List of {"texto": str, "tipo": str} dicts.
    """
    spans: list[dict[str, str]] = []
    current_tokens: list[str] = []
    current_type: str | None = None

    for token, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            if current_tokens and current_type:
                spans.append({"texto": " ".join(current_tokens), "tipo": current_type})
            current_tokens = [token]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_tokens.append(token)
        else:
            if current_tokens and current_type:
                spans.append({"texto": " ".join(current_tokens), "tipo": current_type})
            current_tokens = []
            current_type = None

    if current_tokens and current_type:
        spans.append({"texto": " ".join(current_tokens), "tipo": current_type})

    return spans


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    build_gold()
