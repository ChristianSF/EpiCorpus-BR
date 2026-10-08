"""
Convert a gold-candidates JSONL file to Label Studio import format.

The silver BIO annotations are included as predictions so the annotator
sees them pre-filled and only needs to correct errors.

Usage:
    # text candidates (default)
    python -m src.to_label_studio

    # table candidates
    python -m src.to_label_studio --tables

Output:
    annotation/samples/label_studio_import.json       (text)
    annotation/samples/label_studio_tables_import.json (tables)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.config import ANNOTATION_SAMPLES

logger = logging.getLogger(__name__)

TEXT_CANDIDATES = ANNOTATION_SAMPLES / "gold_candidates.jsonl"
TABLE_CANDIDATES = ANNOTATION_SAMPLES / "gold_table_candidates.jsonl"
TEXT_OUTPUT = ANNOTATION_SAMPLES / "label_studio_import.json"
TABLE_OUTPUT = ANNOTATION_SAMPLES / "label_studio_tables_import.json"


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Return list of (token, start, end) with character offsets."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _bio_to_ls_results(
    text: str,
    tokens: list[str],
    bio_tags: list[str],
) -> list[dict]:
    """Convert BIO tags to Label Studio result items (char-offset spans)."""
    tok_info = _tokenize(text)
    if len(tok_info) != len(tokens):
        # Fallback: re-tokenize from the stored tokens list
        tok_info = []
        pos = 0
        for tok in tokens:
            idx = text.find(tok, pos)
            if idx == -1:
                idx = pos
            tok_info.append((tok, idx, idx + len(tok)))
            pos = idx + len(tok)

    results = []
    i = 0
    uid = 0
    while i < len(bio_tags):
        tag = bio_tags[i]
        if not tag.startswith("B-"):
            i += 1
            continue

        label = tag[2:]
        span_start = tok_info[i][1]
        span_end = tok_info[i][2]
        j = i + 1
        while j < len(bio_tags) and bio_tags[j] == f"I-{label}":
            span_end = tok_info[j][2]
            j += 1

        span_text = text[span_start:span_end]
        results.append({
            "id": f"t{uid}",
            "type": "labels",
            "from_name": "label",
            "to_name": "text",
            "value": {
                "start": span_start,
                "end": span_end,
                "text": span_text,
                "labels": [label],
            },
        })
        uid += 1
        i = j

    return results


def convert(
    candidates_path: Path,
    output_path: Path,
) -> None:
    records = []
    with candidates_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    tasks = []
    for idx, rec in enumerate(records, start=1):
        text = rec["text"] if "text" in rec else rec["texto"]
        tokens = rec.get("tokens", [])
        bio_tags = rec.get("bio_tags", [])

        ls_results = _bio_to_ls_results(text, tokens, bio_tags)

        tasks.append({
            "id": idx,
            "data": {
                "text": text,
                "sentenca_id": rec["sentenca_id"],
                "relatorio": rec["relatorio"],
                "pagina": rec.get("pagina", 0),
            },
            "predictions": [
                {
                    "model_version": "silver_gpt4o_mini",
                    "score": 0.8,
                    "result": ls_results,
                }
            ],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    logger.info("Exported %d tasks to %s", len(tasks), output_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", action="store_true",
                        help="Convert table candidates instead of text candidates.")
    args = parser.parse_args()

    if args.tables:
        convert(TABLE_CANDIDATES, TABLE_OUTPUT)
    else:
        convert(TEXT_CANDIDATES, TEXT_OUTPUT)
