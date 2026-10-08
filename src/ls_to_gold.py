"""
Convert a Label Studio JSON export to gold JSONL.

Usage:
    # text gold (default)
    python -m src.ls_to_gold <export.json>

    # table gold
    python -m src.ls_to_gold <export.json> --tables

Output:
    data/gold/gold_sample.jsonl   (text, default)
    data/gold/gold_tables.jsonl   (tables, with --tables)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from src.config import DATA_GOLD
from src.gold_builder import _spans_to_bio, _bio_to_spans

logger = logging.getLogger(__name__)

GOLD_JSONL = DATA_GOLD / "gold_sample.jsonl"
GOLD_TABLES_JSONL = DATA_GOLD / "gold_tables.jsonl"


def convert(export_path: Path, gold_jsonl: Path = GOLD_JSONL) -> None:
    with export_path.open(encoding="utf-8") as f:
        tasks = json.load(f)

    records = []
    for task in tasks:
        data = task["data"]
        text = data["text"]
        sentenca_id = data["sentenca_id"]
        relatorio = data["relatorio"]
        pagina = data.get("pagina", 0)

        # Pick the first non-cancelled manual annotation
        manual = [
            a for a in task.get("annotations", [])
            if not a.get("was_cancelled") and a.get("completed_by") == 1
        ]
        if not manual:
            logger.warning("No manual annotation for %s -- skipping.", sentenca_id)
            continue

        result = manual[0]["result"]

        # Convert Label Studio char-offset spans to {"texto", "tipo"} dicts
        spans = []
        for item in result:
            val = item.get("value", {})
            labels = val.get("labels", [])
            span_text = val.get("text", "").strip()
            if not labels or not span_text:
                continue
            tipo = labels[0]
            if tipo == "METRICA_EPI":
                # Extract individual numbers (handles pipe-separated blocks)
                for num in re.findall(r"\d+(?:[.,]\d+)*", span_text):
                    spans.append({"texto": num, "tipo": tipo})
            else:
                spans.append({"texto": span_text, "tipo": tipo})

        tokens, _offsets, bio_tags = _spans_to_bio(text, spans)
        final_spans = _bio_to_spans(tokens, bio_tags)

        records.append({
            "relatorio": relatorio,
            "pagina": int(pagina or 0),
            "sentenca_id": sentenca_id,
            "texto": text,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "spans": final_spans,
        })

    gold_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with gold_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Written %d records to %s", len(records), gold_jsonl)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("export", help="Path to Label Studio export JSON.")
    parser.add_argument("--tables", action="store_true",
                        help="Write to gold_tables.jsonl instead of gold_sample.jsonl.")
    args = parser.parse_args()

    out = GOLD_TABLES_JSONL if args.tables else GOLD_JSONL
    convert(Path(args.export), gold_jsonl=out)
