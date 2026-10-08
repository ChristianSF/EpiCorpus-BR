"""
NER evaluation: silver vs gold using seqeval.

Compares the automatic silver annotations (GPT-4o-mini) with the manually
reviewed gold standard for the 180-sentence sample.  Reports per-category
and macro-averaged Precision, Recall and F1 (strict entity matching via
seqeval).

Usage (CLI):
    python -m src.evaluation

Output:
    - Console table with per-category P/R/F1 + macro averages.
    - data/evaluation/results.json  -- machine-readable results dict.
    - data/evaluation/results.csv   -- same as CSV for LaTeX table generation.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from seqeval.scheme import IOB2

from src.config import DATA_GOLD, DATA_SILVER, DATA_SILVER_TABLES, PROJECT_ROOT
from src.utils import read_jsonl

logger = logging.getLogger(__name__)

EVAL_DIR: Path = PROJECT_ROOT / "data" / "evaluation"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_gold(gold_path: Path) -> dict[str, list[str]]:
    """Load gold BIO tag sequences keyed by sentenca_id.

    Args:
        gold_path: Path to the gold JSONL file.

    Returns:
        Mapping from sentenca_id to list of BIO tag strings.
    """
    gold: dict[str, list[str]] = {}
    for record in read_jsonl(gold_path):
        sid = record["sentenca_id"]
        tags = record.get("bio_tags", [])
        if isinstance(tags, list) and tags:
            gold[sid] = [str(t) for t in tags]
    return gold


def _load_silver_index(silver_dir: Path) -> dict[str, list[str]]:
    """Build a sentenca_id -> bio_tags index from all silver JSONL files.

    Args:
        silver_dir: Directory containing per-report silver JSONL files.

    Returns:
        Mapping from sentenca_id to list of BIO tag strings.
    """
    index: dict[str, list[str]] = {}
    for jsonl_path in sorted(silver_dir.glob("*.jsonl")):
        for record in read_jsonl(jsonl_path):
            sid = record["sentenca_id"]
            tags = record.get("bio_tags", [])
            if isinstance(tags, list) and tags:
                index[sid] = [str(t) for t in tags]
    return index


def _align_sequences(
    gold_map: dict[str, list[str]],
    silver_map: dict[str, list[str]],
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Pair gold and silver tag sequences for sentences that exist in both.

    Sequences of different lengths are truncated to the shorter one with a
    warning; this should not happen if both were produced from the same
    tokeniser.

    Args:
        gold_map: sentenca_id -> gold bio_tags.
        silver_map: sentenca_id -> silver bio_tags.

    Returns:
        Tuple of (gold_seqs, silver_seqs, matched_ids) where matched_ids
        lists the sentenca_ids that were successfully paired.
    """
    gold_seqs: list[list[str]] = []
    silver_seqs: list[list[str]] = []
    matched: list[str] = []
    missing = 0

    for sid, g_tags in sorted(gold_map.items()):
        if sid not in silver_map:
            logger.warning("sentenca_id %s not found in silver -- skipping.", sid)
            missing += 1
            continue

        s_tags = silver_map[sid]

        if len(g_tags) != len(s_tags):
            min_len = min(len(g_tags), len(s_tags))
            logger.warning(
                "Length mismatch for %s (gold=%d, silver=%d) -- truncating to %d.",
                sid,
                len(g_tags),
                len(s_tags),
                min_len,
            )
            g_tags = g_tags[:min_len]
            s_tags = s_tags[:min_len]

        gold_seqs.append(g_tags)
        silver_seqs.append(s_tags)
        matched.append(sid)

    if missing:
        logger.warning("%d gold sentences had no matching silver record.", missing)

    return gold_seqs, silver_seqs, matched


def _per_category_scores(
    gold_seqs: list[list[str]],
    silver_seqs: list[list[str]],
    entity_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-category P/R/F1 using seqeval classification_report.

    Args:
        gold_seqs: List of gold BIO tag sequences.
        silver_seqs: List of predicted (silver) BIO tag sequences.
        entity_names: List of entity type names to include.

    Returns:
        Dict mapping entity name (or 'macro avg') to {'precision', 'recall', 'f1'}.
    """
    report_str = classification_report(
        gold_seqs,
        silver_seqs,
        mode="strict",
        scheme=IOB2,
        output_dict=False,
        zero_division=0,
    )
    report_dict: dict[str, Any] = classification_report(
        gold_seqs,
        silver_seqs,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )

    results: dict[str, dict[str, float]] = {}

    for name in entity_names:
        if name in report_dict:
            row = report_dict[name]
            results[name] = {
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1": round(row["f1-score"], 4),
                "support": int(row["support"]),
            }
        else:
            results[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    # Macro and micro averages from seqeval
    for avg_key in ("macro avg", "micro avg", "weighted avg"):
        if avg_key in report_dict:
            row = report_dict[avg_key]
            results[avg_key] = {
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1": round(row["f1-score"], 4),
                "support": int(row.get("support", 0)),
            }

    logger.info("\n%s", report_str)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_combined(
    output_dir: Path = EVAL_DIR,
) -> dict[str, dict[str, float]]:
    """Evaluate silver vs gold concatenating text and table sets.

    Loads gold_sample.jsonl + gold_tables.jsonl and their matching silver
    directories, aligns all sequences, and runs a single seqeval pass so
    the results reflect the full corpus (text + tables).

    Args:
        output_dir: Where to write ``results_combined.json`` and ``.csv``.

    Returns:
        Results dict as returned by :func:`_per_category_scores`.
    """
    from src.config import ENTITY_TYPES

    entity_names = [e["name"] for e in ENTITY_TYPES]

    gold_text = _load_gold(DATA_GOLD / "gold_sample.jsonl")
    gold_tables = _load_gold(DATA_GOLD / "gold_tables.jsonl")
    gold_map = {**gold_text, **gold_tables}
    logger.info("Combined gold: %d text + %d table = %d sentences",
                len(gold_text), len(gold_tables), len(gold_map))

    silver_text = _load_silver_index(DATA_SILVER)
    silver_tables = _load_silver_index(DATA_SILVER_TABLES)
    silver_map = {**silver_text, **silver_tables}
    logger.info("Combined silver index: %d sentences", len(silver_map))

    gold_seqs, silver_seqs, matched_ids = _align_sequences(gold_map, silver_map)
    logger.info("Evaluating %d aligned sentence pairs ...", len(matched_ids))

    results = _per_category_scores(gold_seqs, silver_seqs, entity_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results_combined.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", json_path)

    csv_path = output_dir / "results_combined.csv"
    _write_csv(results, entity_names, csv_path)
    logger.info("CSV saved to %s", csv_path)

    _print_table(results, entity_names)
    return results


def evaluate(
    gold_path: Path = DATA_GOLD / "gold_sample.jsonl",
    silver_dir: Path = DATA_SILVER,
    output_dir: Path = EVAL_DIR,
    results_stem: str = "results",
) -> dict[str, dict[str, float]]:
    """Compare silver vs gold annotations and save results.

    Loads all gold BIO sequences and the matching silver sequences, calls
    seqeval for strict entity-level evaluation, and writes results to
    ``output_dir/results.json`` and ``output_dir/results.csv``.

    Args:
        gold_path: Path to the gold standard JSONL file.
        silver_dir: Directory containing silver JSONL files.
        output_dir: Where to write ``results.json`` and ``results.csv``.

    Returns:
        Results dict as returned by :func:`_per_category_scores`.
    """
    from src.config import ENTITY_TYPES  # avoid circular import at module level

    entity_names = [e["name"] for e in ENTITY_TYPES]

    logger.info("Loading gold annotations from %s ...", gold_path)
    gold_map = _load_gold(gold_path)
    logger.info("  %d gold sentences loaded.", len(gold_map))

    logger.info("Loading silver annotations from %s ...", silver_dir)
    silver_map = _load_silver_index(silver_dir)
    logger.info("  %d silver sentences indexed.", len(silver_map))

    gold_seqs, silver_seqs, matched_ids = _align_sequences(gold_map, silver_map)
    logger.info("Evaluating %d aligned sentence pairs ...", len(matched_ids))

    results = _per_category_scores(gold_seqs, silver_seqs, entity_names)

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{results_stem}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", json_path)

    csv_path = output_dir / f"{results_stem}.csv"
    _write_csv(results, entity_names, csv_path)
    logger.info("CSV saved to %s", csv_path)

    _print_table(results, entity_names)

    return results


def _write_csv(
    results: dict[str, dict[str, float]],
    entity_names: list[str],
    csv_path: Path,
) -> None:
    """Write results to a CSV file suitable for importing into LaTeX/pandas.

    Args:
        results: Per-category results dict.
        entity_names: Ordered list of entity type names.
        csv_path: Destination path.
    """
    rows = []
    for name in entity_names:
        row = results.get(name, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        rows.append(
            {
                "entity": name,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "support": row.get("support", 0),
            }
        )

    for avg_key in ("macro avg", "micro avg", "weighted avg"):
        if avg_key in results:
            row = results[avg_key]
            rows.append(
                {
                    "entity": avg_key,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "support": row.get("support", 0),
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["entity", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)


def _print_table(
    results: dict[str, dict[str, float]],
    entity_names: list[str],
) -> None:
    """Print a formatted results table to stdout.

    Args:
        results: Per-category results dict.
        entity_names: Ordered list of entity type names.
    """
    header = f"{'Entidade':<25}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Suporte':>8}"
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  NER Evaluation: Silver vs Gold (seqeval strict, IOB2)")
    print("=" * len(header))
    print(header)
    print(sep)

    for name in entity_names:
        row = results.get(name, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        sup = row.get("support", 0)
        print(
            f"{name:<25}  {row['precision']:>6.4f}  {row['recall']:>6.4f}  {row['f1']:>6.4f}  {sup:>8}"
        )

    print(sep)
    for avg_key in ("macro avg", "micro avg", "weighted avg"):
        if avg_key in results:
            row = results[avg_key]
            sup = row.get("support", 0)
            print(
                f"{avg_key:<25}  {row['precision']:>6.4f}  {row['recall']:>6.4f}  {row['f1']:>6.4f}  {sup:>8}"
            )

    print("=" * len(header))
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate silver vs gold NER annotations.")
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Evaluate table annotations (gold_tables.jsonl vs silver_tables/).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Evaluate text + tables together (gold_sample + gold_tables vs silver + silver_tables).",
    )
    args = parser.parse_args()

    if args.combined:
        evaluate_combined()
    elif args.tables:
        evaluate(
            gold_path=DATA_GOLD / "gold_tables.jsonl",
            silver_dir=DATA_SILVER_TABLES,
            output_dir=EVAL_DIR,
            results_stem="results_tables",
        )
    else:
        evaluate()
