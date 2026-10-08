"""
Stratified sampling for gold-standard annotation set.

Selects a representative sample of silver-annotated sentences for manual
review, stratifying by report year (and optionally by entity coverage) so
that every year in the corpus is represented proportionally.

Usage (CLI):
    python -m src.sampling                    # default: 180 sentences
    python -m src.sampling --n 200 --seed 99

Output:
    annotation/samples/gold_candidates.jsonl  -- sentences ready for Label Studio import.
    annotation/samples/sampling_log.json      -- per-year counts and metadata.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import ANNOTATION_SAMPLES, DATA_SILVER, DATA_SILVER_TABLES
from src.utils import read_jsonl

logger = logging.getLogger(__name__)

# Default sample size, matching the 180-sentence gold set already collected.
DEFAULT_N: int = 180
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_silver_by_report(silver_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all silver records grouped by relatorio slug.

    Args:
        silver_dir: Directory containing per-report silver JSONL files.

    Returns:
        Dict mapping relatorio name to list of sentence records.
    """
    by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for jsonl_path in sorted(silver_dir.glob("*.jsonl")):
        for record in read_jsonl(jsonl_path):
            by_report[record["relatorio"]].append(record)
    return dict(by_report)


def _proportional_allocation(
    counts: dict[str, int],
    total: int,
    min_per_stratum: int = 1,
) -> dict[str, int]:
    """Allocate ``total`` slots across strata proportionally.

    Uses the largest-remainder method so that the sum of allocations equals
    ``total`` exactly.  Every stratum gets at least ``min_per_stratum``.

    Args:
        counts: Number of available sentences per stratum.
        total: Total slots to allocate.
        min_per_stratum: Minimum allocation per stratum (if available sentences allow).

    Returns:
        Dict mapping stratum name to integer allocation.
    """
    n_strata = len(counts)
    grand_total = sum(counts.values())

    if grand_total == 0:
        return {k: 0 for k in counts}

    # Enforce min_per_stratum first.
    allocation: dict[str, int] = {}
    reserved = 0
    for k, avail in counts.items():
        alloc = min(min_per_stratum, avail)
        allocation[k] = alloc
        reserved += alloc

    remaining_slots = total - reserved
    if remaining_slots < 0:
        # Cannot even fill minimum -- just use minimums.
        return allocation

    # Proportional distribution of remaining slots.
    proportions = {k: v / grand_total for k, v in counts.items()}
    exact = {k: proportions[k] * remaining_slots for k in counts}
    floors = {k: int(v) for k, v in exact.items()}
    remainders = {k: exact[k] - floors[k] for k in counts}

    for k in counts:
        allocation[k] += floors[k]

    leftover = remaining_slots - sum(floors.values())
    for k in sorted(remainders, key=lambda x: -remainders[x])[:leftover]:
        allocation[k] += 1

    # Clip to available sentences.
    for k in allocation:
        allocation[k] = min(allocation[k], counts[k])

    return allocation


def _has_entity(record: dict[str, Any]) -> bool:
    """Return True if the record contains at least one non-O tag."""
    return any(t != "O" for t in record.get("bio_tags", []))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_gold_candidates(
    silver_dir: Path = DATA_SILVER,
    output_dir: Path = ANNOTATION_SAMPLES,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    entity_oversample_ratio: float = 0.5,
    output_filename: str = "gold_candidates.jsonl",
    log_filename: str = "sampling_log.json",
) -> list[dict[str, Any]]:
    """Sample ``n`` sentences from the silver annotations for manual review.

    Stratifies by report (year) so each year is proportionally represented.
    Within each stratum, at least ``entity_oversample_ratio`` of sampled
    sentences contain at least one entity annotation, ensuring the gold set
    is rich enough for meaningful evaluation.

    Args:
        silver_dir: Directory containing silver JSONL files.
        output_dir: Where to write the output files.
        n: Total number of sentences to sample.
        seed: Random seed for reproducibility.
        entity_oversample_ratio: Fraction of each stratum that should come
            from entity-bearing sentences.  0 = pure random; 1 = all entities.

    Returns:
        List of selected sentence records (same schema as silver JSONL).
    """
    rng = random.Random(seed)

    logger.info("Loading silver annotations from %s ...", silver_dir)
    by_report = _load_silver_by_report(silver_dir)

    if not by_report:
        logger.error("No silver records found in %s", silver_dir)
        return []

    counts = {k: len(v) for k, v in by_report.items()}
    logger.info("Reports: %s", sorted(counts.keys()))
    logger.info("Total silver sentences: %d", sum(counts.values()))

    allocation = _proportional_allocation(counts, n, min_per_stratum=5)
    logger.info("Allocation per report: %s", allocation)

    sampled: list[dict[str, Any]] = []
    log: dict[str, Any] = {"n": n, "seed": seed, "strata": {}}

    for report, alloc in sorted(allocation.items()):
        pool = by_report[report]
        if alloc == 0:
            continue

        entity_pool = [r for r in pool if _has_entity(r)]
        non_entity_pool = [r for r in pool if not _has_entity(r)]

        # Split alloc between entity-bearing and non-entity sentences.
        n_entity = min(round(alloc * entity_oversample_ratio), len(entity_pool))
        n_non = min(alloc - n_entity, len(non_entity_pool))

        # If one pool is too small, compensate from the other.
        shortfall_entity = round(alloc * entity_oversample_ratio) - n_entity
        shortfall_non = (alloc - round(alloc * entity_oversample_ratio)) - n_non
        n_entity += min(shortfall_non, len(entity_pool) - n_entity)
        n_non += min(shortfall_entity, len(non_entity_pool) - n_non)

        chosen_entity = rng.sample(entity_pool, min(n_entity, len(entity_pool)))
        chosen_non = rng.sample(non_entity_pool, min(n_non, len(non_entity_pool)))
        chosen = chosen_entity + chosen_non

        # If we still need more (both pools exhausted), draw from full pool.
        if len(chosen) < alloc:
            chosen_ids = {r["sentenca_id"] for r in chosen}
            extras = [r for r in pool if r["sentenca_id"] not in chosen_ids]
            chosen += rng.sample(extras, min(alloc - len(chosen), len(extras)))

        sampled.extend(chosen)
        log["strata"][report] = {
            "allocated": alloc,
            "sampled": len(chosen),
            "with_entity": sum(1 for r in chosen if _has_entity(r)),
        }
        logger.info(
            "  %s: %d/%d sampled (%d with entity)",
            report,
            len(chosen),
            alloc,
            log["strata"][report]["with_entity"],
        )

    # Shuffle final list so report order is not preserved.
    rng.shuffle(sampled)
    logger.info("Total sampled: %d sentences", len(sampled))

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = output_dir / output_filename
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for record in sampled:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Candidates saved to %s", out_jsonl)

    log_path = output_dir / log_filename
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=2)
    logger.info("Log saved to %s", log_path)

    return sampled


def sample_summary(candidates: list[dict[str, Any]]) -> None:
    """Print a summary table of the sampled candidates to stdout.

    Args:
        candidates: Sampled sentence records.
    """
    from collections import Counter

    by_report: Counter = Counter(r["relatorio"] for r in candidates)
    with_entity = sum(1 for r in candidates if _has_entity(r))

    print()
    print("=" * 55)
    print("  Sampling summary")
    print("=" * 55)
    print(f"  Total sentences : {len(candidates)}")
    print(f"  With entity     : {with_entity} ({100 * with_entity / max(len(candidates), 1):.1f}%)")
    print(f"  Without entity  : {len(candidates) - with_entity}")
    print("-" * 55)
    print(f"  {'Report':<30}  {'Count':>6}")
    print("-" * 55)
    for report, count in sorted(by_report.items()):
        print(f"  {report:<30}  {count:>6}")
    print("=" * 55)
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

    parser = argparse.ArgumentParser(description="Sample sentences for gold annotation.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Total sentences to sample.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--entity-ratio",
        type=float,
        default=0.5,
        help="Fraction of each stratum drawn from entity-bearing sentences.",
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Sample from data/silver_tables/ instead of data/silver/. "
             "Output: gold_table_candidates.jsonl",
    )
    args = parser.parse_args()

    if args.tables:
        silver_dir = DATA_SILVER_TABLES
        n = args.n if args.n != DEFAULT_N else 100
        out_filename = "gold_table_candidates.jsonl"
        log_filename = "sampling_table_log.json"
    else:
        silver_dir = DATA_SILVER
        n = args.n
        out_filename = "gold_candidates.jsonl"
        log_filename = "sampling_log.json"

    candidates = sample_gold_candidates(
        silver_dir=silver_dir,
        output_dir=ANNOTATION_SAMPLES,
        n=n,
        seed=args.seed,
        entity_oversample_ratio=args.entity_ratio,
        output_filename=out_filename,
        log_filename=log_filename,
    )
    sample_summary(candidates)
