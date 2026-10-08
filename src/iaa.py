"""
Inter-Annotator Agreement (IAA) for the SIREVA-SUS NER corpus.

Compares two annotation CSV files (same schema as annotation_texto.csv):
  - annotation/review/annotation_texto.csv      : annotator 1
  - annotation/review/annotation_texto_ann2.csv : annotator 2

Final annotation per sentence per annotator:
  entidades_corrigidas  when non-empty  → explicit correction
  entidades_sugeridas   otherwise       → accepted silver as-is

Metrics produced
  - Token-level Cohen's Kappa (over IOB2 label pairs)
  - Token-level % agreement
  - Span-level F1 symmetric: 2|A∩B| / (|A|+|B|)  averaged over sentences
  - Per-entity breakdown of both token-kappa and span-F1

Output
  data/evaluation/iaa_report.json
  Formatted table printed to stdout
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from src.config import PROJECT_ROOT, VALID_ENTITY_NAMES
from src.gold_builder import _spans_to_bio

logger = logging.getLogger(__name__)

ANN1_CSV = PROJECT_ROOT / "annotation" / "review" / "annotation_texto.csv"
ANN2_CSV = PROJECT_ROOT / "annotation" / "review" / "annotation_texto_ann2.csv"
IAA_REPORT = PROJECT_ROOT / "data" / "evaluation" / "iaa_report.json"

_NENHUMA_RE = re.compile(r"^\s*\(?\s*nenhuma\s*\)?\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def _parse_entity_cell(cell: str) -> list[dict[str, str]]:
    """Parse 'texto → TIPO; texto2 → TIPO2' into span dicts."""
    cell = cell.strip()
    if not cell or _NENHUMA_RE.match(cell):
        return []
    spans: list[dict[str, str]] = []
    for part in cell.split(";"):
        part = part.strip()
        if not part or "→" not in part:
            continue
        text_part, _, type_part = part.partition("→")
        span_text = text_part.strip()
        span_type = type_part.strip().upper()
        if not span_text:
            continue
        if span_type not in VALID_ENTITY_NAMES:
            logger.debug("Unknown entity type %r – skipping span.", span_type)
            continue
        spans.append({"texto": span_text, "tipo": span_type})
    return spans


def _get_annotation(row: dict) -> list[dict[str, str]]:
    """Resolve final spans: corrigidas if present, else sugeridas."""
    corr = row.get("entidades_corrigidas", "").strip()
    if corr and not _NENHUMA_RE.match(corr):
        return _parse_entity_cell(corr)
    return _parse_entity_cell(row.get("entidades_sugeridas", ""))


def _read_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows[row["id"]] = dict(row)
    logger.info("Loaded %d rows from %s", len(rows), path.name)
    return rows


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    if n == 0:
        return 1.0
    label_set = set(labels_a) | set(labels_b)
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    freq_a: dict[str, int] = defaultdict(int)
    freq_b: dict[str, int] = defaultdict(int)
    for la, lb in zip(labels_a, labels_b):
        freq_a[la] += 1
        freq_b[lb] += 1
    p_e = sum((freq_a[lbl] / n) * (freq_b[lbl] / n) for lbl in label_set)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def _span_f1_symmetric(
    spans_a: list[tuple[str, str]], spans_b: list[tuple[str, str]]
) -> float:
    """Symmetric F1: 2|A∩B| / (|A|+|B|).  Both empty → 1.0."""
    if not spans_a and not spans_b:
        return 1.0
    denom = len(spans_a) + len(spans_b)
    if denom == 0:
        return 1.0
    inter = len(set(spans_a) & set(spans_b))
    return 2 * inter / denom


def _spans_as_tuples(spans: list[dict]) -> list[tuple[str, str]]:
    return [(s["texto"].lower().strip(), s["tipo"]) for s in spans]


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_iaa(
    ann1_csv: Path = ANN1_CSV,
    ann2_csv: Path = ANN2_CSV,
    report_path: Path = IAA_REPORT,
) -> dict:
    rows1 = _read_csv(ann1_csv)
    rows2 = _read_csv(ann2_csv)

    common_ids = sorted(set(rows1) & set(rows2))
    if not common_ids:
        raise ValueError("No common sentence IDs found between the two CSVs.")
    logger.info("Sentences in common: %d", len(common_ids))

    # Accumulators
    all_tags_a: list[str] = []
    all_tags_b: list[str] = []

    span_f1_per_sent: list[float] = []

    # Per-entity span counts (for global span-F1 per entity)
    ent_span: dict[str, dict[str, int]] = defaultdict(
        lambda: {"intersect": 0, "total_a": 0, "total_b": 0}
    )

    # Per-entity token sequences (for kappa per entity)
    ent_tok_a: dict[str, list[str]] = defaultdict(list)
    ent_tok_b: dict[str, list[str]] = defaultdict(list)

    skipped = 0
    for sid in common_ids:
        row1, row2 = rows1[sid], rows2[sid]
        text = row1.get("texto", "").strip()
        if not text:
            skipped += 1
            continue

        spans1 = _get_annotation(row1)
        spans2 = _get_annotation(row2)

        _, _, tags1 = _spans_to_bio(text, spans1)
        _, _, tags2 = _spans_to_bio(text, spans2)

        # Both calls use the same text → same token count; guard anyway
        min_len = min(len(tags1), len(tags2))
        tags1, tags2 = tags1[:min_len], tags2[:min_len]

        all_tags_a.extend(tags1)
        all_tags_b.extend(tags2)

        # Span metrics
        tups1 = _spans_as_tuples(spans1)
        tups2 = _spans_as_tuples(spans2)
        span_f1_per_sent.append(_span_f1_symmetric(tups1, tups2))

        # Per-entity span counts
        set1: set[tuple[str, str]] = set(tups1)
        set2: set[tuple[str, str]] = set(tups2)
        for etype in {tp for _, tp in set1 | set2}:
            s1 = {txt for txt, tp in set1 if tp == etype}
            s2 = {txt for txt, tp in set2 if tp == etype}
            ent_span[etype]["intersect"] += len(s1 & s2)
            ent_span[etype]["total_a"] += len(s1)
            ent_span[etype]["total_b"] += len(s2)

        # Per-entity token sequences
        for t1, t2 in zip(tags1, tags2):
            e1 = t1.split("-", 1)[-1] if t1 != "O" else None
            e2 = t2.split("-", 1)[-1] if t2 != "O" else None
            entity = e1 or e2
            if entity:
                ent_tok_a[entity].append(t1)
                ent_tok_b[entity].append(t2)

    if skipped:
        logger.warning("Skipped %d rows with empty text.", skipped)

    # Overall metrics
    n_tokens = len(all_tags_a)
    agreement_pct = (
        sum(a == b for a, b in zip(all_tags_a, all_tags_b)) / n_tokens * 100
        if n_tokens else 0.0
    )
    overall_kappa = _cohen_kappa(all_tags_a, all_tags_b)
    mean_span_f1 = (
        sum(span_f1_per_sent) / len(span_f1_per_sent) if span_f1_per_sent else 0.0
    )

    # Per-entity kappa
    per_entity_kappa = {
        etype: round(_cohen_kappa(ent_tok_a[etype], ent_tok_b[etype]), 4)
        for etype in sorted(ent_tok_a)
    }

    # Per-entity span F1
    per_entity_span_f1: dict[str, float] = {}
    for etype, counts in sorted(ent_span.items()):
        denom = counts["total_a"] + counts["total_b"]
        per_entity_span_f1[etype] = (
            round(2 * counts["intersect"] / denom, 4) if denom else 1.0
        )

    report = {
        "sentences_compared": len(common_ids) - skipped,
        "total_tokens": n_tokens,
        "overall": {
            "token_agreement_pct": round(agreement_pct, 2),
            "cohen_kappa": round(overall_kappa, 4),
            "mean_span_f1": round(mean_span_f1, 4),
        },
        "per_entity_token_kappa": per_entity_kappa,
        "per_entity_span_f1": per_entity_span_f1,
        "per_entity_span_counts": {k: v for k, v in sorted(ent_span.items())},
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("IAA report written to %s", report_path)

    return report


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def _print_report(report: dict) -> None:
    ov = report["overall"]
    print("\n=== Inter-Annotator Agreement ===")
    print(f"  Sentences compared : {report['sentences_compared']}")
    print(f"  Total tokens       : {report['total_tokens']}")
    print(f"  Token agreement    : {ov['token_agreement_pct']:.1f}%")
    print(f"  Cohen's Kappa      : {ov['cohen_kappa']:.4f}")
    print(f"  Mean span F1       : {ov['mean_span_f1']:.4f}")

    print("\n--- Per-entity token Kappa ---")
    kappas = report["per_entity_token_kappa"]
    for etype in sorted(kappas, key=lambda e: -kappas[e]):
        print(f"  {etype:<25} kappa = {kappas[etype]:.4f}")

    print("\n--- Per-entity span F1 ---")
    sf1 = report["per_entity_span_f1"]
    counts = report["per_entity_span_counts"]
    for etype in sorted(sf1, key=lambda e: -sf1[e]):
        c = counts.get(etype, {})
        n_a = c.get("total_a", 0)
        n_b = c.get("total_b", 0)
        inter = c.get("intersect", 0)
        print(
            f"  {etype:<25} F1 = {sf1[etype]:.4f}"
            f"  (A={n_a}, B={n_b}, inter={inter})"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IAA for SIREVA-SUS NER.")
    parser.add_argument("--ann1", type=Path, default=ANN1_CSV, help="Annotator 1 CSV")
    parser.add_argument("--ann2", type=Path, default=ANN2_CSV, help="Annotator 2 CSV")
    parser.add_argument("--output", type=Path, default=IAA_REPORT, help="Output JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    result = compute_iaa(ann1_csv=args.ann1, ann2_csv=args.ann2, report_path=args.output)
    _print_report(result)
