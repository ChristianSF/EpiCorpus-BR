"""
BERTimbau NER baseline: silver-supervised fine-tuning evaluated on gold.

Fine-tunes a token-classification model (``neuralmind/bert-base-portuguese-cased``)
on the silver-annotated corpus (narrative text + table rows), **excluding** the
280 gold units to avoid test leakage, and evaluates on the manually reviewed gold
standard with seqeval (strict matching, IOB2) -- the exact protocol used for the
silver-vs-gold comparison in :mod:`src.evaluation`.

This provides a supervised downstream baseline trained under distant (silver)
supervision and tested on gold, answering whether the corpus is usable to train
a dedicated Portuguese NER model rather than only measuring the LLM annotator.

Usage (CLI)::

    python -m src.ner_baseline                     # combined gold test (default)
    python -m src.ner_baseline --split text        # gold_sample.jsonl only
    python -m src.ner_baseline --split tables       # gold_tables.jsonl only
    python -m src.ner_baseline --epochs 4 --seed 42

Outputs (per split, under ``data/evaluation/``):
    - ``results_bertimbau_<split>.json`` -- per-category P/R/F1 (+ averages).
    - ``results_bertimbau_<split>.csv``  -- same, CSV for LaTeX table generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import (
    BIO_LABELS,
    DATA_GOLD,
    DATA_SILVER,
    DATA_SILVER_TABLES,
    ENTITY_TYPES,
    PROJECT_ROOT,
)
from src.utils import read_jsonl

logger = logging.getLogger(__name__)

EVAL_DIR: Path = PROJECT_ROOT / "data" / "evaluation"

MODEL_NAME: str = "neuralmind/bert-base-portuguese-cased"

# Label <-> id maps derived from the project-wide BIO vocabulary.
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _gold_ids() -> set[str]:
    """Return the set of sentenca_ids that belong to the gold standard.

    These are excluded from the silver training pool to prevent test leakage.
    """
    ids: set[str] = set()
    for fname in ("gold_sample.jsonl", "gold_tables.jsonl"):
        for record in read_jsonl(DATA_GOLD / fname):
            ids.add(record["sentenca_id"])
    return ids


def _load_records(
    directories: list[Path],
    keep_ids: set[str] | None = None,
    drop_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect (tokens, bio_tags) records from directories of JSONL files.

    Args:
        directories: Silver directories to read (text and/or tables).
        keep_ids: If given, keep only records whose sentenca_id is in this set.
        drop_ids: If given, drop records whose sentenca_id is in this set.

    Returns:
        List of records having non-empty ``tokens`` and matching ``bio_tags``.
    """
    records: list[dict[str, Any]] = []
    for directory in directories:
        for jsonl_path in sorted(directory.glob("*.jsonl")):
            for rec in read_jsonl(jsonl_path):
                sid = rec.get("sentenca_id")
                tokens = rec.get("tokens") or []
                tags = rec.get("bio_tags") or []
                if not tokens or len(tokens) != len(tags):
                    continue
                if keep_ids is not None and sid not in keep_ids:
                    continue
                if drop_ids is not None and sid in drop_ids:
                    continue
                records.append({"tokens": tokens, "bio_tags": tags})
    return records


def _load_gold_records(fnames: list[str]) -> list[dict[str, Any]]:
    """Load gold records (tokens + bio_tags) from one or more gold files."""
    records: list[dict[str, Any]] = []
    for fname in fnames:
        for rec in read_jsonl(DATA_GOLD / fname):
            tokens = rec.get("tokens") or []
            tags = rec.get("bio_tags") or []
            if tokens and len(tokens) == len(tags):
                records.append({"tokens": tokens, "bio_tags": tags})
    return records


# ---------------------------------------------------------------------------
# Dataset / tokenisation
# ---------------------------------------------------------------------------


class NerDataset(Dataset):
    """Token-classification dataset with sub-word label alignment.

    Each word receives its label on the first sub-word token; remaining
    sub-words and special tokens are masked with ``-100`` so they do not
    contribute to the loss.  Word-level predictions are recovered at eval
    time from the first sub-word of each word.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        tokens: list[str] = rec["tokens"]
        tags: list[str] = rec["bio_tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoding.word_ids()

        labels: list[int] = []
        prev_word: int | None = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word:
                labels.append(LABEL2ID.get(tags[wid], LABEL2ID["O"]))
            else:
                labels.append(-100)  # non-first sub-word
            prev_word = wid

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


def _collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Dynamically pad a batch to its longest sequence (speeds up CPU training)."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention, labels = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def _predict_word_level(
    model: Any,
    dataset: NerDataset,
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
) -> tuple[list[list[str]], list[list[str]]]:
    """Run inference and recover word-level (true, pred) BIO sequences.

    Only the first sub-word of each word (the positions whose gold label is
    not ``-100``) is kept, yielding one tag per original word -- matching the
    word-level sequences seqeval expects.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate(b, pad_token_id),
    )
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds = logits.argmax(dim=-1).cpu()
            for i in range(labels.size(0)):
                seq_true, seq_pred = [], []
                for j in range(labels.size(1)):
                    lab = labels[i, j].item()
                    if lab == -100:
                        continue
                    seq_true.append(ID2LABEL[lab])
                    seq_pred.append(ID2LABEL[preds[i, j].item()])
                if seq_true:
                    y_true.append(seq_true)
                    y_pred.append(seq_pred)
    return y_true, y_pred


def _scores_from_report(
    report_dict: dict[str, Any],
    entity_names: list[str],
) -> dict[str, dict[str, float]]:
    """Reshape a seqeval report dict into the project results schema."""
    results: dict[str, dict[str, float]] = {}
    for name in entity_names:
        row = report_dict.get(name)
        if row:
            results[name] = {
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1": round(row["f1-score"], 4),
                "support": int(row["support"]),
            }
        else:
            results[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    for avg_key in ("macro avg", "micro avg", "weighted avg"):
        if avg_key in report_dict:
            row = report_dict[avg_key]
            results[avg_key] = {
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1": round(row["f1-score"], 4),
                "support": int(row.get("support", 0)),
            }
    return results


def _write_outputs(
    results: dict[str, dict[str, float]],
    entity_names: list[str],
    split: str,
) -> None:
    """Persist results as JSON and CSV, mirroring :mod:`src.evaluation`."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"results_bertimbau_{split}"

    with (EVAL_DIR / f"{stem}.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    rows = []
    for name in entity_names + ["micro avg", "macro avg", "weighted avg"]:
        row = results.get(name)
        if row:
            rows.append(
                {
                    "entity": name,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "support": row.get("support", 0),
                }
            )
    with (EVAL_DIR / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["entity", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results saved to %s.{json,csv}", EVAL_DIR / stem)


def _print_table(results: dict[str, dict[str, float]], entity_names: list[str]) -> None:
    header = f"{'Entidade':<25}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Sup':>6}"
    print("\n" + "=" * len(header))
    print("  BERTimbau (silver-trained) vs Gold  [seqeval strict, IOB2]")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in entity_names:
        r = results.get(name, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
        print(f"{name:<25}  {r['precision']:>6.4f}  {r['recall']:>6.4f}  {r['f1']:>6.4f}  {r.get('support', 0):>6}")
    print("-" * len(header))
    for avg_key in ("micro avg", "macro avg"):
        if avg_key in results:
            r = results[avg_key]
            print(f"{avg_key:<25}  {r['precision']:>6.4f}  {r['recall']:>6.4f}  {r['f1']:>6.4f}  {r.get('support', 0):>6}")
    print("=" * len(header) + "\n")


def _evaluate_split(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pad_id: int,
    batch_size: int,
    max_length: int,
    split: str,
    entity_names: list[str],
) -> dict[str, dict[str, float]]:
    """Evaluate the trained model on one gold split and persist the results."""
    gold_files = {
        "text": ["gold_sample.jsonl"],
        "tables": ["gold_tables.jsonl"],
        "combined": ["gold_sample.jsonl", "gold_tables.jsonl"],
    }[split]
    test_ds = NerDataset(_load_gold_records(gold_files), tokenizer, max_length)

    y_true, y_pred = _predict_word_level(model, test_ds, device, pad_id, batch_size)
    report_dict: dict[str, Any] = classification_report(
        y_true, y_pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0,
    )
    logger.info("[%s]\n%s", split, classification_report(
        y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0))

    results = _scores_from_report(report_dict, entity_names)
    _write_outputs(results, entity_names, split)
    print(f"\n### Gold split: {split.upper()} ###")
    _print_table(results, entity_names)
    return results


def run(
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 192,
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """Train BERTimbau once on silver (minus gold) and evaluate on all gold splits.

    A single model is fine-tuned on the silver corpus with the 280 gold units
    removed, then evaluated on the text-only, tables-only and combined gold
    sets -- mirroring the three-column structure of the silver-vs-gold results
    table in the paper.

    Args:
        epochs: Number of fine-tuning epochs.
        batch_size: Mini-batch size.
        lr: Peak learning rate for AdamW.
        max_length: Max sub-word sequence length (truncation).
        seed: RNG seed for reproducibility.

    Returns:
        Mapping ``split -> results dict`` for ``text``, ``tables``, ``combined``.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entity_names = [e["name"] for e in ENTITY_TYPES]

    # --- Data ---------------------------------------------------------------
    gold_ids = _gold_ids()
    train_records = _load_records([DATA_SILVER, DATA_SILVER_TABLES], drop_ids=gold_ids)
    logger.info("Train (silver minus gold): %d units | device=%s", len(train_records), device)

    # --- Model / tokenizer --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_ds = NerDataset(train_records, tokenizer, max_length)
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: _collate(b, pad_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # --- Training loop ------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += out.loss.item()
            if step % 50 == 0:
                logger.info("epoch %d  step %d/%d  loss %.4f",
                            epoch, step, len(train_loader), running / step)
        logger.info("epoch %d done  avg_loss %.4f", epoch, running / len(train_loader))

    # --- Evaluation on all gold splits --------------------------------------
    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("text", "tables", "combined"):
        all_results[split] = _evaluate_split(
            model, tokenizer, device, pad_id, batch_size, max_length, split, entity_names,
        )
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="BERTimbau silver-trained NER baseline.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )
