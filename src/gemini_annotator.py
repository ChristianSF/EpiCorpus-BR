"""
Gemini LLM baseline: second-annotator comparison on the gold standard.

Runs the *same* few-shot NER prompt used for the GPT-4o-mini silver
annotation (see :mod:`src.silver_annotator`) with a Gemini Flash-tier
model, but only over the 280 gold units, and evaluates the result against
the gold standard with the same seqeval strict-IOB2 protocol used in
:mod:`src.evaluation`.  This provides a second-LLM comparison (addressing
the reviewers' request to compare different LLMs) without re-annotating
the whole corpus.

The Gemini REST API is called via the standard library (``urllib``); no
extra SDK is installed.  The key is read from ``GEMINI_API_KEY`` (or
``GOOGLE_API_KEY``) in the environment / ``.env``.

Usage (CLI)::

    python -m src.gemini_annotator                      # all gold splits
    python -m src.gemini_annotator --model gemini-2.0-flash  # override model
    python -m src.gemini_annotator --rpm 15             # free-tier pacing

Outputs (under ``data/evaluation/``):
    - ``results_gemini_<split>.json`` / ``.csv`` -- per-category P/R/F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from src.config import DATA_GOLD, ENTITY_TYPES, GEMINI_API_KEY, PROJECT_ROOT
from src.silver_annotator import (
    _FEW_SHOT,
    _FEW_SHOT_TABLES,
    _SYSTEM_PROMPT,
    _SYSTEM_PROMPT_TABLES,
    _build_entity_descriptions,
    spans_to_bio,
)
from src.utils import read_jsonl

logger = logging.getLogger(__name__)

EVAL_DIR: Path = PROJECT_ROOT / "data" / "evaluation"

DEFAULT_MODEL: str = "gemini-2.5-flash"
_API_ROOT: str = "https://generativelanguage.googleapis.com/v1beta/models"


# ---------------------------------------------------------------------------
# Prompt conversion (OpenAI chat few-shot -> Gemini contents)
# ---------------------------------------------------------------------------


def _few_shot_to_contents(few_shot: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style role/content turns to Gemini ``contents`` turns.

    OpenAI ``assistant`` maps to Gemini ``model``; the text goes into a
    single ``parts`` entry.
    """
    contents: list[dict[str, Any]] = []
    for turn in few_shot:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    return contents


# ---------------------------------------------------------------------------
# Gemini REST client
# ---------------------------------------------------------------------------


class GeminiAnnotator:
    """Annotates gold sentences via the Gemini REST API (same prompt as silver)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        rpm: int = 15,
        max_retries: int = 4,
    ) -> None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to your .env "
                "(GEMINI_API_KEY=... or GOOGLE_API_KEY=...)."
            )
        self.model = model
        self._delay = 60.0 / max(rpm, 1)
        self._max_retries = max_retries
        entity_descriptions = _build_entity_descriptions()
        self._system_text = _SYSTEM_PROMPT.format(entity_descriptions=entity_descriptions)
        self._system_tables = _SYSTEM_PROMPT_TABLES.format(
            entity_descriptions=entity_descriptions
        )
        self._fewshot_text = _few_shot_to_contents(_FEW_SHOT)
        self._fewshot_tables = _few_shot_to_contents(_FEW_SHOT_TABLES)

    # -- HTTP ---------------------------------------------------------------

    def _call(self, system_instruction: str, contents: list[dict[str, Any]]) -> str:
        """POST one generateContent request and return the raw text part."""
        url = f"{_API_ROOT}/{self.model}:generateContent?key={GEMINI_API_KEY}"
        body = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "maxOutputTokens": 1024,
            },
        }
        data = json.dumps(body).encode("utf-8")

        for attempt in range(1, self._max_retries + 1):
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return _extract_text(payload)
            except urllib.error.HTTPError as exc:
                # 429 (rate limit) / 5xx -> back off and retry.
                if exc.code in (429, 500, 503) and attempt < self._max_retries:
                    wait = self._delay * (2 ** attempt)
                    logger.warning("HTTP %d; backing off %.1fs (try %d)", exc.code, wait, attempt)
                    time.sleep(wait)
                    continue
                logger.error("HTTP %d: %s", exc.code, exc.read().decode("utf-8", "ignore")[:200])
                return "{}"
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self._max_retries:
                    time.sleep(self._delay * (2 ** attempt))
                    continue
                logger.error("Network error: %s", exc)
                return "{}"
        return "{}"

    # -- Annotation ---------------------------------------------------------

    def annotate(self, sentence: dict[str, Any], is_table: bool) -> dict[str, Any]:
        """Annotate one gold record; returns it enriched with tokens/bio_tags/spans."""
        text: str = sentence["texto"]
        system = self._system_tables if is_table else self._system_text
        few_shot = self._fewshot_tables if is_table else self._fewshot_text
        contents = [*few_shot, {"role": "user", "parts": [{"text": f'Sentença: "{text}"'}]}]

        raw = self._call(system, contents)
        try:
            spans = json.loads(raw).get("entidades", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("JSON parse error for %s", sentence.get("sentenca_id"))
            spans = []

        tokens, bio_tags = spans_to_bio(text, spans)
        return {**sentence, "tokens": tokens, "bio_tags": bio_tags, "spans": spans}

    def annotate_file(self, gold_path: Path, is_table: bool) -> dict[str, list[str]]:
        """Annotate every record in a gold file; returns sentenca_id -> pred bio_tags."""
        preds: dict[str, list[str]] = {}
        records = list(read_jsonl(gold_path))
        for i, rec in enumerate(records, 1):
            result = self.annotate(rec, is_table)
            preds[rec["sentenca_id"]] = result["bio_tags"]
            time.sleep(self._delay)
            if i % 25 == 0:
                logger.info("  %d/%d annotated (%s)", i, len(records), gold_path.name)
        return preds


def _extract_text(payload: dict[str, Any]) -> str:
    """Pull the text out of a Gemini generateContent response payload."""
    try:
        parts = payload["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts) or "{}"
    except (KeyError, IndexError, TypeError):
        return "{}"


# ---------------------------------------------------------------------------
# Evaluation (mirrors src.evaluation / src.ner_baseline)
# ---------------------------------------------------------------------------


def _load_gold_tags(gold_path: Path) -> dict[str, list[str]]:
    """sentenca_id -> gold bio_tags."""
    out: dict[str, list[str]] = {}
    for rec in read_jsonl(gold_path):
        tags = rec.get("bio_tags") or []
        if tags:
            out[rec["sentenca_id"]] = [str(t) for t in tags]
    return out


def _score(gold: dict[str, list[str]], pred: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Strict IOB2 seqeval over the paired sequences, in the project schema."""
    entity_names = [e["name"] for e in ENTITY_TYPES]
    g_seqs, p_seqs = [], []
    for sid, g in sorted(gold.items()):
        p = pred.get(sid)
        if p is None:
            continue
        n = min(len(g), len(p))
        g_seqs.append(g[:n])
        p_seqs.append(p[:n])

    report: dict[str, Any] = classification_report(
        g_seqs, p_seqs, mode="strict", scheme=IOB2, output_dict=True, zero_division=0,
    )
    logger.info("\n%s", classification_report(
        g_seqs, p_seqs, mode="strict", scheme=IOB2, zero_division=0))

    results: dict[str, dict[str, float]] = {}
    for name in entity_names:
        row = report.get(name)
        results[name] = (
            {"precision": round(row["precision"], 4), "recall": round(row["recall"], 4),
             "f1": round(row["f1-score"], 4), "support": int(row["support"])}
            if row else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        )
    for avg in ("micro avg", "macro avg", "weighted avg"):
        if avg in report:
            row = report[avg]
            results[avg] = {"precision": round(row["precision"], 4),
                            "recall": round(row["recall"], 4),
                            "f1": round(row["f1-score"], 4),
                            "support": int(row.get("support", 0))}
    return results


def _write(results: dict[str, dict[str, float]], split: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"results_gemini_{split}"
    with (EVAL_DIR / f"{stem}.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    entity_names = [e["name"] for e in ENTITY_TYPES]
    with (EVAL_DIR / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["entity", "precision", "recall", "f1", "support"])
        w.writeheader()
        for name in entity_names + ["micro avg", "macro avg", "weighted avg"]:
            r = results.get(name)
            if r:
                w.writerow({"entity": name, **{k: r[k] for k in ("precision", "recall", "f1")},
                            "support": r.get("support", 0)})
    logger.info("Saved %s.{json,csv}", EVAL_DIR / stem)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(model: str = DEFAULT_MODEL, rpm: int = 15) -> dict[str, dict[str, dict[str, float]]]:
    """Annotate the 280 gold units with Gemini and evaluate all three splits."""
    annotator = GeminiAnnotator(model=model, rpm=rpm)
    logger.info("Gemini model: %s", model)

    text_gold = DATA_GOLD / "gold_sample.jsonl"
    table_gold = DATA_GOLD / "gold_tables.jsonl"

    logger.info("Annotating text gold (180) ...")
    pred_text = annotator.annotate_file(text_gold, is_table=False)
    logger.info("Annotating table gold (100) ...")
    pred_tables = annotator.annotate_file(table_gold, is_table=True)

    gold_text = _load_gold_tags(text_gold)
    gold_tables = _load_gold_tags(table_gold)

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for split, gold, pred in (
        ("text", gold_text, pred_text),
        ("tables", gold_tables, pred_tables),
        ("combined", {**gold_text, **gold_tables}, {**pred_text, **pred_tables}),
    ):
        logger.info("=== Gemini vs Gold [%s] ===", split)
        results = _score(gold, pred)
        _write(results, split)
        all_results[split] = results
    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Gemini LLM baseline on the gold standard.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id (Flash tier).")
    parser.add_argument("--rpm", type=int, default=15, help="Requests per minute (free-tier pacing).")
    args = parser.parse_args()
    run(model=args.model, rpm=args.rpm)
