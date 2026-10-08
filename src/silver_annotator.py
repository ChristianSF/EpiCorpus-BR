"""
Silver annotation via OpenAI GPT API.

Annotates sentences from SIREVA-SUS reports with NER tags using
gpt-4o-mini in a few-shot setting. Output: one JSONL per report in
data/silver/, each record containing tokens, bio_tags, and spans.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from .config import (
    DATA_EXTRACTED,
    DATA_EXTRACTED_TABLES,
    DATA_SILVER,
    DATA_SILVER_TABLES,
    DEFAULT_LLM_MODEL,
    ENTITY_TYPES,
    OPENAI_API_KEY,
    VALID_ENTITY_NAMES,
)
from .utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Split text into whitespace tokens with character offsets.

    Returns:
        (tokens, offsets) where each offset is a (start, end) char index.
    """
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


# ---------------------------------------------------------------------------
# BIO conversion
# ---------------------------------------------------------------------------

def spans_to_bio(
    text: str, spans: list[dict[str, str]]
) -> tuple[list[str], list[str]]:
    """Convert entity spans to BIO-tagged token lists.

    Args:
        text: Original sentence string.
        spans: List of {"texto": "<exact span>", "tipo": "<CATEGORY>"} dicts
               as returned by the LLM.

    Returns:
        (tokens, bio_tags) — two lists of equal length.
    """
    tokens, offsets = tokenize(text)
    bio_tags = ["O"] * len(tokens)

    for span in spans:
        span_text: str = span.get("texto", "")
        span_type: str = span.get("tipo", "")
        if not span_text or span_type not in VALID_ENTITY_NAMES:
            continue

        # Locate span in the sentence (case-sensitive, then case-insensitive).
        idx = text.find(span_text)
        if idx == -1:
            idx = text.lower().find(span_text.lower())
        if idx == -1:
            logger.debug("Span not found in text: %r", span_text)
            continue

        span_start = idx
        span_end = idx + len(span_text)

        # Find tokens that overlap with the span character range.
        matching = [
            i
            for i, (ts, te) in enumerate(offsets)
            if ts < span_end and te > span_start
        ]
        if not matching:
            continue

        bio_tags[matching[0]] = f"B-{span_type}"
        for i in matching[1:]:
            bio_tags[i] = f"I-{span_type}"

    return tokens, bio_tags


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_entity_descriptions() -> str:
    lines = []
    for et in ENTITY_TYPES:
        exs = ", ".join(f'"{e}"' for e in et["examples"][:3])
        lines.append(f"- {et['name']}: {et['description']} Ex: {exs}")
    return "\n".join(lines)


_SYSTEM_PROMPT = """\
Você é um anotador especialista em reconhecimento de entidades nomeadas (NER) \
para textos epidemiológicos em português brasileiro.

Sua tarefa: identificar entidades nomeadas em sentenças de relatórios \
epidemiológicos do SIREVA-SUS (vigilância de meningite e pneumonia bacteriana).

Categorias disponíveis:
{entity_descriptions}

Regras:
- Retorne SOMENTE um JSON com a chave "entidades": lista de objetos \
{{"texto": "<trecho exato>", "tipo": "<CATEGORIA>"}}.
- Use apenas as categorias listadas acima.
- O campo "texto" deve ser o trecho EXATO como aparece na sentença.
- Não anote valores numéricos isolados em METRICA_EPI — anote o conceito \
(ex: "taxa de incidência"), não o número.
- Se não houver entidades, retorne {{"entidades": []}}.
"""

_SYSTEM_PROMPT_TABLES = """\
Você é um anotador especialista em reconhecimento de entidades nomeadas (NER) \
para linhas de tabelas de relatórios epidemiológicos em português brasileiro.

Cada sentença representa uma linha de tabela no formato:
"[legenda da tabela] | [coluna]: [valor] | [coluna]: [valor] | ..."

Categorias disponíveis:
{entity_descriptions}

Regras:
- Retorne SOMENTE um JSON com a chave "entidades": lista de objetos \
{{"texto": "<trecho exato>", "tipo": "<CATEGORIA>"}}.
- Use apenas as categorias listadas acima.
- O campo "texto" deve ser o trecho EXATO como aparece na sentença.
- METRICA_EPI: anote cada valor numérico individual (contagens e percentuais) \
que representa uma medida epidemiológica. Ex: "42", "97,4", "0,0".
- METODO_LAB: anote antimicrobianos (Ceftriaxona, Penicilina, Rifampicina, \
Cloranfenicol, etc.) e fontes de isolamento (Hemocultura, LCR, Líquido Sinovial).
- FAIXA_ETARIA: anote grupos etários como "< 12 meses", "5-14 anos", \
"≥ 60 anos", "Subtotal (1)", "Subtotal (2)".
- Não anote rótulos estruturais como "n:", "%:", "Total:", "Diagnóstico:".
- Se não houver entidades, retorne {{"entidades": []}}.
"""

_FEW_SHOT: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            'Sentença: "A taxa de incidência de meningite por Neisseria meningitidis'
            ' sorogrupo B foi maior na região Sudeste em 2022."'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entidades": [
                    {"texto": "taxa de incidência", "tipo": "METRICA_EPI"},
                    {"texto": "meningite", "tipo": "MANIFESTACAO_CLINICA"},
                    {"texto": "Neisseria meningitidis", "tipo": "PATOGENO"},
                    {"texto": "sorogrupo B", "tipo": "SOROTIPO"},
                    {"texto": "região Sudeste", "tipo": "LOCAL"},
                    {"texto": "2022", "tipo": "PERIODO_TEMPORAL"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Sentença: "Crianças menores de 5 anos apresentaram maior letalidade'
            " por pneumonia pneumocócica sorotipo 19A, diagnosticada por cultura"
            ' e PCR em tempo real."'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entidades": [
                    {"texto": "Crianças menores de 5 anos", "tipo": "FAIXA_ETARIA"},
                    {"texto": "letalidade", "tipo": "METRICA_EPI"},
                    {"texto": "pneumonia", "tipo": "MANIFESTACAO_CLINICA"},
                    {"texto": "pneumocócica", "tipo": "PATOGENO"},
                    {"texto": "sorotipo 19A", "tipo": "SOROTIPO"},
                    {"texto": "cultura", "tipo": "METODO_LAB"},
                    {"texto": "PCR em tempo real", "tipo": "METODO_LAB"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Sentença: "O número de casos de doença invasiva por Haemophilus influenzae'
            ' tipo b em menores de 12 meses aumentou no estado de São Paulo em 2019."'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entidades": [
                    {"texto": "número de casos", "tipo": "METRICA_EPI"},
                    {"texto": "doença invasiva", "tipo": "MANIFESTACAO_CLINICA"},
                    {"texto": "Haemophilus influenzae", "tipo": "PATOGENO"},
                    {"texto": "tipo b", "tipo": "SOROTIPO"},
                    {"texto": "menores de 12 meses", "tipo": "FAIXA_ETARIA"},
                    {"texto": "São Paulo", "tipo": "LOCAL"},
                    {"texto": "2019", "tipo": "PERIODO_TEMPORAL"},
                ]
            },
            ensure_ascii=False,
        ),
    },
]


# Few-shot examples specifically for synthetic table sentences.
# Format: "[caption] | col: val | col: val | ..."
_FEW_SHOT_TABLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            'Sentença: "Tabela 1. Número de cepas por grupo etário e sexo'
            " | Grupo etário: < 12 meses | Masculino n: 9 | Masculino %: 39,1"
            ' | Feminino n: 7 | Feminino %: 30,4 | Total n: 23 | Total %: 17,7"'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entidades": [
                    {"texto": "< 12 meses", "tipo": "FAIXA_ETARIA"},
                    {"texto": "9", "tipo": "METRICA_EPI"},
                    {"texto": "39,1", "tipo": "METRICA_EPI"},
                    {"texto": "7", "tipo": "METRICA_EPI"},
                    {"texto": "30,4", "tipo": "METRICA_EPI"},
                    {"texto": "23", "tipo": "METRICA_EPI"},
                    {"texto": "17,7", "tipo": "METRICA_EPI"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Sentença: "Tabela 2. Número de cepas por diagnóstico e grupo etário'
            " | Grupo etário: 24-59 meses | Diagnóstico Pneumonia n: 2"
            ' | Diagnóstico Meningite n: 8 | Diagnóstico Sepse/Bacteriemia n: 7 | Total n: 17"'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entidades": [
                    {"texto": "24-59 meses", "tipo": "FAIXA_ETARIA"},
                    {"texto": "Pneumonia", "tipo": "MANIFESTACAO_CLINICA"},
                    {"texto": "2", "tipo": "METRICA_EPI"},
                    {"texto": "Meningite", "tipo": "MANIFESTACAO_CLINICA"},
                    {"texto": "8", "tipo": "METRICA_EPI"},
                    {"texto": "Sepse/Bacteriemia", "tipo": "MANIFESTACAO_CLINICA"},
                    {"texto": "7", "tipo": "METRICA_EPI"},
                    {"texto": "17", "tipo": "METRICA_EPI"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Sentença: "Tabela 5. Suscetibilidade a antimicrobianos por grupo etário'
            " | Grupo etário: 5-14 anos | Ceftriaxona meningites (CIM) Suscetível n: 37"
            ' | %: 94,9 | Resistente n: 2 | %: 5,1 | Total n: 39"'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entidades": [
                    {"texto": "5-14 anos", "tipo": "FAIXA_ETARIA"},
                    {"texto": "Ceftriaxona", "tipo": "METODO_LAB"},
                    {"texto": "meningites", "tipo": "MANIFESTACAO_CLINICA"},
                    {"texto": "37", "tipo": "METRICA_EPI"},
                    {"texto": "94,9", "tipo": "METRICA_EPI"},
                    {"texto": "2", "tipo": "METRICA_EPI"},
                    {"texto": "5,1", "tipo": "METRICA_EPI"},
                    {"texto": "39", "tipo": "METRICA_EPI"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Sentença: "Tabela 3. Número de cepas por fonte de isolamento'
            " | Grupo etário: Subtotal (2) | Hemocultura n: 16 | %: 42,1"
            ' | LCR n: 22 | %: 57,9 | Líquido Sinovial n: 0 | %: 0,0 | Total n: 38"'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entidades": [
                    {"texto": "Subtotal (2)", "tipo": "FAIXA_ETARIA"},
                    {"texto": "Hemocultura", "tipo": "METODO_LAB"},
                    {"texto": "16", "tipo": "METRICA_EPI"},
                    {"texto": "42,1", "tipo": "METRICA_EPI"},
                    {"texto": "LCR", "tipo": "METODO_LAB"},
                    {"texto": "22", "tipo": "METRICA_EPI"},
                    {"texto": "57,9", "tipo": "METRICA_EPI"},
                    {"texto": "Líquido Sinovial", "tipo": "METODO_LAB"},
                    {"texto": "0", "tipo": "METRICA_EPI"},
                    {"texto": "0,0", "tipo": "METRICA_EPI"},
                    {"texto": "38", "tipo": "METRICA_EPI"},
                ]
            },
            ensure_ascii=False,
        ),
    },
]


# ---------------------------------------------------------------------------
# Annotator class
# ---------------------------------------------------------------------------


class SilverAnnotator:
    """Annotates SIREVA-SUS sentences with NER tags via gpt-4o-mini."""

    def __init__(self, model: str = DEFAULT_LLM_MODEL, rpm_limit: int = 500) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self._delay = 60.0 / rpm_limit
        entity_descriptions = _build_entity_descriptions()
        self._system = _SYSTEM_PROMPT.format(entity_descriptions=entity_descriptions)
        self._system_tables = _SYSTEM_PROMPT_TABLES.format(entity_descriptions=entity_descriptions)

    def annotate_sentence(self, sentence: dict[str, Any]) -> dict[str, Any]:
        """Annotate a single sentence record.

        Args:
            sentence: Dict with at least {"sentenca_id": ..., "texto": ...}.

        Returns:
            Original dict enriched with "tokens", "bio_tags", and "spans".
        """
        text: str = sentence["texto"]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system},
            *_FEW_SHOT,
            {"role": "user", "content": f'Sentença: "{text}"'},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=512,
        )

        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            spans: list[dict[str, str]] = parsed.get("entidades", [])
        except json.JSONDecodeError:
            logger.warning("JSON parse error for %s", sentence["sentenca_id"])
            spans = []

        tokens, bio_tags = spans_to_bio(text, spans)

        return {**sentence, "tokens": tokens, "bio_tags": bio_tags, "spans": spans}

    def annotate_report(self, report_name: str) -> None:
        """Annotate all sentences in one report and write to data/silver/.

        Args:
            report_name: Stem of the JSONL file, e.g. "sireva_2013".
        """
        src = DATA_EXTRACTED / f"{report_name}.jsonl"
        dst = DATA_SILVER / f"{report_name}.jsonl"

        sentences = list(read_jsonl(src))
        annotated: list[dict[str, Any]] = []

        for i, sent in enumerate(sentences):
            try:
                result = self.annotate_sentence(sent)
            except Exception as exc:
                logger.error("Error on %s: %s", sent.get("sentenca_id"), exc)
                tokens, _ = tokenize(sent.get("texto", ""))
                result = {
                    **sent,
                    "tokens": tokens,
                    "bio_tags": ["O"] * len(tokens),
                    "spans": [],
                }
            annotated.append(result)
            time.sleep(self._delay)

            if (i + 1) % 50 == 0:
                logger.info("  %d/%d sentences annotated", i + 1, len(sentences))

        write_jsonl(annotated, dst)
        logger.info("Saved %d records -> %s", len(annotated), dst)

    def annotate_all(self) -> None:
        """Annotate all reports found in data/extracted/."""
        reports = sorted(p.stem for p in DATA_EXTRACTED.glob("*.jsonl"))
        logger.info("Found %d reports to annotate.", len(reports))
        for report in reports:
            logger.info("Annotating %s ...", report)
            self.annotate_report(report)

    def annotate_table_sentence(self, sentence: dict[str, Any]) -> dict[str, Any]:
        """Annotate a single table row record using table-specific few-shot."""
        text: str = sentence["texto"]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_tables},
            *_FEW_SHOT_TABLES,
            {"role": "user", "content": f'Sentença: "{text}"'},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=512,
        )

        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            spans: list[dict[str, str]] = parsed.get("entidades", [])
        except json.JSONDecodeError:
            logger.warning("JSON parse error for %s", sentence["sentenca_id"])
            spans = []

        tokens, bio_tags = spans_to_bio(text, spans)
        return {**sentence, "tokens": tokens, "bio_tags": bio_tags, "spans": spans}

    def annotate_table_report(self, report_name: str, force: bool = False) -> None:
        """Annotate all table rows for one report and write to data/silver_tables/.

        Skips the report if the output file already exists, unless force=True.
        """
        src = DATA_EXTRACTED_TABLES / f"{report_name}.jsonl"
        dst = DATA_SILVER_TABLES / f"{report_name}.jsonl"

        if dst.exists() and not force:
            logger.info("Skipping %s (already annotated)", report_name)
            return

        rows = list(read_jsonl(src))
        annotated: list[dict[str, Any]] = []

        for i, row in enumerate(rows):
            try:
                result = self.annotate_table_sentence(row)
            except Exception as exc:
                logger.error("Error on %s: %s", row.get("sentenca_id"), exc)
                tokens, _ = tokenize(row.get("texto", ""))
                result = {
                    **row,
                    "tokens": tokens,
                    "bio_tags": ["O"] * len(tokens),
                    "spans": [],
                }
            annotated.append(result)
            time.sleep(self._delay)

            if (i + 1) % 50 == 0:
                logger.info("  %d/%d rows annotated", i + 1, len(rows))

        write_jsonl(annotated, dst)
        logger.info("Saved %d records -> %s", len(annotated), dst)

    def annotate_all_tables(self, force: bool = False) -> None:
        """Annotate all table JSONL files found in data/extracted_tables/."""
        reports = sorted(p.stem for p in DATA_EXTRACTED_TABLES.glob("*.jsonl"))
        logger.info("Found %d table files to annotate.", len(reports))
        for report in reports:
            logger.info("Annotating tables for %s ...", report)
            self.annotate_table_report(report, force=force)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Silver annotator for SIREVA-SUS corpus.")
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Annotate table rows (data/extracted_tables/ -> data/silver_tables/) "
             "instead of narrative text.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate even if output file already exists.",
    )
    args = parser.parse_args()

    annotator = SilverAnnotator()
    if args.tables:
        annotator.annotate_all_tables(force=args.force)
    else:
        annotator.annotate_all()


if __name__ == "__main__":
    main()
