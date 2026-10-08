"""Unit tests for src/extraction.py."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.extraction import (
    _is_narrative,
    _is_sep_line,
    _is_table_line,
    _remove_tables,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# _is_table_line
# ---------------------------------------------------------------------------


def test_is_table_line_basic():
    assert _is_table_line("| Coluna A | Coluna B |")


def test_is_table_line_not_table():
    assert not _is_table_line("Texto normal sem pipe.")


def test_is_table_line_too_few_pipes():
    # Single pipe is not a table row.
    assert not _is_table_line("ver | nota")


# ---------------------------------------------------------------------------
# _is_sep_line
# ---------------------------------------------------------------------------


def test_is_sep_line_basic():
    assert _is_sep_line("|---|---|")
    assert _is_sep_line("| :--- | ---: |")


def test_is_sep_line_not_sep():
    assert not _is_sep_line("| conteudo | valor |")


# ---------------------------------------------------------------------------
# _remove_tables
# ---------------------------------------------------------------------------

_TEXT_WITH_TABLE = textwrap.dedent(
    """\
    Introducao ao relatorio epidemiologico.

    Tabela 1: Distribuicao por sorotipo.
    | Sorotipo | Casos |
    |---|---|
    | 19A | 45 |
    | 14 | 30 |

    Em 2024 foram registrados 265 casos de doenca pneumococica invasiva.
    """
)


def test_remove_tables_removes_rows():
    result = _remove_tables(_TEXT_WITH_TABLE, keep_caption=True)
    assert "| 19A |" not in result
    assert "| Sorotipo |" not in result


def test_remove_tables_keeps_caption():
    result = _remove_tables(_TEXT_WITH_TABLE, keep_caption=True)
    assert "Tabela 1" in result


def test_remove_tables_keeps_narrative():
    result = _remove_tables(_TEXT_WITH_TABLE)
    assert "265 casos" in result
    assert "Introducao" in result


def test_remove_tables_without_caption():
    result = _remove_tables(_TEXT_WITH_TABLE, keep_caption=False)
    # Caption may or may not be preserved -- only table rows must be gone.
    assert "| 19A |" not in result


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    text = "O sorotipo 19A foi o mais prevalente. A cobertura vacinal diminuiu em 2023."
    sents = _split_sentences(text)
    assert len(sents) == 2
    assert sents[0].startswith("O sorotipo")
    assert sents[1].startswith("A cobertura")


def test_split_sentences_single():
    text = "Em 2024 foram registrados 265 casos."
    sents = _split_sentences(text)
    assert len(sents) == 1


def test_split_sentences_merges_short_fragment():
    # A very short fragment (< 15 chars) after a boundary should merge into
    # the previous sentence.  We construct a case where the second candidate
    # sentence would be only 8 characters long.
    # Boundary fires at "." before "Ou" (uppercase); "Ou nao." is 7 chars.
    text = "Este relatorio descreve os dados coletados em 2024. Ou nao. Resultados completos abaixo."
    sents = _split_sentences(text)
    # "Ou nao." has 7 chars and must be merged into the previous sentence.
    assert all(len(s) >= 15 for s in sents)


def test_split_sentences_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


# ---------------------------------------------------------------------------
# _is_narrative
# ---------------------------------------------------------------------------


def test_is_narrative_accepts_prose():
    assert _is_narrative("Em 2024 foram registrados 265 casos de doenca invasiva no Brasil.")


def test_is_narrative_rejects_toc_line():
    assert not _is_narrative("Introducao  ....................................................  4")
    assert not _is_narrative("Haemophilus influenzae ..................................  5")


def test_is_narrative_rejects_short():
    assert not _is_narrative("Ok")
    assert not _is_narrative("2024")


def test_is_narrative_rejects_no_alpha():
    assert not _is_narrative("123")
    assert not _is_narrative("---")


# ---------------------------------------------------------------------------
# extract_narrative_text -- integration test (skipped without PDFs)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_extract_narrative_text_returns_records(tmp_path):
    """Smoke test against the first real PDF if available."""
    from src.config import DATA_RAW
    from src.extraction import extract_narrative_text

    pdfs = sorted(DATA_RAW.glob("sireva_*.pdf"))
    if not pdfs:
        pytest.skip("No PDFs found in data/ -- integration test skipped.")

    records = extract_narrative_text(pdfs[0])
    assert isinstance(records, list)
    assert len(records) > 0

    first = records[0]
    assert "relatorio" in first
    assert "pagina" in first
    assert "sentenca_id" in first
    assert "texto" in first
    assert isinstance(first["texto"], str)
    assert len(first["texto"]) >= 15


@pytest.mark.integration
def test_extract_and_save_writes_jsonl(tmp_path):
    from src.config import DATA_RAW
    from src.extraction import extract_and_save

    pdfs = sorted(DATA_RAW.glob("sireva_*.pdf"))
    if not pdfs:
        pytest.skip("No PDFs found in data/ -- integration test skipped.")

    out = extract_and_save(pdfs[0], output_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".jsonl"

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0
    record = json.loads(lines[0])
    assert "sentenca_id" in record
