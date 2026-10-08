"""
Project configuration: entity types, filesystem paths, and environment loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# Gemini key (Google AI Studio); accepts GEMINI_API_KEY or GOOGLE_API_KEY.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# PDFs are at data/ directly (not data/raw/) in this repository.
DATA_RAW: Path = PROJECT_ROOT / "data"
DATA_EXTRACTED: Path = PROJECT_ROOT / "data" / "extracted"
DATA_EXTRACTED_TABLES: Path = PROJECT_ROOT / "data" / "extracted_tables"
DATA_SILVER: Path = PROJECT_ROOT / "data" / "silver"
DATA_SILVER_TABLES: Path = PROJECT_ROOT / "data" / "silver_tables"
DATA_GOLD: Path = PROJECT_ROOT / "data" / "gold"
ANNOTATION_SAMPLES: Path = PROJECT_ROOT / "annotation" / "samples"

for _dir in (
    DATA_EXTRACTED, DATA_EXTRACTED_TABLES,
    DATA_SILVER, DATA_SILVER_TABLES,
    DATA_GOLD, ANNOTATION_SAMPLES,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Entity type definitions
# ---------------------------------------------------------------------------


class EntityType(TypedDict):
    """Schema for a single NER entity class."""

    name: str
    bio_prefix: str
    description: str
    examples: list[str]


ENTITY_TYPES: list[EntityType] = [
    {
        "name": "PATOGENO",
        "bio_prefix": "PATOGENO",
        "description": "Organismo causador de doenca invasiva (bacterias, virus, fungos).",
        "examples": [
            "Streptococcus pneumoniae",
            "Haemophilus influenzae",
            "N. meningitidis",
            "pneumococo",
            "meningococo",
        ],
    },
    {
        "name": "SOROTIPO",
        "bio_prefix": "SOROTIPO",
        "description": (
            "Identificacao serologica especifica do patogeno: sorotipo, sorogrupo ou tipo,"
            " quando o contexto indica classificacao serologica."
        ),
        "examples": [
            "sorotipo 19A",
            "sorogrupo B",
            "tipo b",
            "sorotipo 14",
            "sorogrupo W135",
        ],
    },
    {
        "name": "FAIXA_ETARIA",
        "bio_prefix": "FAIXA_ETARIA",
        "description": "Intervalo etario usado como estratificacao epidemiologica.",
        "examples": [
            "menor que 12 meses",
            "12 a 23 meses",
            "60 anos ou mais",
            "lactentes",
            "criancas menores de 5 anos",
        ],
    },
    {
        "name": "LOCAL",
        "bio_prefix": "LOCAL",
        "description": "Localizacao geografica nomeada (paises, estados, municipios, regioes).",
        "examples": [
            "Brasil",
            "Sao Paulo",
            "regiao Sudeste",
            "Manaus",
            "Regiao Norte",
        ],
    },
    {
        "name": "PERIODO_TEMPORAL",
        "bio_prefix": "PERIODO_TEMPORAL",
        "description": "Delimitacao temporal explicita: anos, semanas epidemiologicas, meses.",
        "examples": [
            "2024",
            "2013 a 2024",
            "janeiro de 2023",
            "semana epidemiologica 35",
            "primeiro semestre de 2022",
        ],
    },
    {
        "name": "METODO_LAB",
        "bio_prefix": "METODO_LAB",
        "description": "Tecnica laboratorial ou diagnostica nomeada.",
        "examples": [
            "cultura",
            "PCR em tempo real",
            "teste de aglutinacao",
            "MALDI-TOF",
            "antibiograma",
        ],
    },
    {
        "name": "METRICA_EPI",
        "bio_prefix": "METRICA_EPI",
        "description": (
            "Medida quantitativa epidemiologica nomeada -- o conceito, nao o valor numerico."
        ),
        "examples": [
            "taxa de incidencia",
            "numero de casos",
            "cobertura vacinal",
            "letalidade",
            "prevalencia",
        ],
    },
    {
        "name": "MANIFESTACAO_CLINICA",
        "bio_prefix": "MANIFESTACAO_CLINICA",
        "description": "Doenca, sindrome ou condicao clinica nomeada.",
        "examples": [
            "meningite",
            "pneumonia",
            "doenca invasiva",
            "sepse",
            "doenca pneumococica invasiva",
        ],
    },
]

# Flat set of valid entity names -- useful for validation.
VALID_ENTITY_NAMES: frozenset[str] = frozenset(e["name"] for e in ENTITY_TYPES)

# BIO label vocabulary (O + B-/I- for each type).
BIO_LABELS: list[str] = ["O"] + [
    prefix + name
    for name in VALID_ENTITY_NAMES
    for prefix in ("B-", "I-")
]

# Default LLM model for silver annotation.
DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
