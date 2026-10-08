"""
Extrai e anota (silver) os novos boletins epidemiológicos adicionados ao corpus.

Uso:
    python scripts/process_boletins.py             # extração + anotação
    python scripts/process_boletins.py --extract   # só extração
    python scripts/process_boletins.py --annotate  # só anotação (extração já feita)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Garante que src/ é importável a partir da raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_EXTRACTED, DATA_RAW, DATA_SILVER  # noqa: E402
from src.extraction import extract_and_save  # noqa: E402
from src.silver_annotator import SilverAnnotator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapeamento: slug limpo -> nome do arquivo PDF (relativo a data/)
# ---------------------------------------------------------------------------

BOLETINS: dict[str, str] = {
    "boletim_eletronico_ano08_n17": "Boletim Eletrônico Epidemiológico - Ano 08 nº 17.pdf",
    "boletim_eletronico_ano08_n18": "Boletim Eletrônico Epidemiológico - Ano 08 nº 18.pdf",
    "boletim_eletronico_ano10_n03": "Boletim Eletrônico Epidemiológico - Ano 10 nº 3.pdf",
    "boletim_epidemiologico_nesp":  "Boletim Epidemiológico - Número especial.pdf",
    "boletim_epidemiologico_v47_n29": "Boletim Epidemiológico - Volume 47 nº 29.pdf",
    "boletim_epidemiologico_v50_n03": "Boletim Epidemiológico - Volume 50 nº 03.pdf",
    "boletim_epidemiologico_n25":   "Boletim Epidemiológico nº 25.pdf",
    "boletim_epidemiologico_n38":   "Boletim Epidemiológico nº 38.pdf",
    "informe_meningite_ed01":       "informe Meningite.pdf",
    "informe_meningite_ed02":       "informe-meningite-2a-edicao.pdf",
}


def run_extraction(slugs: list[str]) -> list[str]:
    """Extrai sentenças dos PDFs e salva em data/extracted/. Retorna slugs processados."""
    done: list[str] = []
    for slug, filename in BOLETINS.items():
        if slugs and slug not in slugs:
            continue
        pdf_path = DATA_RAW / filename
        if not pdf_path.exists():
            logger.warning("PDF não encontrado: %s", pdf_path)
            continue
        out_path = DATA_EXTRACTED / f"{slug}.jsonl"
        if out_path.exists():
            logger.info("Já extraído: %s — pulando.", slug)
            done.append(slug)
            continue
        try:
            extract_and_save(pdf_path, output_dir=DATA_EXTRACTED, report_name=slug)
            done.append(slug)
        except Exception as exc:
            logger.error("Falha na extração de %s: %s", slug, exc)
    return done


def run_annotation(slugs: list[str]) -> None:
    """Anota (silver) os slugs extraídos que ainda não têm JSONL em data/silver/."""
    annotator = SilverAnnotator()
    for slug in slugs:
        dst = DATA_SILVER / f"{slug}.jsonl"
        if dst.exists():
            logger.info("Já anotado: %s — pulando.", slug)
            continue
        src = DATA_EXTRACTED / f"{slug}.jsonl"
        if not src.exists():
            logger.warning("Arquivo extraído não encontrado para anotação: %s", src)
            continue
        logger.info("Anotando %s ...", slug)
        annotator.annotate_report(slug)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract",  action="store_true", help="Só extração")
    parser.add_argument("--annotate", action="store_true", help="Só anotação")
    parser.add_argument("slugs", nargs="*", help="Slugs específicos (padrão: todos)")
    args = parser.parse_args()

    only_extract  = args.extract and not args.annotate
    only_annotate = args.annotate and not args.extract
    slugs = args.slugs

    if not only_annotate:
        extracted = run_extraction(slugs)
        logger.info("Extração concluída: %d/%d documentos.", len(extracted), len(BOLETINS))
    else:
        extracted = slugs or list(BOLETINS.keys())

    if not only_extract:
        run_annotation(extracted)
        logger.info("Anotação silver concluída.")


if __name__ == "__main__":
    main()
