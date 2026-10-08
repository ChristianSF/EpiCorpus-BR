"""Shared utility functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, one dict per line.

    Args:
        path: Path to the JSONL file.

    Yields:
        Parsed JSON objects.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a JSONL file.

    Args:
        records: List of serializable dicts.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_sentences(extracted_dir: Path, glob: str = "*.jsonl") -> list[dict[str, Any]]:
    """Load all sentence records from a directory of JSONL files.

    Args:
        extracted_dir: Directory containing per-report JSONL files.
        glob: Glob pattern to select files.

    Returns:
        Combined list of sentence dicts, in file-sorted order.
    """
    sentences: list[dict[str, Any]] = []
    for jsonl_path in sorted(extracted_dir.glob(glob)):
        sentences.extend(read_jsonl(jsonl_path))
    return sentences
