from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import chardet
import pandas as pd


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def detect_encoding(file_path: str | Path) -> str:
    with open(file_path, "rb") as f:
        raw = f.read(100_000)
    detected = chardet.detect(raw)
    return detected.get("encoding") or "utf-8"


def detect_delimiter(file_path: str | Path, encoding: str) -> str:
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(10_000)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ";" if ";" in sample.splitlines()[0] else ","


def read_csv_smart(file_path: str | Path) -> pd.DataFrame:
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    return pd.read_csv(file_path, encoding=encoding, sep=delimiter)


def save_json(data: Any, file_path: str | Path) -> None:
    ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json_flexible(file_path: str | Path) -> Any:
    """Suporta JSON array, objeto {dados:[...]}, ou JSONL."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "dados" in payload:
            return payload["dados"]
        return payload
    except json.JSONDecodeError:
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        return lines
