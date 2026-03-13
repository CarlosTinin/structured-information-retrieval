from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from .io_utils import ensure_parent_dir, read_csv_smart


def normalize_typography(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_legal_text(text: str, min_size: int = 30) -> tuple[str, Counter]:
    counters: Counter = Counter()
    if not isinstance(text, str):
        return "", counters

    original = text
    boilerplates = [
        r"vistos[, ]*etc\.?",
        r"é o relatório",
        r"relatório dispensado",
        r"passo a decidir",
        r"ante o exposto",
        r"publique-se[, ]*registre-se[, ]*intimem-se\.?",
    ]
    for pattern in boilerplates:
        updated, n = re.subn(pattern, " ", text, flags=re.IGNORECASE)
        text = updated
        counters["boilerplate"] += n

    patterns = {
        "cnj": r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}",
        "numeros_extensos": r"\b\d{6,}\b",
        "nomes_caps": r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,}(?:\s[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,})+\b",
        "partes": r"\b(autor|réu|apelante|apelado|impetrante|impetrado)\b",
        "valores": r"r\$ ?\d+[.,\d]*",
    }

    for name, pattern in patterns.items():
        text, n = re.subn(pattern, " ", text, flags=re.IGNORECASE)
        counters[name] += n

    for date_pattern in [r"\d{2}/\d{2}/\d{4}", r"\d{1,2} de [a-zç]+ de \d{4}"]:
        text, n = re.subn(date_pattern, " ", text, flags=re.IGNORECASE)
        counters["datas"] += n

    for legal_pattern in [r"art\.? ?\d+[ºo]?", r"artigo ?\d+", r"§ ?\d+º?", r"inciso [ivxlcdm]+"]:
        text, n = re.subn(legal_pattern, " ", text, flags=re.IGNORECASE)
        counters["referencias_legais"] += n

    text = normalize_typography(text)
    if len(text) < min_size:
        return original, counters

    return text, counters


def run_stage2(
    input_csv: str,
    output_csv: str,
    text_column: str = "Extracted Text",
    label_column: str = "decisao",
    normalized_column: str = "texto_normalizado",
) -> dict:
    df = read_csv_smart(input_csv)
    for col in (text_column, label_column):
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória não encontrada: {col}")

    global_counter: Counter = Counter()
    normalized_texts: list[str] = []

    for raw in tqdm(df[text_column].tolist(), desc="Normalizando"):
        normalized, counters = normalize_legal_text(raw)
        normalized_texts.append(normalized)
        global_counter.update(counters)

    df[normalized_column] = normalized_texts
    export = df[[text_column, normalized_column, label_column]].copy()
    ensure_parent_dir(output_csv)
    export.to_csv(output_csv, index=False)

    avg_before = float(df[text_column].astype(str).str.len().mean())
    avg_after = float(df[normalized_column].astype(str).str.len().mean())
    changed_docs = int((df[text_column].astype(str) != df[normalized_column].astype(str)).sum())

    return {
        "total_docs": int(len(df)),
        "changed_docs": changed_docs,
        "avg_len_before": avg_before,
        "avg_len_after": avg_after,
        "transform_counts": dict(global_counter),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 2 - Pré-processamento para classificadores")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="Extracted Text")
    parser.add_argument("--label-column", default="decisao")
    parser.add_argument("--normalized-column", default="texto_normalizado")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_stage2(
        input_csv=args.input,
        output_csv=args.output,
        text_column=args.text_column,
        label_column=args.label_column,
        normalized_column=args.normalized_column,
    )
    print("\nRelatório da etapa 2:")
    for k, v in report.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
