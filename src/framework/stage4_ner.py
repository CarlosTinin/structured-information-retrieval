from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from transformers import pipeline

from .io_utils import ensure_parent_dir, load_json_flexible, save_json


def to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


def flatten_to_csv_rows(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in results:
        entities = item.get("entidades", [])
        if not entities:
            rows.append(
                {
                    "id": item["id"],
                    "sentenca": item["sentenca"],
                    "secao": item["label"],
                    "entidade_texto": None,
                    "entidade_label": None,
                    "entidade_start": None,
                    "entidade_end": None,
                    "entidade_score": None,
                }
            )
            continue

        for ent in entities:
            rows.append(
                {
                    "id": item["id"],
                    "sentenca": item["sentenca"],
                    "secao": item["label"],
                    "entidade_texto": ent["text"],
                    "entidade_label": ent["label"],
                    "entidade_start": ent["start"],
                    "entidade_end": ent["end"],
                    "entidade_score": ent["score"],
                }
            )
    return rows


def run_stage4_ner(
    input_json: str,
    output_json: str,
    output_csv: str,
    model_name: str = "dominguesm/legal-bert-ner-base-cased-ptbr",
    sentence_key: str = "sentenca",
    label_key: str = "label",
) -> dict:
    data = load_json_flexible(input_json)
    if not isinstance(data, list):
        raise ValueError("Formato de entrada inválido: esperado lista de registros")

    ner = pipeline("ner", model=model_name, aggregation_strategy="simple", device=-1)

    results = []
    for idx, item in enumerate(data):
        sentenca = str(item.get(sentence_key, ""))
        section = str(item.get(label_key, ""))
        entities_raw = ner(sentenca) if sentenca else []
        entities = [
            {
                "text": str(ent["word"]),
                "label": str(ent["entity_group"]),
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "score": float(round(float(ent["score"]), 4)),
            }
            for ent in entities_raw
        ]
        results.append(
            {
                "id": idx,
                "sentenca": sentenca,
                "label": section,
                "entidades": entities,
                "num_entidades": len(entities),
            }
        )

    ensure_parent_dir(output_json)
    save_json(to_native(results), output_json)

    ensure_parent_dir(output_csv)
    pd.DataFrame(flatten_to_csv_rows(results)).to_csv(output_csv, index=False, encoding="utf-8-sig")

    total = len(results)
    with_entities = sum(1 for x in results if x["num_entidades"] > 0)
    entities_total = sum(x["num_entidades"] for x in results)

    stats = {
        "total_sentencas": total,
        "sentencas_com_entidades": with_entities,
        "total_entidades": entities_total,
        "media_entidades_por_sentenca": float(entities_total / total) if total else 0.0,
    }
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 4 - Extração NER")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-name", default="dominguesm/legal-bert-ner-base-cased-ptbr")
    parser.add_argument("--sentence-key", default="sentenca")
    parser.add_argument("--label-key", default="label")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = run_stage4_ner(
        input_json=args.input_json,
        output_json=args.output_json,
        output_csv=args.output_csv,
        model_name=args.model_name,
        sentence_key=args.sentence_key,
        label_key=args.label_key,
    )
    print("\nResumo etapa 4:")
    for k, v in stats.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
