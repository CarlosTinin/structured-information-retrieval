from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import pipeline

from .io_utils import ensure_parent_dir, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_native(obj):
    """Convert numpy types to native Python types for JSON serialisation."""
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


def _load_resultado_anotacao(path: str) -> list[dict]:
    """Load ``resultado_anotacao.json`` and return the ``resultados`` array.

    Each element is a document dict containing at least ``doc_id`` and
    ``dados`` (list of sentence records with ``sentenca`` and ``label``).
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "resultados" in payload:
        return payload["resultados"]

    raise ValueError(
        "Formato de entrada inválido: esperado objeto com chave 'resultados'"
    )


# ---------------------------------------------------------------------------
# Sliding-window NER
# ---------------------------------------------------------------------------


def _sliding_window_ner(
    text: str,
    tokenizer,
    ner_pipeline,
    max_tokens: int = 510,
    stride: int = 256,
) -> tuple[list[dict], bool]:
    """Run NER over *text* using a sliding window when it exceeds *max_tokens*.

    Returns ``(entities, was_windowed)`` where *entities* is a deduplicated
    list of entity dicts (keys: text, label, start, end, score) with offsets
    relative to the **original** *text*, and *was_windowed* indicates whether
    the text required more than one window.
    """
    if not text:
        return [], False

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # --- single window (fits in model context) ---
    if len(token_ids) <= max_tokens:
        raw = ner_pipeline(text)
        entities = [
            {
                "text": str(e["word"]),
                "label": str(e["entity_group"]),
                "start": int(e["start"]),
                "end": int(e["end"]),
                "score": float(round(float(e["score"]), 4)),
            }
            for e in raw
        ]
        return entities, False

    # --- multiple overlapping windows ---
    # Build a mapping from token index -> character offset so that we can
    # recover the original-text offsets after running NER on each chunk.
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets_map = encoding["offset_mapping"]  # list of (char_start, char_end)

    all_entities: list[dict] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))

        # Character span covered by this window
        chunk_char_start = offsets_map[start][0]
        chunk_char_end = offsets_map[end - 1][1]
        chunk_text = text[chunk_char_start:chunk_char_end]

        # Safety: re-tokenization of decoded text may exceed max_tokens due to
        # whitespace/boundary differences.  Truncate explicitly.
        raw = []
        if chunk_text.strip():
            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
            if len(chunk_tokens) > max_tokens + 2:
                # Re-truncate the chunk text to fit
                trunc_ids = tokenizer.encode(chunk_text, add_special_tokens=False)[:max_tokens]
                chunk_text = tokenizer.decode(trunc_ids, skip_special_tokens=True)
            raw = ner_pipeline(chunk_text)

        for e in raw:
            # Shift character offsets back to the original text
            abs_start = int(e["start"]) + chunk_char_start
            abs_end = int(e["end"]) + chunk_char_start
            all_entities.append(
                {
                    "text": str(e["word"]),
                    "label": str(e["entity_group"]),
                    "start": abs_start,
                    "end": abs_end,
                    "score": float(round(float(e["score"]), 4)),
                }
            )

        if end >= len(token_ids):
            break
        start += stride

    # Deduplicate overlapping entities: when two entities share the same
    # (label) and have overlapping character spans, keep the one with the
    # higher confidence score.
    entities = _deduplicate_entities(all_entities)
    return entities, True


def _spans_overlap(a: dict, b: dict) -> bool:
    """Return True if two entity spans overlap."""
    return a["start"] < b["end"] and b["start"] < a["end"]


def _deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Merge overlapping entities of the same label, keeping highest score."""
    if not entities:
        return []

    # Sort by start offset, then by score descending
    sorted_ents = sorted(entities, key=lambda e: (e["start"], -e["score"]))
    kept: list[dict] = []

    for ent in sorted_ents:
        merged = False
        for i, existing in enumerate(kept):
            if ent["label"] == existing["label"] and _spans_overlap(ent, existing):
                # Keep the one with the higher score
                if ent["score"] > existing["score"]:
                    kept[i] = ent
                merged = True
                break
        if not merged:
            kept.append(ent)

    return sorted(kept, key=lambda e: e["start"])


# ---------------------------------------------------------------------------
# CSV flattening
# ---------------------------------------------------------------------------


def flatten_to_csv_rows(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in results:
        base = {
            "doc_id": item["doc_id"],
            "id": item["id"],
            "sentenca": item["sentenca"],
            "secao": item["label"],
            "windowed": item.get("windowed", False),
        }
        entities = item.get("entidades", [])
        if not entities:
            rows.append(
                {
                    **base,
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
                    **base,
                    "entidade_texto": ent["text"],
                    "entidade_label": ent["label"],
                    "entidade_start": ent["start"],
                    "entidade_end": ent["end"],
                    "entidade_score": ent["score"],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Per-document aggregation
# ---------------------------------------------------------------------------


def aggregate_by_document(results: list[dict]) -> list[dict]:
    """Group sentence-level NER results by ``doc_id`` and deduplicate entities.

    Deduplication is by ``(text, label)`` pair, keeping the highest score.
    """
    doc_entities: dict[int, dict[tuple[str, str], dict]] = defaultdict(dict)

    for item in results:
        doc_id = item["doc_id"]
        for ent in item.get("entidades", []):
            key = (ent["text"], ent["label"])
            existing = doc_entities[doc_id].get(key)
            if existing is None or ent["score"] > existing["score"]:
                doc_entities[doc_id][key] = {
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": ent["score"],
                }

    aggregated = []
    for doc_id in sorted(doc_entities.keys()):
        entities = sorted(
            doc_entities[doc_id].values(),
            key=lambda e: (e["label"], e["text"]),
        )
        aggregated.append(
            {
                "doc_id": doc_id,
                "total_entities": len(entities),
                "extracted_entities": entities,
            }
        )
    return aggregated


# ---------------------------------------------------------------------------
# Section-aware entity grouping
# ---------------------------------------------------------------------------


def group_entities_by_section(results: list[dict]) -> list[dict]:
    """Group entities by ``(doc_id, section)`` and organise them by entity type.

    For each document-section pair, entities are deduplicated by
    ``(text, label)`` keeping the highest confidence score, then organised
    into a dict keyed by entity label with sorted unique values.

    Returns a list of document objects, each containing a ``sections`` array::

        [
          {
            "doc_id": 0,
            "sections": [
              {
                "section": "DOS_FATOS",
                "entities_by_type": {
                  "PESSOA": [{"text": "...", "score": 0.99}, ...],
                  "LOCAL":  [{"text": "...", "score": 0.98}, ...],
                  ...
                },
                "total_entities": 12
              },
              ...
            ]
          },
          ...
        ]
    """
    # doc_id -> section -> (text, label) -> best entity
    tree: dict[int, dict[str, dict[tuple[str, str], dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for item in results:
        doc_id = item["doc_id"]
        section = item["label"]
        for ent in item.get("entidades", []):
            key = (ent["text"], ent["label"])
            existing = tree[doc_id][section].get(key)
            if existing is None or ent["score"] > existing["score"]:
                tree[doc_id][section][key] = {
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": ent["score"],
                }

    output: list[dict] = []
    for doc_id in sorted(tree.keys()):
        sections_list: list[dict] = []
        for section in sorted(tree[doc_id].keys()):
            # Organise by entity type
            by_type: dict[str, list[dict]] = defaultdict(list)
            for ent in tree[doc_id][section].values():
                by_type[ent["label"]].append(
                    {"text": ent["text"], "score": ent["score"]}
                )
            # Sort values inside each type alphabetically
            for label in by_type:
                by_type[label] = sorted(by_type[label], key=lambda e: e["text"])

            total = sum(len(v) for v in by_type.values())
            sections_list.append(
                {
                    "section": section,
                    "entities_by_type": dict(sorted(by_type.items())),
                    "total_entities": total,
                }
            )
        output.append({"doc_id": doc_id, "sections": sections_list})

    return output


# ---------------------------------------------------------------------------
# Sentence-level co-occurrence grouping
# ---------------------------------------------------------------------------


def group_entities_by_sentence(results: list[dict]) -> list[dict]:
    """Group entities by sentence, preserving co-occurrence context.

    Only sentences with at least 2 distinct entity types are included,
    since single-type sentences don't provide relational information.

    Returns a list of document objects, each containing a ``sentences`` array::

        [
          {
            "doc_id": 0,
            "sentences": [
              {
                "sentence_id": 18,
                "section": "DOS_FATOS",
                "sentenca": "No período...",
                "entities_by_type": {
                  "PESSOA": ["EDVALDO AVELINO"],
                  "TEMPO": ["02/03/2021 a 26/04/2021"],
                  "LOCAL": ["Fazenda Tamboril"]
                }
              },
              ...
            ]
          },
          ...
        ]
    """
    doc_sentences: dict[int, list[dict]] = defaultdict(list)

    for item in results:
        entities = item.get("entidades", [])
        if not entities:
            continue

        # Group entities by type within this sentence
        by_type: dict[str, list[str]] = defaultdict(list)
        for ent in entities:
            if ent["text"] not in by_type[ent["label"]]:
                by_type[ent["label"]].append(ent["text"])

        # Only include sentences with 2+ distinct entity types
        if len(by_type) < 2:
            continue

        doc_sentences[item["doc_id"]].append(
            {
                "sentence_id": item["id"],
                "section": item["label"],
                "sentenca": item["sentenca"],
                "entities_by_type": dict(sorted(by_type.items())),
            }
        )

    output: list[dict] = []
    for doc_id in sorted(doc_sentences.keys()):
        output.append(
            {
                "doc_id": doc_id,
                "total_sentences_with_cooccurrence": len(doc_sentences[doc_id]),
                "sentences": doc_sentences[doc_id],
            }
        )
    return output


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_stage4_ner(
    input_json: str,
    output_json: str,
    output_csv: str,
    output_by_doc: str,
    output_by_section: str,
    output_by_sentence: str,
    model_name: str = "dominguesm/legal-bert-ner-base-cased-ptbr",
    sentence_key: str = "sentenca",
    label_key: str = "label",
    window_stride: int = 256,
) -> dict:
    documents = _load_resultado_anotacao(input_json)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner = pipeline(
        "ner",
        model=model_name,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        device=-1,
    )
    max_tokens = min(getattr(tokenizer, "model_max_length", 512), 512) - 2

    results: list[dict] = []
    sentence_idx = 0
    windowed_count = 0

    for doc in documents:
        doc_id = doc["doc_id"]
        sentences = doc.get("dados", [])
        print(
            f"[doc_id={doc_id}] Processando {len(sentences)} sentenças...",
            flush=True,
        )

        for sent_record in sentences:
            text = str(sent_record.get(sentence_key, ""))
            section = str(sent_record.get(label_key, ""))

            entities, was_windowed = _sliding_window_ner(
                text,
                tokenizer,
                ner,
                max_tokens=max_tokens,
                stride=window_stride,
            )
            if was_windowed:
                windowed_count += 1

            results.append(
                {
                    "doc_id": doc_id,
                    "id": sentence_idx,
                    "sentenca": text,
                    "label": section,
                    "entidades": entities,
                    "num_entidades": len(entities),
                    "windowed": was_windowed,
                }
            )
            sentence_idx += 1

    # --- Save sentence-level outputs ---
    ensure_parent_dir(output_json)
    save_json(to_native(results), output_json)

    ensure_parent_dir(output_csv)
    pd.DataFrame(flatten_to_csv_rows(results)).to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )

    # --- Save per-document aggregated output ---
    aggregated = aggregate_by_document(results)
    ensure_parent_dir(output_by_doc)
    save_json(to_native(aggregated), output_by_doc)

    # --- Save section-aware grouped output ---
    grouped = group_entities_by_section(results)
    ensure_parent_dir(output_by_section)
    save_json(to_native(grouped), output_by_section)

    # --- Save sentence-level co-occurrence output ---
    by_sentence = group_entities_by_sentence(results)
    ensure_parent_dir(output_by_sentence)
    save_json(to_native(by_sentence), output_by_sentence)

    # --- Statistics ---
    total = len(results)
    with_entities = sum(1 for x in results if x["num_entidades"] > 0)
    entities_total = sum(x["num_entidades"] for x in results)

    stats = {
        "total_documentos": len(documents),
        "total_sentencas": total,
        "sentencas_com_entidades": with_entities,
        "total_entidades": entities_total,
        "sentencas_com_sliding_window": windowed_count,
        "media_entidades_por_sentenca": float(entities_total / total) if total else 0.0,
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 4 - Extração NER")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--output-by-doc",
        default="files/NER/ner_results_by_document.json",
        help="Output JSON aggregated by document (default: files/NER/ner_results_by_document.json)",
    )
    parser.add_argument(
        "--output-by-section",
        default="files/NER/ner_results_by_section.json",
        help="Output JSON with entities grouped by document and section (default: files/NER/ner_results_by_section.json)",
    )
    parser.add_argument(
        "--output-by-sentence",
        default="files/NER/ner_results_by_sentence.json",
        help="Output JSON with sentence-level entity co-occurrence (default: files/NER/ner_results_by_sentence.json)",
    )
    parser.add_argument(
        "--model-name", default="dominguesm/legal-bert-ner-base-cased-ptbr"
    )
    parser.add_argument("--sentence-key", default="sentenca")
    parser.add_argument("--label-key", default="label")
    parser.add_argument(
        "--window-stride",
        type=int,
        default=256,
        help="Stride (in tokens) for the sliding window on long sentences (default: 256)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = run_stage4_ner(
        input_json=args.input_json,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_by_doc=args.output_by_doc,
        output_by_section=args.output_by_section,
        output_by_sentence=args.output_by_sentence,
        model_name=args.model_name,
        sentence_key=args.sentence_key,
        label_key=args.label_key,
        window_stride=args.window_stride,
    )
    print("\nResumo etapa 4:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
