"""Experiment 1: NER without segmentation ablation.

Runs NER directly on full (unsegmented) conviction documents and compares
with the segmented pipeline output to quantify the impact of Stage 3.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def run_unsegmented_ner(
    input_csv: str = "files/output/dataset_normalized_for_ner.csv",
    segmented_results: str = "files/NER/ner_results_by_document_v2.json",
    output_json: str = "output/experiment1_ner_ablation.json",
    model_name: str = "dominguesm/legal-bert-ner-base-cased-ptbr",
    text_column: str = "texto_ner",
    filter_column: str = "decisao",
    filter_value: str = "condenação",
    window_stride: int = 256,
):
    """Run NER on unsegmented documents and compare with segmented pipeline."""
    
    from transformers import AutoTokenizer, pipeline as hf_pipeline
    
    # 1. Load conviction documents
    print("Loading conviction documents...")
    df = pd.read_csv(input_csv)
    conviction_docs = df[df[filter_column] == filter_value].reset_index(drop=True)
    print(f"  Found {len(conviction_docs)} conviction documents")
    
    # 2. Load segmented results for comparison
    print("Loading segmented NER results...")
    with open(segmented_results, "r", encoding="utf-8") as f:
        segmented_data = json.load(f)
    print(f"  Loaded {len(segmented_data)} segmented document results")
    
    # 3. Initialize NER model
    print(f"Loading NER model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner = hf_pipeline(
        "ner",
        model=model_name,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        device=-1,
    )
    max_tokens = min(getattr(tokenizer, "model_max_length", 512), 512) - 2
    print(f"  Model loaded, max_tokens={max_tokens}")
    
    # Import sliding window from stage4
    from .stage4_ner import _sliding_window_ner
    
    # 4. Run NER on each full document (unsegmented)
    print("\nRunning NER on unsegmented documents...")
    unsegmented_results = []
    
    for idx, row in conviction_docs.iterrows():
        text = str(row[text_column])
        doc_id = idx
        
        print(f"  Doc {doc_id}: {len(text)} chars, ", end="")
        
        entities, was_windowed = _sliding_window_ner(
            text, tokenizer, ner, max_tokens=max_tokens, stride=window_stride
        )
        
        # Deduplicate by (text, label) keeping highest score
        deduped = {}
        for ent in entities:
            key = (ent["text"], ent["label"])
            if key not in deduped or ent["score"] > deduped[key]["score"]:
                deduped[key] = ent
        
        unique_entities = sorted(deduped.values(), key=lambda e: (e["label"], e["text"]))
        
        unsegmented_results.append({
            "doc_id": doc_id,
            "total_entities_raw": len(entities),
            "total_entities_deduped": len(unique_entities),
            "windowed": was_windowed,
            "entities": unique_entities,
        })
        
        print(f"{len(entities)} raw -> {len(unique_entities)} unique entities "
              f"({'windowed' if was_windowed else 'single-pass'})")
    
    # 5. Compare with segmented results
    print("\nComparing segmented vs unsegmented...")
    comparison = compare_results(segmented_data, unsegmented_results)
    
    # 6. Build output
    output = {
        "description": "NER ablation: segmented pipeline vs direct application on full documents",
        "model": model_name,
        "total_documents": len(conviction_docs),
        "unsegmented_summary": {
            "total_unique_entities": sum(d["total_entities_deduped"] for d in unsegmented_results),
            "total_raw_entities": sum(d["total_entities_raw"] for d in unsegmented_results),
            "docs_requiring_windowing": sum(1 for d in unsegmented_results if d["windowed"]),
            "avg_entities_per_doc": float(np.mean([d["total_entities_deduped"] for d in unsegmented_results])),
            "std_entities_per_doc": float(np.std([d["total_entities_deduped"] for d in unsegmented_results])),
        },
        "segmented_summary": {
            "total_unique_entities": sum(d["total_entities"] for d in segmented_data),
            "avg_entities_per_doc": float(np.mean([d["total_entities"] for d in segmented_data])),
            "std_entities_per_doc": float(np.std([d["total_entities"] for d in segmented_data])),
        },
        "comparison": comparison,
        "per_document": unsegmented_results,
    }
    
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(convert(output), f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY:")
    print(f"  Segmented pipeline:   {output['segmented_summary']['total_unique_entities']} unique entities")
    print(f"  Unsegmented baseline: {output['unsegmented_summary']['total_unique_entities']} unique entities")
    print(f"  Entity overlap (Jaccard): {comparison['jaccard_similarity']:.2%}")
    print(f"  Entities only in segmented: {comparison['only_in_segmented']}")
    print(f"  Entities only in unsegmented: {comparison['only_in_unsegmented']}")
    print(f"  Avg confidence (segmented):   {comparison.get('avg_confidence_segmented', 'N/A')}")
    print(f"  Avg confidence (unsegmented): {comparison.get('avg_confidence_unsegmented', 'N/A')}")
    print(f"\nResults saved to {output_json}")
    
    return output


def compare_results(segmented_data: list[dict], unsegmented_results: list[dict]) -> dict:
    """Compare entity extraction between segmented and unsegmented approaches."""
    
    # Build entity sets per document for comparison
    total_overlap = 0
    total_only_seg = 0
    total_only_unseg = 0
    
    confidence_segmented = []
    confidence_unsegmented = []
    
    entity_type_counts_seg = defaultdict(int)
    entity_type_counts_unseg = defaultdict(int)
    
    per_doc_jaccard = []
    
    for seg_doc, unseg_doc in zip(segmented_data, unsegmented_results):
        # Segmented entities: set of (text, label)
        seg_entities = {(e["text"], e["label"]) for e in seg_doc["extracted_entities"]}
        unseg_entities = {(e["text"], e["label"]) for e in unseg_doc["entities"]}
        
        # Confidence scores
        for e in seg_doc["extracted_entities"]:
            confidence_segmented.append(e["score"])
            entity_type_counts_seg[e["label"]] += 1
        for e in unseg_doc["entities"]:
            confidence_unsegmented.append(e["score"])
            entity_type_counts_unseg[e["label"]] += 1
        
        # Set operations
        overlap = seg_entities & unseg_entities
        only_seg = seg_entities - unseg_entities
        only_unseg = unseg_entities - seg_entities
        
        total_overlap += len(overlap)
        total_only_seg += len(only_seg)
        total_only_unseg += len(only_unseg)
        
        # Jaccard per doc
        union = seg_entities | unseg_entities
        jaccard = len(overlap) / len(union) if union else 1.0
        per_doc_jaccard.append(jaccard)
    
    # Overall Jaccard
    total_union = total_overlap + total_only_seg + total_only_unseg
    overall_jaccard = total_overlap / total_union if total_union > 0 else 0.0
    
    return {
        "jaccard_similarity": overall_jaccard,
        "entities_in_both": total_overlap,
        "only_in_segmented": total_only_seg,
        "only_in_unsegmented": total_only_unseg,
        "total_union": total_union,
        "avg_confidence_segmented": float(np.mean(confidence_segmented)) if confidence_segmented else None,
        "avg_confidence_unsegmented": float(np.mean(confidence_unsegmented)) if confidence_unsegmented else None,
        "avg_jaccard_per_doc": float(np.mean(per_doc_jaccard)),
        "std_jaccard_per_doc": float(np.std(per_doc_jaccard)),
        "entity_type_distribution_segmented": dict(entity_type_counts_seg),
        "entity_type_distribution_unsegmented": dict(entity_type_counts_unseg),
    }


if __name__ == "__main__":
    run_unsegmented_ner()
