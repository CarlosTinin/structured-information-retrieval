"""Experiment 3: spaCy Portuguese NER comparison baseline.

Compares the domain-specific Legal-BERT NER model against spaCy's
general-purpose Portuguese NER (pt_core_news_lg) to contextualize
the model choice.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy


# Entity type mapping: spaCy -> Legal-BERT taxonomy
SPACY_TO_LEGAL_BERT = {
    "PER": "PESSOA",
    "ORG": "ORGANIZACAO",
    "LOC": "LOCAL",
    "MISC": None,  # No direct mapping
}

# Legal-BERT types that have no spaCy equivalent
LEGAL_BERT_ONLY_TYPES = {"LEGISLACAO", "TEMPO", "JURISPRUDENCIA"}


def run_spacy_ner_on_documents(
    input_csv: str = "files/output/dataset_normalized_for_ner.csv",
    segmented_results: str = "files/NER/ner_results_by_document_v2.json",
    output_json: str = "output/experiment3_spacy_ner_comparison.json",
    text_column: str = "texto_ner",
    filter_column: str = "decisao",
    filter_value: str = "condenação",
    spacy_model: str = "pt_core_news_lg",
):
    """Run spaCy NER and compare with Legal-BERT results."""
    
    # 1. Load data
    print("Loading conviction documents...")
    df = pd.read_csv(input_csv)
    conviction_docs = df[df[filter_column] == filter_value].reset_index(drop=True)
    print(f"  Found {len(conviction_docs)} conviction documents")
    
    print("Loading Legal-BERT NER results for comparison...")
    with open(segmented_results, "r", encoding="utf-8") as f:
        legal_bert_data = json.load(f)
    
    # 2. Load spaCy model
    print(f"Loading spaCy model: {spacy_model}")
    nlp = spacy.load(spacy_model)
    # Increase max_length for legal documents
    nlp.max_length = 200000
    print(f"  Model loaded (pipeline: {[p for p in nlp.pipe_names]})")
    
    # 3. Run spaCy NER on each document
    print("\nRunning spaCy NER...")
    spacy_results = []
    
    for idx, row in conviction_docs.iterrows():
        text = str(row[text_column])
        doc_id = idx
        
        doc = nlp(text)
        
        # Extract entities, deduplicate by (text, label)
        entity_map = {}
        for ent in doc.ents:
            key = (ent.text.strip(), ent.label_)
            if key not in entity_map:
                entity_map[key] = {
                    "text": ent.text.strip(),
                    "label_spacy": ent.label_,
                    "label_mapped": SPACY_TO_LEGAL_BERT.get(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
        
        entities = sorted(entity_map.values(), key=lambda e: (e["label_spacy"], e["text"]))
        
        spacy_results.append({
            "doc_id": doc_id,
            "total_entities": len(entities),
            "entities": entities,
        })
        
        print(f"  Doc {doc_id}: {len(entities)} unique entities "
              f"(PER={sum(1 for e in entities if e['label_spacy']=='PER')}, "
              f"ORG={sum(1 for e in entities if e['label_spacy']=='ORG')}, "
              f"LOC={sum(1 for e in entities if e['label_spacy']=='LOC')}, "
              f"MISC={sum(1 for e in entities if e['label_spacy']=='MISC')})")
    
    # 4. Compare with Legal-BERT results
    print("\nComparing spaCy vs Legal-BERT NER...")
    comparison = compare_ner_systems(legal_bert_data, spacy_results)
    
    # 5. Build output
    output = {
        "description": "spaCy NER baseline comparison with Legal-BERT NER",
        "spacy_model": spacy_model,
        "legal_bert_model": "dominguesm/legal-bert-ner-base-cased-ptbr",
        "total_documents": len(conviction_docs),
        "spacy_summary": {
            "total_unique_entities": sum(d["total_entities"] for d in spacy_results),
            "avg_per_doc": float(np.mean([d["total_entities"] for d in spacy_results])),
            "entity_type_distribution": dict(comparison["spacy_type_counts"]),
        },
        "legal_bert_summary": {
            "total_unique_entities": sum(d["total_entities"] for d in legal_bert_data),
            "avg_per_doc": float(np.mean([d["total_entities"] for d in legal_bert_data])),
            "entity_type_distribution": dict(comparison["legal_bert_type_counts"]),
        },
        "comparison": comparison["metrics"],
        "interpretation": {
            "note": "spaCy pt_core_news_lg provides PER/ORG/LOC/MISC entity types. "
                    "Legal-BERT provides PESSOA/ORGANIZACAO/LOCAL/TEMPO/LEGISLACAO/JURISPRUDENCIA. "
                    "Comparison is limited to overlapping types (PER↔PESSOA, ORG↔ORGANIZACAO, LOC↔LOCAL). "
                    "Legal-BERT's unique types (TEMPO, LEGISLACAO, JURISPRUDENCIA) represent domain-specific "
                    "value that general-purpose NER cannot provide.",
        },
    }
    
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY:")
    print(f"  spaCy total entities:      {output['spacy_summary']['total_unique_entities']}")
    print(f"  Legal-BERT total entities: {output['legal_bert_summary']['total_unique_entities']}")
    print(f"\n  Comparable types (PER/ORG/LOC):")
    print(f"    spaCy:      {comparison['metrics']['spacy_comparable_count']}")
    print(f"    Legal-BERT: {comparison['metrics']['legal_bert_comparable_count']}")
    print(f"    Text overlap (Jaccard on comparable types): {comparison['metrics']['comparable_jaccard']:.2%}")
    print(f"\n  Legal-BERT exclusive types (TEMPO+LEGISLACAO+JURISPRUDENCIA):")
    print(f"    {comparison['metrics']['legal_bert_exclusive_count']} entities")
    print(f"\n  spaCy MISC entities (no Legal-BERT equivalent):")
    print(f"    {comparison['metrics']['spacy_misc_count']} entities")
    print(f"\nResults saved to {output_json}")
    
    return output


def compare_ner_systems(legal_bert_data: list[dict], spacy_results: list[dict]) -> dict:
    """Compare entity extraction between Legal-BERT and spaCy."""
    
    spacy_type_counts = defaultdict(int)
    legal_bert_type_counts = defaultdict(int)
    
    # For comparable types only
    comparable_overlap = 0
    comparable_only_spacy = 0
    comparable_only_bert = 0
    
    for bert_doc, spacy_doc in zip(legal_bert_data, spacy_results):
        # Legal-BERT entities
        bert_entities_comparable = set()
        bert_entities_all = set()
        for e in bert_doc["extracted_entities"]:
            legal_bert_type_counts[e["label"]] += 1
            bert_entities_all.add((e["text"], e["label"]))
            if e["label"] in ("PESSOA", "ORGANIZACAO", "LOCAL"):
                bert_entities_comparable.add((e["text"].lower().strip(), e["label"]))
        
        # spaCy entities (mapped to Legal-BERT taxonomy)
        spacy_entities_comparable = set()
        for e in spacy_doc["entities"]:
            spacy_type_counts[e["label_spacy"]] += 1
            if e["label_mapped"]:  # Has a Legal-BERT equivalent
                spacy_entities_comparable.add((e["text"].lower().strip(), e["label_mapped"]))
        
        # Compare on comparable types (case-insensitive)
        overlap = bert_entities_comparable & spacy_entities_comparable
        only_bert = bert_entities_comparable - spacy_entities_comparable
        only_spacy = spacy_entities_comparable - bert_entities_comparable
        
        comparable_overlap += len(overlap)
        comparable_only_bert += len(only_bert)
        comparable_only_spacy += len(only_spacy)
    
    # Totals for comparable types
    total_bert_comparable = comparable_overlap + comparable_only_bert
    total_spacy_comparable = comparable_overlap + comparable_only_spacy
    total_union_comparable = comparable_overlap + comparable_only_bert + comparable_only_spacy
    jaccard_comparable = comparable_overlap / total_union_comparable if total_union_comparable > 0 else 0.0
    
    # Legal-BERT exclusive types
    bert_exclusive = sum(legal_bert_type_counts.get(t, 0) for t in LEGAL_BERT_ONLY_TYPES)
    
    metrics = {
        "comparable_jaccard": jaccard_comparable,
        "comparable_overlap": comparable_overlap,
        "comparable_only_legal_bert": comparable_only_bert,
        "comparable_only_spacy": comparable_only_spacy,
        "spacy_comparable_count": total_spacy_comparable,
        "legal_bert_comparable_count": total_bert_comparable,
        "legal_bert_exclusive_count": bert_exclusive,
        "spacy_misc_count": int(spacy_type_counts.get("MISC", 0)),
    }
    
    return {
        "metrics": metrics,
        "spacy_type_counts": dict(spacy_type_counts),
        "legal_bert_type_counts": dict(legal_bert_type_counts),
    }


if __name__ == "__main__":
    run_spacy_ner_on_documents()
