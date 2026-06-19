"""Experiment 5: Regex-based segmentation baseline.

Compares LLM-assisted segmentation against a simple rule-based approach
using keyword/regex matching for section headers in Brazilian criminal sentences.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


# Section header patterns for Brazilian criminal sentences
SECTION_PATTERNS = {
    "RELATORIO": [
        r"(?i)^\s*(?:I\s*[-–.)\s]+\s*)?relat[óo]rio\s*$",
        r"(?i)^\s*relat[óo]rio\s*$",
        r"(?i)^\s*I\s*[-–.)]\s*RELAT[ÓO]RIO",
    ],
    "DOS_FATOS": [
        r"(?i)^\s*(?:II\s*[-–.)\s]+\s*)?dos?\s+fatos?\s*$",
        r"(?i)^\s*dos?\s+fatos?\s*$",
        r"(?i)^\s*da\s+(?:instrução|instru[çc][ãa]o)\s*$",
        r"(?i)^\s*II\s*[-–.)]\s*DOS\s+FATOS",
    ],
    "FUNDAMENTACAO": [
        r"(?i)^\s*(?:III\s*[-–.)\s]+\s*)?fundamenta[çc][ãa]o\s*$",
        r"(?i)^\s*fundamenta[çc][ãa]o\s*$",
        r"(?i)^\s*(?:da\s+)?motiva[çc][ãa]o\s*$",
        r"(?i)^\s*III\s*[-–.)]\s*FUNDAMENTA",
        r"(?i)^\s*é\s+o\s+relat[óo]rio\.?\s+(?:passo\s+a\s+)?decid",
    ],
    "DOSIMETRIA": [
        r"(?i)^\s*(?:da\s+)?dosimetria\s*$",
        r"(?i)^\s*(?:IV\s*[-–.)\s]+\s*)?dosimetria\s*$",
        r"(?i)^\s*da\s+(?:fixa[çc][ãa]o\s+da\s+)?pena\s*$",
        r"(?i)^\s*IV\s*[-–.)]\s*DOSIMETRIA",
    ],
    "DISPOSITIVO": [
        r"(?i)^\s*(?:V\s*[-–.)\s]+\s*)?dispositivo\s*$",
        r"(?i)^\s*dispositivo\s*$",
        r"(?i)^\s*ante\s+o\s+exposto",
        r"(?i)^\s*(?:diante|em\s+face)\s+(?:do|de\s+todo\s+o)\s+exposto",
        r"(?i)^\s*isto\s+posto",
        r"(?i)^\s*(?:pelo\s+exposto|por\s+(?:todo\s+o|tudo\s+(?:quanto|isso)))",
        r"(?i)^\s*V\s*[-–.)]\s*DISPOSITIVO",
    ],
}


def detect_section_from_sentence(sentence: str) -> str | None:
    """Check if a sentence is a section header. Returns section label or None."""
    stripped = sentence.strip()
    if len(stripped) > 200:  # Section headers are short
        return None
    
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, stripped):
                return section
    return None


def segment_document_by_regex(sentences: list[dict]) -> list[dict]:
    """Assign section labels to sentences using regex header detection.
    
    Strategy: scan for section headers, assign all subsequent sentences
    to that section until the next header is found. Sentences before
    the first header are labeled OUTROS.
    """
    current_section = "OUTROS"
    result = []
    
    for sent_record in sentences:
        text = sent_record.get("sentenca", "")
        detected = detect_section_from_sentence(text)
        
        if detected:
            current_section = detected
        
        result.append({
            "sentenca": text,
            "label_regex": current_section,
            "label_llm": sent_record.get("label", ""),
            "is_header": detected is not None,
        })
    
    return result


def compute_agreement(segmented_docs: list[dict]) -> dict:
    """Compute agreement metrics between regex and LLM segmentation."""
    total = 0
    agree = 0
    per_section_agree = {}
    per_section_total = {}
    
    # Track section coverage
    regex_sections_found = set()
    llm_sections_found = set()
    
    for sent in segmented_docs:
        label_regex = sent["label_regex"]
        label_llm = sent["label_llm"].upper() if sent["label_llm"] else ""
        
        # Normalize labels for comparison
        label_llm_norm = label_llm.replace(" ", "_")
        
        total += 1
        regex_sections_found.add(label_regex)
        llm_sections_found.add(label_llm_norm)
        
        if label_regex == label_llm_norm:
            agree += 1
        
        # Per-section stats
        if label_llm_norm not in per_section_total:
            per_section_total[label_llm_norm] = 0
            per_section_agree[label_llm_norm] = 0
        per_section_total[label_llm_norm] += 1
        if label_regex == label_llm_norm:
            per_section_agree[label_llm_norm] += 1
    
    per_section_accuracy = {}
    for section in per_section_total:
        per_section_accuracy[section] = {
            "agree": per_section_agree[section],
            "total": per_section_total[section],
            "accuracy": per_section_agree[section] / per_section_total[section] if per_section_total[section] > 0 else 0.0,
        }
    
    return {
        "total_sentences": total,
        "sentences_agree": agree,
        "overall_accuracy": agree / total if total > 0 else 0.0,
        "per_section_accuracy": per_section_accuracy,
        "regex_sections_found": sorted(regex_sections_found),
        "llm_sections_found": sorted(llm_sections_found),
    }


def run_experiment5(
    input_json: str = "files/Documentos-Segmentados/resultado_anotacao.json",
    output_json: str = "output/experiment5_regex_segmentation_baseline.json",
):
    """Run regex segmentation baseline and compare with LLM."""
    
    print("Loading LLM segmentation results...")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = data["resultados"]
    print(f"  Loaded {len(documents)} documents")
    
    all_segmented = []
    per_doc_results = []
    docs_with_no_headers = []
    
    for doc in documents:
        doc_id = doc["doc_id"]
        sentences = doc["dados"]
        
        # Apply regex segmentation
        segmented = segment_document_by_regex(sentences)
        all_segmented.extend(segmented)
        
        # Count headers found
        headers_found = sum(1 for s in segmented if s["is_header"])
        sections_found = set(s["label_regex"] for s in segmented if s["is_header"])
        
        # Per-doc agreement
        doc_agree = compute_agreement(segmented)
        doc_agree["doc_id"] = doc_id
        doc_agree["headers_detected"] = headers_found
        doc_agree["sections_detected"] = sorted(sections_found)
        per_doc_results.append(doc_agree)
        
        if headers_found == 0:
            docs_with_no_headers.append(doc_id)
        
        print(f"  Doc {doc_id}: {headers_found} headers detected, "
              f"accuracy={doc_agree['overall_accuracy']:.2%}, "
              f"sections={sorted(sections_found)}")
    
    # Overall agreement
    overall = compute_agreement(all_segmented)
    
    # Summary
    output = {
        "description": "Regex-based segmentation baseline vs LLM segmentation",
        "method": "Keyword/regex header detection with state-machine section assignment",
        "total_documents": len(documents),
        "total_sentences": overall["total_sentences"],
        "overall_accuracy": overall["overall_accuracy"],
        "per_section_accuracy": overall["per_section_accuracy"],
        "docs_with_no_headers_detected": docs_with_no_headers,
        "num_docs_with_no_headers": len(docs_with_no_headers),
        "per_document_results": per_doc_results,
        "interpretation": {
            "note": "Regex baseline relies on explicit section headers being present in the text. "
                    "Documents without clear headers (common in shorter or less formally structured sentences) "
                    "default to OUTROS for all content, yielding 0% accuracy on those documents.",
            "strengths": "100% reproducible, zero cost, no API dependency, instant execution",
            "weaknesses": "Cannot detect implicit section boundaries, fails on documents without explicit headers, "
                         "cannot handle interleaved factual/legal content within a single section",
        },
    }
    
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS:")
    print(f"  Total sentences: {overall['total_sentences']}")
    print(f"  Overall accuracy: {overall['overall_accuracy']:.2%}")
    print(f"  Docs with NO headers: {len(docs_with_no_headers)}/{len(documents)}")
    print(f"\n  Per-section accuracy:")
    for section, stats in sorted(overall["per_section_accuracy"].items()):
        print(f"    {section:20s}: {stats['accuracy']:.2%} ({stats['agree']}/{stats['total']})")
    print(f"\nResults saved to {output_json}")
    
    return output


if __name__ == "__main__":
    run_experiment5()
