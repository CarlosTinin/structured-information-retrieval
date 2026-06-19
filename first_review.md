# Reviewer Report — Neurocomputing

**Manuscript:** "A Multi-Layer Framework for Filtering, Merit Classification, Sentence Segmentation, and Named-Entity Extraction in Brazilian Criminal Labor-Law Decisions"

**Recommendation: Major Revision**

---

## 1. Summary

The paper proposes a four-stage pipeline for extracting structured information from Brazilian criminal labor-law judicial decisions: (1) rule-based document-type filtering, (2) BERT-embedding-based penal-merit classification, (3) LLM-assisted rhetorical segmentation, and (4) NER via a pre-trained Portuguese legal BERT model. The corpus comprises 2,581 documents from the PJe system, progressively filtered down to 25 conviction decisions from which entities are extracted.

---

## 2. Major Issues

### 2.1 Lack of Quantitative Evaluation for Core Contributions (Critical)

The paper's most fundamental weakness is that **two of its four stages (Stage 3 and Stage 4) lack any formal quantitative evaluation**. The segmentation stage is assessed only through "expert review of 5 randomly sampled documents" with no inter-annotator agreement metric. The NER stage has **no gold-standard evaluation whatsoever** — no precision, recall, or F1 is reported for entity extraction. The authors explicitly acknowledge this and defer it to "future work."

For a venue like Neurocomputing, this is unacceptable. The paper's title and stated contributions center on information extraction, yet the extraction component is evaluated only qualitatively. A reviewer cannot determine whether the pipeline actually works at its stated purpose. At minimum, the authors should annotate 15-20 documents (as they themselves propose in the future work section) and report entity-level P/R/F1 **before** submission.

### 2.2 The Best-Performing Component Is Not the Authors' Contribution

The merit classification stage (Stage 2) is the only stage with rigorous quantitative evaluation. However:

- The best result (Legal-BERT + Logistic Regression) achieves **F1 = 0.66** — a mediocre result that the authors themselves acknowledge "requires human verification."
- The LLM baseline (Gemini zero-shot) achieves **F1 = 0.98**, completely dominating the proposed approach.
- The paper's argument that the embedding pipeline is preferable due to determinism, cost, and air-gapped deployability is operationally valid but **does not constitute a scientific contribution**. It is an engineering trade-off discussion, not a research finding.

This creates a problematic framing: the paper's only rigorously evaluated stage demonstrates that the proposed method is vastly inferior to a straightforward LLM baseline, and the paper's actual novel contributions (the pipeline architecture, the segmentation, the NER) are not quantitatively validated.

### 2.3 Extremely Small Dataset with No Statistical Significance Testing

The entire classification experiment operates on **N=89 documents** with 3-fold cross-validation. This produces test sets of ~30 samples each. At this scale:

- Confidence intervals on F1 are extremely wide (the reported std of ±0.05 is optimistic given the sample size).
- No statistical significance test (e.g., paired t-test, McNemar's test, bootstrap CI) is applied to compare classifiers or encoders.
- The difference between Legal-BERT (F1=0.6632) and BERTimbau-large (F1=0.6374) is **0.026** — well within noise at N=89. Claiming Legal-BERT "outperforms" is not statistically supported.
- The NER evaluation covers only **25 documents**. By any standard in the NER literature, this is a prototype demonstration, not a scientific evaluation.

### 2.4 Stage 1 Is Not a Research Contribution

Stage 1.1 extracts a document-type label from a machine-generated header using a regular expression. Stage 1.2 applies standard text cleaning (regex-based footer removal, lowercasing, Unicode normalization). While operationally necessary, **these are preprocessing steps, not methodological contributions**. Claiming a "96.2% reduction with zero error" as a contribution is misleading — it is simply reading a metadata field. Any practitioner would do the same. Presenting this as a core contribution inflates the paper's apparent novelty.

### 2.5 No Ablation of Segmentation's Impact on NER

The paper claims that segmentation "provides structural context for NER" and that the pipeline's sequential architecture is beneficial. However, the stated ablation — "NER applied directly to unsegmented conviction documents" — **is never reported in the Results section**. The ablation is described in Section 5.2 but no corresponding results appear in Section 6. Without this comparison, the claimed benefit of segmentation for NER is unsubstantiated.

### 2.6 Novelty Concerns

The paper's claimed novelty is that "no prior framework integrates document-type filtering, penal-merit classification, structural segmentation, and named-entity extraction into a single end-to-end pipeline for Portuguese-language legal decisions." This is an extremely narrow novelty claim — it is essentially a language/domain-specific instantiation of well-established NLP pipeline patterns. Each individual component uses off-the-shelf models (Legal-BERT, Gemini API, a pre-trained NER model) without architectural innovation. The contribution is one of **application engineering** rather than methodological advance. While valuable, Neurocomputing typically expects stronger technical novelty.

---

## 3. Minor Issues

### 3.1 Inconsistent Scope Claims

- The abstract states "Results indicate that layered filtering and task-oriented preprocessing substantially improve downstream extraction quality" — but no quantitative evidence for this claim is presented. There is no comparison of NER with vs. without upstream filtering.
- The paper claims to be about "criminal labor-law decisions" broadly but is actually restricted to Article 149 cases from a single court system (TRF1).

### 3.2 The "Dual-Branch Preprocessing" Is Trivially Simple

The task-adaptive preprocessing (Section 4.3) is presented as a contribution, but it simply means "apply aggressive cleaning for classification, apply minimal cleaning for NER." This is common sense, not a novel design principle. Any practitioner building a multi-task pipeline would do the same. The paper's framing ("a task-adaptive preprocessing design that generates two distinct normalized outputs") elevates a trivial engineering decision to the status of a research contribution.

### 3.3 LLM Segmentation Lacks Reproducibility

- The Gemini 2.5 Flash Lite model used for segmentation is a proprietary, version-mutable API. The paper acknowledges non-reproducibility but does not offer any mitigation beyond caching.
- The segmentation prompt is described as "iteratively refined through five pilot documents" — but no details of the refinement process, the errors observed, or the version history are provided. This makes the protocol unreproducible.
- 32% first-pass failure rate for an API-dependent stage is a significant practical limitation that undermines the "end-to-end" framing.

### 3.4 Related Work Is Exhaustive But Lacks Critical Synthesis

The Related Work section (Section 2) is approximately 3 pages of dense citation-by-citation summaries with limited comparative analysis. Many paragraphs follow the pattern "X did Y on Z" without critically positioning the present work's technical choices. The "Comparative synthesis" subsection (Section 2.4) is only one paragraph. A more effective structure would organize around technical dimensions (long document handling, class imbalance, domain adaptation) rather than citation enumeration.

### 3.5 Missing Comparisons with Relevant Baselines

- For NER, no comparison with spaCy, Stanza, or other Portuguese NER tools.
- For segmentation, no comparison with rule-based approaches (e.g., keyword/regex matching for section headers).
- For the overall pipeline, no comparison with an end-to-end LLM approach (e.g., prompting Gemini to extract entities directly from raw text without the intermediate stages).

### 3.6 Questionable Metric Choices

- Weighted F1 as the primary metric for a 3-class imbalanced problem is standard but **masks minority-class performance**. The acquittal class (n=46) dominates the weighted score. Per-class F1 for the minority conviction class (n=25) — which is the class of actual interest — should be prominently reported.
- The paper reports F1=0.66 without clearly stating that this means approximately 1 in 3 convictions may be misclassified, which is operationally catastrophic for the stated supply-chain application.

### 3.7 Circular Reasoning in Pipeline Justification

The paper argues that the pipeline is necessary because "applying NER directly to the full 2,581-document corpus would process 25.1x more text, the majority of which contains no relevant entities." But this conflates efficiency with quality. The relevant question is whether the pipeline **improves NER quality** over direct application — which is never tested.

### 3.8 Presentation Issues

- The paper uses both `\cite` and `\citep` inconsistently.
- Table 1 (document type distribution) sums to ~2,578, not 2,581.
- The "Information density" paragraph (Section 6.4) claims "40-60 hours of expert manual annotation" equivalence with no justification for this estimate.
- The Acknowledgments section contains only TODO comments.

---

## 4. Questions for the Authors

1. Why not use the Gemini model (which achieves F1=0.98 on classification) for the NER task directly, comparing it against the Legal-BERT NER model? This would establish whether the pipeline is actually necessary or whether a single LLM call could replace the entire multi-stage architecture.

2. What is the per-class F1 for the *conviction* class specifically in Stage 2? Given that the downstream pipeline only processes convictions, the recall of the conviction class determines how many relevant documents are lost.

3. The paper states 25 conviction documents were segmented and annotated with NER. Were these the 25 documents identified by the expert labels (ground truth) or by the classifier's predictions? If the latter, misclassified documents would propagate into downstream stages.

4. What happens when the pipeline is applied to new data (e.g., decisions from 2024-2025)? Is there any temporal drift analysis?

---

## 5. Verdict

The paper addresses a relevant problem (structured information extraction from Brazilian legal decisions for supply-chain transparency) and the pipeline architecture is sensible. However, the current manuscript suffers from:

1. **Absence of quantitative evaluation for its core contributions** (NER and segmentation).
2. **The only rigorously evaluated stage shows the proposed method is vastly outperformed** by a simple LLM baseline.
3. **Sample sizes too small for reliable conclusions**, with no significance testing.
4. **Novelty is primarily in application engineering** (combining existing tools for a new domain/language) rather than methodological innovation.

For acceptance at Neurocomputing, the authors would need to:
- Construct and evaluate against a gold-standard annotation for NER (minimum 15-20 documents).
- Report the segmentation ablation results that are promised but missing.
- Provide statistical significance testing for all comparisons.
- Substantially sharpen the novelty claims or introduce genuine technical innovation (e.g., a novel architecture for long-document legal NER, a domain-adapted fine-tuning strategy, or a joint segmentation+extraction model).
- Complete the missing ablation: full LLM end-to-end extraction vs. the proposed pipeline.

---

**Overall Score: 4/10 (Below acceptance threshold)**

**Confidence: High** — The strengths and weaknesses are clearly identifiable from the manuscript.
