# Neurocomputing Submission Checklist (Project-Specific)

## 1) Novelty and positioning

- Clear novelty claim versus prior legal NLP pipelines.
- Explicit statement of why a **multi-layer** pipeline is necessary for heterogeneous judicial corpora.
- Comparative table against at least 3 strong related papers (same period and task-adjacent setting).

## 2) Research rigor

- Precise research questions and testable hypotheses.
- Full dataset provenance and inclusion/exclusion criteria.
- Label policy grounded in legal doctrine (Art. 149 framing and decision outcomes).
- Annotation protocol with inter-annotator agreement (Cohen's kappa / Krippendorff's alpha).

## 3) Data quality and governance

- Class distribution before/after filtering (all stages).
- Quantification of removed noise (duplicate footers, boilerplate, malformed rows).
- Handling of missing fields and malformed text.
- Privacy, anonymization, and legal-compliance statement (LGPD-sensitive handling).

## 4) Methodological transparency

- Stage-by-stage formalization with inputs/outputs for each stage.
- Explain dual normalization outputs and why each one is needed.
- Explain model choices and hyperparameters.
- Include seeds, number of folds, and split strategy.

## 5) Experimental design expected by high-impact venues

- Strong baselines (lexical and transformer-based where feasible).
- Ablation study for each stage:
  - Without stage1_1 filtering
  - Without stage1_2 dual normalization
  - Without stage3 segmentation
  - Alternative NER model comparison
- Sensitivity analysis (prompt variations, class imbalance, random seeds).
- Statistical significance tests and confidence intervals.

## 6) Metrics and reporting quality

- Classification: macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrices.
- Segmentation: schema-validity, section-boundary agreement, human validation score.
- NER: entity-level strict and relaxed matching; micro and macro metrics.
- End-to-end utility metric: how many usable structured records are produced and with what error rate.

## 7) Error analysis depth

- Taxonomy of errors by stage (data, classification, segmentation, NER).
- Representative hard cases from legal writing style variation.
- Cascading-error discussion (how upstream errors affect downstream tasks).

## 8) Reproducibility package

- Scripted pipeline commands documented from raw input to final outputs.
- Dependency pinning and environment specification.
- Artifact map: where each table/figure/result file is generated.
- Statement about what data can be publicly released and what is restricted.

## 9) Figures and tables that should appear

- Pipeline activity diagram (full workflow).
- Dataset filtering funnel chart (2581 -> filtered subsets).
- Classification performance table (all models).
- Confusion matrices per model.
- Segmentation quality table.
- NER label distribution and performance table.
- Comparative table versus related work.

## 10) Discussion expected for acceptance strength

- Practical implications for labor-law enforcement and supply-chain risk monitoring.
- Limits of transferability to other courts and legal systems.
- Ethical risks of automation in judicial text analysis.
- Cost-latency tradeoff for LLM-assisted segmentation in production.

## 11) Writing quality controls

- Consistent terminology (decision, sentence, conviction, extinction, acquittal).
- Avoid overclaiming causality from observational judicial text.
- Threats to validity section with internal/external/construct validity.
- Clear contribution list aligned with experimental evidence.

## 12) Final submission compliance

- Match Neurocomputing scope in introduction and conclusion.
- Keep manuscript in polished scientific English.
- Ensure references are current (2021-2026) and high-impact.
- Verify template compliance, reference style, and figure resolution.
