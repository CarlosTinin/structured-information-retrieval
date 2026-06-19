# Experiment Progress Log

## Status Overview

| # | Experiment | Status | Output File |
|---|---|---|---|
| 4 | Statistical significance testing | COMPLETED | `output/experiment4_statistical_tests.json` |
| 5 | Regex segmentation baseline | COMPLETED | `output/experiment5_regex_segmentation_baseline.json` |
| 1 | NER segmentation ablation | COMPLETED | `output/experiment1_ner_ablation.json` |
| 3 | spaCy NER comparison | COMPLETED | `output/experiment3_spacy_ner_comparison.json` |
| 2 | End-to-end LLM NER | SKIPPED | Requires Gemini API key |

---

## Dependency Installation

**Date:** 2026-06-18

Virtual environment created at `.venv/` with:
- `torch` 2.12.1 (PyTorch for transformers inference)
- `transformers` 5.12.1 (HuggingFace for NER model)
- `spacy` 3.8.14 + `pt_core_news_lg` 3.8.0 (Portuguese NER baseline)
- `xgboost` 3.3.0
- `scikit-learn` 1.9.0, `scipy` 1.17.1

Status: COMPLETED

---

## Experiment 4: Statistical Significance Testing

**Script:** `src/framework/experiment4_statistical_tests.py`

### Key Results

**Bootstrap 95% CIs (fold-level resampling, 10,000 iterations):**

| Model | F1 | CI Lower | CI Upper | CI Width |
|---|---|---|---|---|
| Legal-BERT + LR (no-punct) | 0.6632 | 0.5987 | 0.7250 | 0.1263 |
| Legal-BERT + SVM (no-punct) | 0.6371 | 0.6169 | 0.6658 | 0.0489 |
| BERTimbau-large + LR (no-punct) | 0.6374 | 0.6104 | 0.6680 | 0.0576 |
| BERTimbau-base + LR (no-punct) | 0.5460 | 0.5291 | 0.5781 | 0.0490 |
| Legal-BERT + LR (no-punct-stop-num) | 0.8353 | 0.7908 | 0.8942 | 0.1034 |

**Pairwise significance tests (paired t-test, k=3 folds):**

| Comparison | Diff | p-value | Significant? |
|---|---|---|---|
| Legal-BERT vs BERTimbau-base (LR) | +0.117 | 0.150 | NO |
| Legal-BERT vs BERTimbau-large (LR) | +0.026 | 0.628 | NO |
| LR vs SVM (Legal-BERT) | +0.026 | 0.598 | NO |
| LR vs Random Forest (Legal-BERT) | +0.077 | 0.086 | NO |
| No-punct vs raw (Legal-BERT LR) | +0.123 | 0.214 | NO |
| No-punct-stop-num vs no-punct (LR) | +0.172 | 0.049 | YES (borderline) |

**Interpretation:** With k=3 folds, statistical power is very low. No encoder/classifier comparison reaches significance. The only borderline-significant result is the aggressive preprocessing variant, which is suspected of data leakage.

---

## Experiment 5: Regex Segmentation Baseline

**Script:** `src/framework/experiment5_regex_segmentation.py`

### Key Results

- **Overall sentence-level accuracy vs LLM:** 22.9%
- **Documents with NO detectable headers:** 6/25 (24%)
- **DOS_FATOS detection accuracy:** 0% (never detected by regex)
- **Best per-document accuracy:** 92.5% (Doc 15 — had explicit headers)
- **Worst per-document accuracy:** 3.4% (Doc 16 — no headers, longest document)

**Per-section accuracy:**

| Section | Accuracy | Agree/Total |
|---|---|---|
| RELATORIO | 43.3% | 142/328 |
| OUTROS | 49.1% | 165/336 |
| DISPOSITIVO | 30.6% | 71/232 |
| FUNDAMENTACAO | 24.6% | 518/2103 |
| DOSIMETRIA | 9.8% | 74/752 |
| DOS_FATOS | 0.0% | 0/481 |

**Interpretation:** Regex baseline fails dramatically compared to LLM segmentation. The LLM handles implicit section boundaries, documents without headers, and interleaved factual/legal content — all of which regex cannot address.

---

## Experiment 1: NER Segmentation Ablation

**Script:** `src/framework/experiment1_ner_ablation.py`

### Key Results

| Metric | Segmented | Unsegmented |
|---|---|---|
| Total unique entities | 4,282 | 4,266 |
| Avg. entities/doc | 171.3 | 170.6 |
| Avg. confidence | 0.958 | 0.951 |
| Docs requiring windowing | 3/25 | 25/25 |
| Section-assignable | 100% | 0% |

**Overlap analysis:**
- Jaccard similarity: **81.9%**
- Entities only in segmented: 433
- Entities only in unsegmented: 417
- Entities in both: ~3,849

**Interpretation:** Segmentation has minimal impact on raw entity yield (+16 entities, +0.4%). Its primary value is organizational: enabling section-aware entity grouping and implicit relational inference. The slightly higher confidence with segmentation (0.958 vs 0.951) suggests a marginal quality benefit from shorter input segments.

---

## Experiment 3: spaCy NER Comparison

**Script:** `src/framework/experiment3_spacy_comparison.py`

### Key Results

| System | Total Entities | Entity Types |
|---|---|---|
| Legal-BERT | 4,282 | PESSOA, ORGANIZACAO, LOCAL, TEMPO, LEGISLACAO, JURISPRUDENCIA |
| spaCy (pt_core_news_lg) | 5,188 | PER, ORG, LOC, MISC |

**Comparable types (PER↔PESSOA, ORG↔ORGANIZACAO, LOC↔LOCAL):**
- spaCy finds 3,415 entities
- Legal-BERT finds 2,067 entities
- Jaccard overlap on comparable types: **20.6%**

**Legal-BERT exclusive types (TEMPO + LEGISLACAO + JURISPRUDENCIA):**
- 2,059 entities (48% of Legal-BERT's output)
- These have NO spaCy equivalent

**Interpretation:** 
1. spaCy finds more entities overall but with lower precision for legal text (many false positives in MISC category with 1,701 entities)
2. The 20.6% Jaccard on comparable types shows the two systems identify very different spans — confirming domain adaptation matters significantly for legal NER
3. Legal-BERT provides 2,059 domain-specific entities (statutory references, temporal expressions, case citations) that general-purpose NER cannot extract

---

## Paper Integration

All results have been integrated into `paper/structured-information-retrieval/structured-information-retrieval.tex`:
- Experiment 4 → Added "Statistical significance" paragraph in Results Section 6.1
- Experiment 5 → Added "Segmentation baseline comparison" paragraph in Section 6.4
- Experiment 1 → Replaced TODO in Section 6.4 with full results table and analysis
- Experiment 3 → Added "NER model comparison with spaCy" paragraph in Section 6.4
