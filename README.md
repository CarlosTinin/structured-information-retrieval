# Hybrid AI Framework for Multi-Stage Information Extraction from Judicial Documents

A reproducible hybrid corpus-to-record pipeline for extracting structured information from Brazilian federal-court decisions concerning Article 149 of the Penal Code (modern slavery). The framework progressively filters, classifies, segments, and annotates heterogeneous legal corpora from the PJe (Processo Judicial Eletr&ocirc;nico) system.

The pipeline transforms 2,581 raw documents into structured, section-aware entity records through four stages: metadata-based document-type filtering, penal-merit classification (BERT embeddings + classical classifiers, with an LLM baseline), LLM-assisted rhetorical segmentation, and legal-domain named-entity recognition.

## Code Organization (`src/framework/`)

### Pipeline Modules

- `io_utils.py` &mdash; I/O utilities for CSV detection, reading, and writing.
- `stage1_1_document_type.py` &mdash; Stage 1.1: document-type normalization and filtering (retains only judicial sentences).
- `stage1_2_preprocessing.py` &mdash; Stage 1.2: task-adaptive legal text preprocessing (dual output: aggressive normalization for classification, conservative normalization for segmentation/NER).
- `stage2_embeddings.py` &mdash; Stage 2: frozen BERT embeddings with classical classifiers (Logistic Regression, SVM, Random Forest, XGBoost) for penal-merit classification.
- `stage2_llm_baseline.py` &mdash; Stage 2: Gemini 2.5 Pro zero-shot and few-shot baseline for merit classification, evaluated on the same K-fold splits as the embedding classifiers.
- `stage2_explainability.py` &mdash; Stage 2: SHAP token-level attribution analysis for SVM Linear and XGBoost classifiers.
- `stage2_compare_encoders.py` &mdash; Stage 2: multi-encoder comparison across Legal-BERT, BERTimbau-base, and BERTimbau-large (standalone script, not in CLI).
- `stage3_segmentation.py` &mdash; Stage 3: LLM-assisted rhetorical segmentation of conviction decisions using Gemini, with retry logic and JSON repair.
- `stage4_ner.py` &mdash; Stage 4: legal-domain NER extraction using `dominguesm/legal-bert-ner-base-cased-ptbr`, with sliding-window processing for long sentences.
- `stage4_ner_viz.py` &mdash; Stage 4: NER visualization generation (section&times;entity-type heatmap, pairwise co-occurrence figure, single-document LaTeX table).
- `cli.py` &mdash; Unified CLI entry point for all pipeline stages.
- `__main__.py` &mdash; Package entry point (enables `python -m src.framework`).
- `__init__.py` &mdash; Package initializer.

### Experiment Scripts (standalone, not in CLI)

- `experiment1_ner_ablation.py` &mdash; NER without segmentation ablation: compares NER on segmented vs. unsegmented documents to quantify Stage 3 impact.
- `experiment3_spacy_comparison.py` &mdash; spaCy Portuguese NER baseline: compares Legal-BERT NER against spaCy `pt_core_news_lg`.
- `experiment4_statistical_tests.py` &mdash; Statistical significance testing: bootstrap 95% CIs and approximate significance tests between encoder/classifier pairs.
- `experiment5_regex_segmentation.py` &mdash; Regex segmentation baseline: compares LLM segmentation against keyword/regex header matching.

### Deprecated

- `stage2_finetune.py` &mdash; Fine-tuning approach for Stage 2. Removed from the active pipeline due to high variance with the small labeled set (~89 documents); kept for reference only.

> **Note on stage numbering:** Macro-stage 1 is split into `stage1_1` and `stage1_2`. The former stage 3 was renumbered to stage 2. Segmentation remains stage 3 and NER is the final stage 4.

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

For Stages 2-LLM and 3 (which use the Gemini API), set `GEMINI_API_KEY` in a `.env` file at the project root or as an environment variable. See `.env.example` for the expected format.

### Pipeline Stages (CLI)

All stages are invoked from the project root:

```bash
python -m src.framework <stage> [options]
```

#### Stage 1.1 &mdash; Document-type filtering

Extracts document-type metadata from PJe headers and retains only judicial sentences. Reduces the corpus from 2,581 to 99 documents.

```bash
python -m src.framework stage1_1 \
  --input files/datasets/dataset_completo.csv \
  --output files/output/dataset_filtered_by_type.csv
```

#### Stage 1.2 &mdash; Task-adaptive preprocessing

Applies shared PJe footer removal, then generates two normalized outputs: one for classification (aggressive) and one for segmentation/NER (conservative).

```bash
python -m src.framework stage1_2 \
  --input files/output/dataset_filtered_by_type.csv \
  --output-classification files/output/dataset_normalized.csv \
  --output-ner files/output/dataset_normalized_for_ner.csv
```

#### Stage 2 &mdash; Merit classification (BERT embeddings)

Generates frozen BERT embeddings and trains classical classifiers for penal-merit classification (condenacao, absolvicao, extinto). The recommended preprocessing flags are shown below.

```bash
python -m src.framework stage2-embeddings \
  --input files/output/dataset_normalized.csv \
  --output-root output \
  --strip-punctuation --strip-stopwords --strip-numbers
```

**Outputs:** LaTeX tables in `output/tables/`, confusion matrices in `output/images/`, results JSON in `output/`.

**Optional flags:** `--model-name` (default: `dominguesm/legal-bert-base-cased-ptbr`), `--k-folds` (default: 3), `--target-labels` (default: `condenação,extinto,absolvição`).

#### Stage 2 &mdash; LLM baseline (Gemini 2.5 Pro)

Evaluates zero-shot and few-shot (3 examples/class) merit classification on the same K-fold splits as the embedding classifiers.

```bash
python -m src.framework stage2-llm-baseline \
  --input files/output/dataset_normalized.csv \
  --prompt-file src/prompts/prompt_classification_merit.txt \
  --output-root output
```

**Outputs:** `output/stage2_llm_baseline_results.json`, LaTeX table in `output/tables/table_llm_baseline.tex`, confusion matrices in `output/images/`.

#### Stage 2 &mdash; Explainability (SHAP)

Generates SHAP token-level attribution for SVM Linear and XGBoost classifiers using Legal-BERT embeddings.

```bash
python -m src.framework stage2-explainability \
  --input files/output/dataset_normalized.csv \
  --output-root output \
  --strip-punctuation --strip-stopwords --strip-numbers
```

**Optional flags:** `--results-json` (precomputed embeddings), `--num-background` (default: 20), `--max-tokens` (default: 256).

#### Stage 3 &mdash; Rhetorical segmentation

Decomposes conviction decisions into sentences annotated with rhetorical-function labels (relatorio, dos_fatos, fundamentacao, dosimetria, dispositivo, outros) using a Gemini LLM.

```bash
python -m src.framework stage3-segmentation \
  --input files/output/dataset_normalized_for_ner.csv \
  --prompt-file src/prompts/prompt_segmentation.txt \
  --output-json files/Documentos-Segmentados/resultado_anotacao.json
```

By default, filters only `decisao=condenação` (~25 documents). Configurable via `--filter-label-column` and `--filter-label-value`.

**Authentication:** `GEMINI_API_KEY` from environment or `.env` file.

**Optional flags:** `--model-name` (default: `gemini-2.5-flash-lite`), `--max-docs`, `--sleep-seconds`, `--request-timeout` (default: 180), `--max-retries` (default: 3).

#### Stage 4 &mdash; NER extraction

Applies legal-domain NER to segmented sentences using `dominguesm/legal-bert-ner-base-cased-ptbr`, with sliding-window processing for sentences exceeding the 512-token limit.

```bash
python -m src.framework stage4 \
  --input-json files/Documentos-Segmentados/resultado_anotacao.json \
  --output-json files/NER/ner_results.json \
  --output-csv files/NER/ner_results.csv
```

#### Stage 4 &mdash; NER visualizations

Generates publication-ready figures and tables from NER output.

```bash
python -m src.framework stage4-viz \
  --input-by-section files/NER/ner_results_by_section_v2.json \
  --output-root output \
  --doc-id 0
```

**Outputs:** Heatmap (`output/images/fig2_ner_heatmap.{png,pdf}`), co-occurrence figure (`output/images/fig3_cooccurrence_heatmap.{png,pdf}`), single-document LaTeX table (`output/tables/table_ner_single_doc.tex`).

### Standalone Scripts

These scripts are run directly and use default paths (not through the CLI):

#### Multi-encoder comparison

Compares Stage 2 results across multiple BERT encoder backbones and generates a combined comparison figure and LaTeX table.

```bash
python -m src.framework.stage2_compare_encoders \
  --result-a output/stage2_embeddings_results_no_punct_no_stop_no_num.json \
  --result-b output/encoder_comparison/neuralmind_bert/stage2_embeddings_results_no_punct_no_stop_no_num.json \
  --output-tex output/tables/table_stage2_encoder_model_comparison.tex \
  --output-png output/images/figure_stage2_encoder_model_comparison_f1.png
```

#### Experiment 1 &mdash; NER ablation (without segmentation)

Runs NER directly on full unsegmented conviction documents and compares with the segmented pipeline output to quantify the impact of Stage 3.

```bash
python -m src.framework.experiment1_ner_ablation
```

**Output:** `output/experiment1_ner_ablation.json`

#### Experiment 3 &mdash; spaCy NER comparison

Compares the domain-specific Legal-BERT NER model against spaCy's general-purpose Portuguese NER (`pt_core_news_lg`).

```bash
python -m src.framework.experiment3_spacy_comparison
```

**Output:** `output/experiment3_spacy_ner_comparison.json`

#### Experiment 4 &mdash; Statistical significance testing

Computes bootstrap 95% confidence intervals on per-fold F1 scores and approximate significance tests between encoder/classifier pairs.

```bash
python -m src.framework.experiment4_statistical_tests
```

**Output:** `output/experiment4_statistical_tests.json`

#### Experiment 5 &mdash; Regex segmentation baseline

Compares LLM-assisted segmentation against a simple rule-based approach using keyword/regex matching for section headers in Brazilian criminal sentences.

```bash
python -m src.framework.experiment5_regex_segmentation
```

**Output:** `output/experiment5_regex_segmentation_baseline.json`

## Reproducibility

- All random seeds fixed at 42 for all stochastic operations (K-fold splitting, classifier initialization).
- Dependency versions pinned in `requirements.txt`.
- Stage 3 segmentation results cached in `files/Documentos-Segmentados/`, decoupling Stage 4 from the Gemini API.
- LLM baseline predictions cached in output artifacts, enabling re-runs without API calls.
- All BERT models used are publicly available on HuggingFace.
