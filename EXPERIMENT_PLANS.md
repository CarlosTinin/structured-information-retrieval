# Experiment Plans for Revision

These experiments address the reviewer's critical concerns about missing ablations and baselines.

---

## Experiment 1: NER Without Segmentation Ablation

**Addresses:** Reviewer issues 2.5, 3.7 (missing ablation of segmentation's impact on NER)

### Objective
Quantify whether upstream rhetorical segmentation (Stage 3) improves downstream NER quality compared to applying NER directly to unsegmented documents.

### Design

**Baseline (unsegmented):**  
- Take the 25 conviction documents from `files/output/dataset_normalized_for_ner.csv` (filter by `decisao == "condenação"`)
- For each document, treat the entire text as a single "sentence" with label `"FULL_DOCUMENT"`
- Run `_sliding_window_ner()` on the full document text (the sliding window will handle the long input)
- Aggregate entities at document level using the same deduplication logic

**Treatment (segmented, existing pipeline):**  
- Use the existing `files/NER/ner_results_by_document_v2.json` output

**Comparison metrics:**
1. **Entity yield**: total unique entities per document (segmented vs unsegmented)
2. **Entity type distribution**: does segmentation change which entity types are extracted?
3. **Average confidence scores**: are entities from segmented inputs higher confidence?
4. **Section-assignability**: proportion of entities that can be linked to a rhetorical function (100% for segmented, 0% for unsegmented by construction)
5. **Entity boundary quality**: sample 50 entities from each condition and manually assess partial-span rate
6. **Overlap analysis**: what fraction of unsegmented entities also appear in segmented output? (Jaccard similarity)

### Implementation Plan

Create `src/framework/stage4_ner_ablation.py`:

```python
"""Ablation: NER on unsegmented documents vs. segmented pipeline."""

def run_unsegmented_ner(
    input_csv: str,           # dataset_normalized_for_ner.csv
    filter_column: str,       # "decisao"  
    filter_value: str,        # "condenação"
    text_column: str,         # "texto_ner"
    output_json: str,         # files/NER/ner_ablation_unsegmented.json
    model_name: str = "dominguesm/legal-bert-ner-base-cased-ptbr",
) -> dict:
    """Run NER directly on full document text without segmentation."""
    # 1. Load CSV, filter to conviction docs
    # 2. For each document:
    #    - Get full text from text_column
    #    - Run _sliding_window_ner() on the full text
    #    - Store results with doc_id, entities, windowed flag
    # 3. Aggregate by document (same dedup logic as stage4_ner.py)
    # 4. Save output JSON
    # 5. Compare against segmented results
    pass

def compare_segmented_vs_unsegmented(
    segmented_json: str,       # files/NER/ner_results_by_document_v2.json  
    unsegmented_json: str,     # files/NER/ner_ablation_unsegmented.json
    output_report: str,        # output/ablation_segmentation_report.json
) -> dict:
    """Generate comparative metrics between conditions."""
    # Metrics: entity counts, type distributions, confidence, Jaccard overlap
    pass
```

### CLI Integration
Add `stage4-ablation` subcommand to `cli.py`.

### Expected Outcome
- If segmentation helps: segmented entities should have higher confidence, fewer partial spans, and the section-aware organization provides structural value even if raw counts are similar.
- If segmentation doesn't help raw extraction quality: the contribution is purely organizational (entity-section association), which still has downstream value for relational inference but is a weaker claim.

---

## Experiment 2: End-to-End LLM NER Baseline

**Addresses:** Reviewer issues 2.6, 3.5, Question 1 (why not use LLM for NER directly?)

### Objective
Test whether a single LLM call (Gemini) can extract the same entities from raw conviction documents without the multi-stage pipeline, establishing whether the pipeline architecture adds value.

### Design

**Baseline (single LLM call):**
- For each of the 25 conviction documents, send the full text to Gemini with a NER-focused prompt
- Request structured JSON output with entity types matching the Legal-BERT NER model's taxonomy: PESSOA, LEGISLACAO, TEMPO, ORGANIZACAO, LOCAL, JURISPRUDENCIA
- Also request the section/context where each entity appears (to test implicit segmentation)

**Treatment (pipeline):**
- Use existing `files/NER/ner_results_by_document_v2.json`

**Comparison metrics:**
1. **Entity overlap (Jaccard)**: what fraction of pipeline entities are also found by LLM, and vice versa?
2. **Entity type agreement**: when both systems find the same text span, do they agree on the entity type?
3. **Coverage**: does the LLM find entities that the pipeline misses? (qualitative assessment)
4. **Hallucination rate**: does the LLM produce entity mentions not present in the source text?
5. **Section assignment accuracy**: if the LLM also provides section context, does it match the Stage 3 segmentation?
6. **Cost comparison**: API cost for 25 docs × full-text prompt vs. pipeline cost

### Implementation Plan

Create `src/framework/stage4_llm_ner_baseline.py`:

```python
"""End-to-end LLM NER baseline using Gemini."""

def run_llm_ner_baseline(
    input_csv: str,           # dataset_normalized_for_ner.csv
    prompt_file: str,         # src/prompts/prompt_ner_llm_baseline.txt (new)
    output_json: str,         # output/llm_ner_baseline_results.json
    model_name: str = "gemini-2.5-pro",
    filter_column: str = "decisao",
    filter_value: str = "condenação",
    text_column: str = "texto_ner",
) -> dict:
    """Extract entities via single LLM call per document."""
    # 1. Load + filter docs
    # 2. For each doc: prompt LLM with full text + NER instructions
    # 3. Parse structured JSON response
    # 4. Save results
    pass
```

Create `src/prompts/prompt_ner_llm_baseline.txt`:
- Instruct Gemini to extract all named entities from the document
- Define entity types: PESSOA, LEGISLACAO, TEMPO, ORGANIZACAO, LOCAL, JURISPRUDENCIA
- Request JSON output: `[{"text": "...", "label": "...", "context_section": "..."}]`
- Include examples for each entity type

### Expected Outcome
- LLM likely achieves higher recall (finds more entities) but may hallucinate some
- Pipeline likely achieves higher precision (fewer false positives due to the trained NER model)
- The pipeline's section-aware organization is a unique structural contribution not easily replicated by a single LLM call
- If LLM dominates both precision and recall: the pipeline's value is reduced to cost/reproducibility arguments

---

## Experiment 3: spaCy/Stanza Portuguese NER Comparison

**Addresses:** Reviewer issue 3.5 (missing baselines for NER)

### Objective
Compare the domain-specific Legal-BERT NER model against general-purpose Portuguese NER tools to contextualize the model choice.

### Design

**Baselines:**
1. `spaCy` with `pt_core_news_lg` model (Portuguese NER with standard entity types: PER, ORG, LOC, MISC)
2. `Stanza` with Portuguese models (NER entity types: PER, ORG, LOC)

**Treatment:**
- Existing Legal-BERT NER results

**Challenge:** Entity type taxonomies differ. Need a mapping:
- PESSOA → PER
- ORGANIZACAO → ORG  
- LOCAL → LOC
- TEMPO → (no direct equivalent in spaCy/Stanza - DATE/TIME are not standard NER labels in PT models)
- LEGISLACAO → (no equivalent)
- JURISPRUDENCIA → (no equivalent)

**Comparison metrics (on overlapping entity types only):**
1. Entity counts for PER/ORG/LOC across all three systems
2. Agreement rate: when two systems both find an entity, do they extract the same span?
3. Unique entities: what does each system find that others miss?
4. For the entity types unique to Legal-BERT (LEGISLACAO, TEMPO, JURISPRUDENCIA): report coverage that alternative tools cannot provide

### Implementation Plan

Create `src/framework/stage4_ner_baselines.py`:

```python
"""NER baselines: spaCy and Stanza for comparison."""

def run_spacy_ner(texts: list[str], model: str = "pt_core_news_lg") -> list[list[dict]]:
    """Run spaCy NER on texts."""
    pass

def run_stanza_ner(texts: list[str]) -> list[list[dict]]:
    """Run Stanza NER on texts."""
    pass

def compare_ner_systems(
    legal_bert_results: str,
    spacy_results: str,
    stanza_results: str,
    output_report: str,
) -> dict:
    """Generate comparative report across NER systems."""
    pass
```

### Expected Outcome
- Legal-BERT NER likely outperforms spaCy/Stanza on legal-specific entities (LEGISLACAO, JURISPRUDENCIA)
- spaCy/Stanza may perform comparably on PER/ORG/LOC since these are well-covered by general models
- The comparison justifies the domain-specific model choice and quantifies the value of domain adaptation for NER

---

## Experiment 4: Statistical Significance Testing for Stage 2

**Addresses:** Reviewer issue 2.3 (no significance testing, small dataset)

### Objective
Add bootstrap confidence intervals and statistical tests to determine which performance differences are significant at N=89.

### Design

**Tests to implement:**
1. **Bootstrap 95% CI** on weighted F1 for each (encoder, classifier) combination
   - 1000 bootstrap resamples of the pooled predictions
   - Report [lower, upper] bounds alongside mean F1
2. **McNemar's test** (or paired permutation test) for pairwise comparisons:
   - Legal-BERT LR vs. BERTimbau-large LR (encoder effect)
   - Legal-BERT LR vs. Legal-BERT SVM (classifier effect)
   - Best embedding model vs. LLM baseline (architecture effect)
3. **Increase K** to 5 (minority class n=18 allows 3-4 samples per fold)
   - Re-run all experiments with K=5 for more stable estimates
   - Report both K=3 and K=5 results

### Implementation Plan

Add to `src/framework/stage2_embeddings.py`:

```python
def bootstrap_f1_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for weighted F1."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        scores.append(f1_score(y_true[indices], y_pred[indices], average='weighted'))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(lower), float(np.mean(scores)), float(upper)

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test: are two classifiers significantly different?"""
    # Count discordant pairs
    # a_correct_b_wrong = sum((y_pred_a == y_true) & (y_pred_b != y_true))
    # a_wrong_b_correct = sum((y_pred_a != y_true) & (y_pred_b == y_true))
    # chi2 = (|a-b| - 1)^2 / (a + b)
    pass
```

### Expected Outcome
- Most encoder differences (Legal-BERT vs BERTimbau-base, ~0.12 F1 gap) are likely significant
- Small differences (Legal-BERT vs BERTimbau-large, ~0.03 F1 gap) are likely NOT significant
- The LLM vs embedding gap (~0.31 F1) is overwhelmingly significant
- Bootstrap CIs will be wide (reflecting the small sample), making this explicit

---

## Experiment 5: Regex-Based Segmentation Baseline

**Addresses:** Reviewer issue 3.5 (missing baseline for segmentation)

### Objective
Compare LLM-based segmentation against a simple rule-based approach using keyword/regex matching for section headers.

### Design

**Baseline (regex):**
Brazilian criminal sentences typically contain explicit section markers:
- "RELATÓRIO" / "I - RELATÓRIO" → RELATORIO
- "DOS FATOS" / "II - DOS FATOS" → DOS_FATOS  
- "FUNDAMENTAÇÃO" / "III - FUNDAMENTAÇÃO" → FUNDAMENTACAO
- "DOSIMETRIA" / "DA DOSIMETRIA" → DOSIMETRIA
- "DISPOSITIVO" / "ANTE O EXPOSTO" / "ISTO POSTO" → DISPOSITIVO
- Default → OUTROS

**Treatment:** Existing LLM segmentation results

**Comparison metrics:**
1. Section boundary accuracy (how many section transitions does regex correctly identify?)
2. Sentence-level label agreement (what % of sentences get the same label?)
3. Coverage: does regex handle all 25 documents without failure? (vs. LLM's 32% first-pass failure)
4. Failure cases: which documents have NO explicit section headers?

### Implementation Plan

Create `src/framework/stage3_regex_baseline.py`:

```python
"""Regex-based segmentation baseline for comparison with LLM segmentation."""

SECTION_PATTERNS = {
    "RELATORIO": [r"(?i)\brelat[óo]rio\b", r"(?i)^I[\s\-\.]+"],
    "DOS_FATOS": [r"(?i)\bdos?\s+fatos?\b", r"(?i)^II[\s\-\.]+"],
    "FUNDAMENTACAO": [r"(?i)\bfundamenta[çc][ãa]o\b"],
    "DOSIMETRIA": [r"(?i)\bdosimetria\b", r"(?i)\bda\s+pena\b"],
    "DISPOSITIVO": [r"(?i)\bdispositivo\b", r"(?i)\bante\s+o\s+exposto\b", r"(?i)\bisto\s+posto\b"],
}

def segment_by_regex(text: str) -> list[dict]:
    """Segment document using regex section header detection."""
    pass
```

### Expected Outcome
- Regex likely works well for documents with explicit section headers (common in longer, well-structured sentences)
- Regex likely fails for shorter documents or those without explicit headers
- LLM handles implicit section boundaries better (where section changes without an explicit header)
- The comparison quantifies when a simple baseline suffices vs. when LLM adds value

---

## Execution Priority

1. **Experiment 1** (Segmentation ablation) — highest priority, directly addresses "critical" reviewer concern
2. **Experiment 4** (Statistical tests) — can be done without new model runs, just reprocesses existing predictions
3. **Experiment 2** (LLM NER baseline) — addresses reviewer Question 1 and novelty concern
4. **Experiment 3** (spaCy/Stanza) — medium priority, contextualizes model choice
5. **Experiment 5** (Regex segmentation) — lowest priority, supplements the main arguments

## Dependencies

- Experiment 1 needs: `transformers`, existing NER model, `dataset_normalized_for_ner.csv`
- Experiment 2 needs: Gemini API key, prompt design
- Experiment 3 needs: `spacy` + `pt_core_news_lg` model, `stanza` + Portuguese models
- Experiment 4 needs: only existing predictions (from `output/stage2_embeddings_results_no_punct.json`)
- Experiment 5 needs: existing segmentation results + the NER dataset
