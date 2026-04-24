# Organização do código (`src`)

Estrutura modular proposta:

- `framework/io_utils.py`: utilitários de leitura/escrita e detecção de CSV.
- `framework/stage1_1_document_type.py`: subetapa 1.1 (classificação por tipo já conhecido da extração).
- `framework/stage1_2_preprocessing.py`: subetapa 1.2 (normalização de texto jurídico para classificação).
- `framework/stage2_embeddings.py`: etapa 2 (usa BERT como encoder e classifica mérito penal com modelos clássicos, foco em `condenação`, `extinto` e `absolvição`).
- `framework/stage3_segmentation.py`: etapa 3 (segmentação de sentenças com Gemini, usando prompt em `src/prompts/prompt_segmentation.txt`).
- `framework/stage4_ner.py`: etapa 4 (extração de entidades - NER, etapa final).
- `framework/cli.py`: interface única para execução etapa por etapa.

> Nota: a macro-etapa 1 foi dividida em `stage1_1` e `stage1_2`. A antiga etapa 3 foi renumerada para etapa 2. A segmentação permanece em `stage3` e a NER é a etapa final em `stage4`.
> Decisão metodológica atual: o estágio de fine-tuning foi removido da pipeline devido ao baixo número de amostras rotuladas, que gerou alta variância e desempenho inferior ao baseline com BERT como encoder + classificadores clássicos.

## Execução

No diretório raiz do projeto:

1. Instalar dependências:

`pip install -r requirements.txt`

1. Executar uma etapa:

`python -m src.framework.cli stage1_1 --input files/datasets/dataset_completo.csv --output files/output/dataset_filtered_by_type.csv`

`python -m src.framework.cli stage1_2 --input files/output/dataset_filtered_by_type.csv --output-classification files/output/dataset_normalized.csv --output-ner files/output/dataset_normalized_for_ner.csv`

`python -m src.framework.cli stage2-embeddings --input files/output/dataset_normalized.csv --output-root output`

Saídas da etapa `stage2-embeddings`: tabela LaTeX em `output/tables/table.tex` e matrizes de confusão por modelo em `output/images/matriz_confusao_*modelo*.png`.

`python -m src.framework.cli stage3-segmentation --input files/output/dataset_normalized_for_ner.csv --prompt-file src/prompts/prompt_segmentation.txt --output-json files/Documentos-Segmentados/resultado_anotacao.json`

Observação da etapa `stage3-segmentation`: cada linha do dataset é enviada para o Gemini com o prompt-base e o resultado consolidado é salvo em JSON.
Autenticação Gemini: a chave pode ser lida de `GEMINI_API_KEY` no ambiente ou automaticamente do arquivo `.env` na raiz do projeto.
Por padrão, a etapa filtra apenas `decisao=condenação` antes de enviar para a LLM (para processar o subconjunto de interesse, ~25 documentos). Isso pode ser alterado por `--filter-label-column` e `--filter-label-value`.

`python -m src.framework.cli stage4 --input-json files/Documentos-Segmentados/resultado_anotacao_02.json --output-json files/NER/sentencas_com_entidades.json --output-csv files/NER/sentencas_com_entidades.csv`

## Possíveis melhorias

1. Fine-tune Stage2 encoder with active learning
Gap identified from: Ariai et al. 02\_02, Bellandi et al. 01\_01, Bhattacharya et al. 03\_05

All three works emphasize that domain-specific fine-tuning consistently outperforms embedding+classifier pipelines when any annotated data is available. Currently, Stage2 uses frozen BERT embeddings because the labeled set (~99 sentences, 25 convictions) is too small for stable fine-tuning. An active learning loop — starting from the current classifier to select the most informative unlabeled documents for expert annotation — could double the labeled set within 2–3 annotation rounds and enable proper fine-tuning. This is the single change most likely to push the F1 above the 0.65 ceiling observed in Table1.

---

1. Add inter-annotator agreement (IAA) metrics for segmentation
Gap identified from: Nuranti et al. 02\_03, Bhattacharya et al. 03\_05

Both papers explicitly measure and report Fleiss' Kappa or similar IAA scores. The current Stage3 segmentation uses a single LLM without any human cross-validation step and without a formal IAA measure. Adding even a second human rater for 10–15% of the segmented documents and computing Cohen's Kappa per section type would substantially strengthen the credibility of the segmentation quality claims for a top-tier journal.

---

1. Implement a schema-validated gold standard for segmentation and NER

Gap identified from: Bhattacharya et al. 03\_05, Nuranti et al. 02\_03, Ragazzi et al. 03\_04
LAWSUIT and DeepRhole both provide expert-annotated gold standards that allow model outputs to be evaluated against human judgments. The present pipeline currently lacks a gold standard for the segmentation and NER stages. Even annotating 15–20 documents from the conviction subset (with one legal expert) would allow computation of section-level F1 for segmentation and entity-level precision/recall for NER — both critical for a Results section that reviewers of Expert Systems with Applications or Knowledge-Based Systems will expect.

---

1. Evaluate the pipeline end-to-end against a no-filtering baseline

Gap identified from: The comparative synthesis in the Related Work table
The paper's main selling point — "multi-layer filtering improves downstream extraction quality" — is currently asserted but not measured. A controlled ablation comparing NER extraction quality (entity precision/recall) on: (a) the full 2,581-document corpus without filtering vs. (b) the 99-document filtered set vs. (c) the 25-document conviction subset would directly answer RQ1 and make the paper's core claim empirically defensible rather than intuitive.

---

1. Add explainability output to Stage2 classification

Gap identified from: Resck et al. 02\_01, Ariai et al. 02\_02
LegalVis demonstrates that legal professionals require explainability alongside prediction scores. A simple SHAP or LIME layer on top of the SVM/XGBoost classifiers would: (i) provide interpretable feature importance for the most informative tokens in merit classification; (ii) strengthen the Discussion section with examples of which document fragments drive classification decisions; and (iii) directly address the "explainability" open challenge identified by the survey 02\_02, increasing the paper's contribution profile.

---

1. Benchmark against a Portuguese-law LLM baseline for Stage2

Gap identified from: Ariai et al. 02\_02, Chen et al. 03\_06

The survey notes that LLMs are being increasingly used for zero-shot and few-shot legal classification. Adding a GPT-4o or Gemini Pro 2.5 zero-shot baseline for penal-merit classification (with a carefully designed prompt stating the Art.~149 framing and the three target classes) would contextualize the Stage2 results against the most capable available models — and the expected finding that the BERT+SVM pipeline is more stable under class imbalance would strengthen the methodological argument for the embedding approach.

---

1. Report pipeline runtime and LLM cost profile

Gap identified from: Chen et al. 03\_06, Guha et al. 03\_08
Both papers providing production-scale results (CoTHSSum and the TI DSS system) discuss computational overhead explicitly. Reviewers of applied AI journals will expect: total Stage3 LLM API cost (number of tokens × price), average latency per document, and proportion of documents requiring retry. This is already a placeholder in the Experimental Setup section but should be treated as a first-class result, not a footnote