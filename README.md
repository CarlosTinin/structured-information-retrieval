# Organização do código (`src`)

Estrutura modular proposta:

- `framework/io_utils.py`: utilitários de leitura/escrita e detecção de CSV.
- `framework/stage1_1_document_type.py`: subetapa 1.1 (classificação por tipo já conhecido da extração).
- `framework/stage1_2_preprocessing.py`: subetapa 1.2 (normalização de texto jurídico para classificação).
- `framework/stage2_finetune.py`: etapa 2 (fine-tuning de BERT).
- `framework/stage2_embeddings.py`: etapa 2 (usa BERT como encoder e classifica mérito penal com modelos clássicos, foco em `condenação`, `extinto` e `absolvição`).
- `framework/stage3_segmentation.py`: etapa 3 (segmentação de sentenças com Gemini, usando prompt em `src/prompts/prompt_segmentation.txt`).
- `framework/stage5_ner.py`: etapa 5 (extração de entidades - NER).
- `framework/cli.py`: interface única para execução etapa por etapa.

> Nota: a macro-etapa 1 foi dividida em `stage1_1` e `stage1_2`. A antiga etapa 3 foi renumerada para etapa 2. A antiga etapa 4 foi movida para `stage5`, deixando a etapa 4 reservada para inclusão posterior.

## Execução

No diretório raiz do projeto:

1. Instalar dependências:

`pip install -r requirements.txt`

1. Executar uma etapa:

`python -m src.framework.cli stage1_1 --input files/datasets/dataset_completo.csv --output files/output/dataset_filtered_by_type.csv`

`python -m src.framework.cli stage1_2 --input files/output/dataset_filtered_by_type.csv --output-classification files/output/dataset_normalized.csv --output-ner files/output/dataset_normalized_for_ner.csv`

`python -m src.framework.cli stage2-finetune --input files/output/dataset_normalized.csv --output-dir files/results/stage2_finetune`

`python -m src.framework.cli stage2-embeddings --input files/output/dataset_normalized.csv --output-root output`

Saídas da etapa `stage2-embeddings`: tabela LaTeX em `output/tables/table.tex` e matrizes de confusão por modelo em `output/images/matriz_confusao_*modelo*.png`.

`python -m src.framework.cli stage3-segmentation --input files/output/dataset_normalized_for_ner.csv --prompt-file src/prompts/prompt_segmentation.txt --output-json files/Documentos-Segmentados/resultado_anotacao.json`

Observação da etapa `stage3-segmentation`: cada linha do dataset é enviada para o Gemini com o prompt-base e o resultado consolidado é salvo em JSON.
Autenticação Gemini: a chave pode ser lida de `GEMINI_API_KEY` no ambiente ou automaticamente do arquivo `.env` na raiz do projeto.

`python -m src.framework.cli stage5 --input-json files/Documentos-Segmentados/resultado_anotacao_02.json --output-json files/NER/sentencas_com_entidades.json --output-csv files/NER/sentencas_com_entidades.csv`
