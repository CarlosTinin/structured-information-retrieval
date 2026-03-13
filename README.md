# Organização do código (`src`)

Estrutura modular proposta:

- `framework/io_utils.py`: utilitários de leitura/escrita e detecção de CSV.
- `framework/stage1_1_document_type.py`: subetapa 1.1 (classificação por tipo já conhecido da extração).
- `framework/stage1_2_preprocessing.py`: subetapa 1.2 (normalização de texto jurídico para classificação).
- `framework/stage2_finetune.py`: etapa 2 (fine-tuning de BERT).
- `framework/stage2_embeddings.py`: etapa 2 (embeddings BERT + modelos clássicos).
- `framework/stage5_ner.py`: etapa 5 (extração de entidades - NER).
- `framework/cli.py`: interface única para execução etapa por etapa.

> Nota: a macro-etapa 1 foi dividida em `stage1_1` e `stage1_2`. A antiga etapa 3 foi renumerada para etapa 2. A antiga etapa 4 foi movida para `stage5`, deixando a etapa 4 reservada para inclusão posterior.

## Execução

No diretório raiz do projeto:

1. Instalar dependências:

`pip install -r requirements.txt`

1. Executar uma etapa:

`python -m src.framework.cli stage1_1 --input files/NER/temp/extracted_pdf_text.csv --out-with-type files/NER/temp/extracted_pdf_text_com_tipo.csv --out-sentencas-acordaos files/NER/temp/extracted_pdf_text_sentencas_acordaos.csv`

`python -m src.framework.cli stage1_2 --input files/datasets/dataset_processos_filtrados_por_merito.csv --output files/datasets/dataset_normalizado.csv`

`python -m src.framework.cli stage2-finetune --input files/datasets/dataset_normalizado.csv --output-dir files/results/stage2_finetune`

`python -m src.framework.cli stage2-embeddings --input files/datasets/dataset_normalizado.csv --output-dir files/results/stage2_embeddings`

`python -m src.framework.cli stage5 --input-json files/Documentos-Segmentados/resultado_anotacao_02.json --output-json files/NER/sentencas_com_entidades.json --output-csv files/NER/sentencas_com_entidades.csv`
