# Organização do código (`src`)

Estrutura modular proposta:

- `framework/io_utils.py`: utilitários de leitura/escrita e detecção de CSV.
- `framework/stage1_document_type.py`: classificação por tipo de documento.
- `framework/stage2_preprocessing.py`: normalização de texto jurídico.
- `framework/stage3_finetune.py`: fine-tuning de BERT.
- `framework/stage3_embeddings.py`: embeddings BERT + modelos clássicos.
- `framework/stage4_ner.py`: extração de entidades (NER).
- `framework/cli.py`: interface única para execução etapa por etapa.

## Execução

No diretório raiz do projeto:

1. Instalar dependências:

`pip install -r requirements.txt`

1. Executar uma etapa:

`python -m src.framework.cli stage1 --input files/NER/temp/extracted_pdf_text.csv --out-with-type files/NER/temp/extracted_pdf_text_com_tipo.csv --out-sentencas-acordaos files/NER/temp/extracted_pdf_text_sentencas_acordaos.csv`

`python -m src.framework.cli stage2 --input files/datasets/dataset_processos_filtrados_por_merito.csv --output files/datasets/dataset_normalizado.csv`

`python -m src.framework.cli stage3-finetune --input files/datasets/dataset_normalizado.csv --output-dir files/results/stage3_finetune`

`python -m src.framework.cli stage3-embeddings --input files/datasets/dataset_normalizado.csv --output-dir files/results/stage3_embeddings`

`python -m src.framework.cli stage4 --input-json files/Documentos-Segmentados/resultado_anotacao_02.json --output-json files/NER/sentencas_com_entidades.json --output-csv files/NER/sentencas_com_entidades.csv`
