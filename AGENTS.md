# A Multi-Layer Document Filtering and Structured Information Extraction Framework for Criminal Labor-Law Decisions

Este repositório é um trabalho em andamento de mestrado que deverá ser publicado em um dos seguintes journals:

- Information Processing & Management
- Neurocomputing
- Knowledge-Based Systems
- Expert Systems with Applications

Gostaria que o rigor técnico e científico do trabalho fosse digno de revistas de computação de algo nível (Qualis A1-A2)

## Resumo breve do trabalho

Realizar correlação indireta entre empresas na cadeia de suprimentos no brasil ainda é um desafio complexo.
Em cenários que queremos investigar relação de empresas que cometaram algum crime com outras empresas na cadeia de suprimentos, não conseguimos uma base de dados de transações entre elas.
Se não conseguirmos provas concretas de relação não há a possiblidade de conectar essas empresas.
Uma alternativa para a construção seria obter informações disponíveis sobre as empresas de múltiplas fontes e realizar conexões com os dados obtidos.
Para isso, nesse trabalho propõe-se elaborar uma pipeline de processamento de dados jurídicos utilizando modelos de IA para obter dados das empresas julgadas.
Dessa forma, foi sugerido a análise de alguns processos jurídicos que contem decisões judicias sobre casos de escravidão moderna na cadeia de suprimentos.
O objetivo da tarefa é analisar automaticamente decisões judiciais utilizando técnicas de NER (Named Entity Recognition) para identificar detalhes sobre crime, a pena, o local, os envolvidos, etc.
Esses dados poderão complementar aqueles extraídos de outras fontes como a lista suja.

## Base de dados

A base de dados conta com 2581 documentos jurídicos.
Dentre eles, nem todos são decisões/sentenças, tem-se despachos, atos ordinários e atas de audiência. Portanto, é necessário uma **análise em múltiplas camadas** antes de extrair as entidades nomeadas ou o modelo irá errar muito.
O interesse dessa extração é exclusivo de **sentenças**, porque é garantido que o réu cometeu o crime e de qualquer forma, qualquer outro tipo de documento não contem as informações relevantes para o estudo.

## Metodologia

A base de dados obtida contém tipos de documentos que não são relevantes para o experimento. Portanto, é necessário realizar uma filtragem em múltiplas camadas para obter os documentos relevantes para a extração de informações.

![Pipeline de atividades](images/activity_diagram.png)

A Figura acima é um diagrama de atividades que ilustra a pipeline completa da tarefa.

### Classificação por tipo de documento

A etapa inicial será de classificação dos documentos por tipo e extrair apenas os que contem decisões judiciais (a princípio, sentenças e acórdãos).
Em seguida, para melhorar a performance do modelo é desejável a segmentar os documentos nas partes alvo para isolar as partes na qual o modelo deverá procurar por informações específicas. Penas só aparecem em **sentenças e acórdãos**, nosso interesse inicial se restringe a esses grupos (minoria na base de dados).
Acódãos serão descosiderados para a etapa de classificação de mérito penal, porque eles não trazem informações relevantes para o estudo, e geralmente são decisões de recursos, ou seja, reavaliação de uma sentença, portanto, não trazem informações adicionais.
No entanto, dentre os documentos de sentenças, podem haver aquelas com mérito penal, faz-se necessário aprofundar nessa discussão.
Para fazer isso da melhor forma, seria necessário um **especialista analisar manualmente cada caso**, ou pelo menos um **conjunto de casos** que possam servir de treinamento para um **modelo transformer**.

No código da pipeline, esta macro-etapa foi consolidada em duas subetapas sequenciais: **stage1_1** (classificação/normalização do tipo documental já conhecido na extração, como sentença, decisão, acórdão e despacho, com filtragem inicial de interesse) e **stage1_2** (pré-processamento textual dos documentos filtrados para alimentar os classificadores de mérito).

Essa organização evita duplicidade entre "etapa 1" e "etapa 2" quando a tipologia documental já está disponível no dado bruto.

Resultando em 99 sentenças.

### Filtragem por decisão por mérito penal

Esta etapa do trabalho consiste na identificação e filtragem de documentos que tratam especificamente de decisões de mérito penal no contexto do trabalho análogo à escravidão. Para fins desta análise, considera-se **condenação penal** todo documento que examine a responsabilidade penal do réu, com enquadramento no **artigo 149 do Código Penal**, independentemente do desfecho do processo.
Nesse sentido, foram incluídos documentos cujas decisões resultaram em condenação, absolvição, indeferimento ou extinção da punibilidade.
-> Sentenças e acórdãos sobre recursos (embargos de declaração) serão desconsiderados da análise atual como mencionado anteriormente.
O recurso poderá ser aberto contra uma condenação ou contra uma absolvição. O objetivo aqui é obter informações de condenações, recursos re-avaliam as informações, geralmente não trazem informações adicionais, portanto, é sensato removê-los da análise.
-> Observação quanto a casos de condenação parcial. Na minha opinião, quando houve a condenação de alguma das partes envolvidas com base no artigo 149, entraria como condenação. Para o treinamento do modelo de classificação de mérito penal, serão consideradas apenas três categorias (absolvição, condenação e extintos). As demais podem ser removidas nas etapas de pré-processamento.

### Segmentação de sentenças

Essa etapa foi realizada para determinar os segmentos dos documentos jurídicos e foi realizada utilizando a seguinte sequencia: criação do gold standard, anotação utilizando uma llm (Gemini Pro 2.5) E a avaliação humana após a análise da llm.

### NER

Foi realizada a aplicação de um modelo NER para realizar a obtenção das entidades dos documentos, que então seriam exportadas e armazenadas num banco de dados json.

## Estrutura de pastas

Na pasta src estão os códigos utilizados para realizar as atividades descritas acima, como a classificação por tipo de documento, a filtragem por mérito penal, a segmentação de sentenças e a aplicação do modelo NER.

Na implementação atual, a primeira macro-etapa aparece como `stage1_1` e `stage1_2`, para refletir melhor o encadeamento do diagrama de atividades e a origem das informações de tipo documental.
Na versão atual do código, as rotinas de classificação anteriormente na etapa 3 foram renumeradas para `stage2` (com variantes de abordagem), e a extração NER foi deslocada para `stage5`, mantendo `stage4` reservado para inclusão posterior.

Na pasta paper estão os arquivos relacionados à escrita do artigo, como o template em latex, o arquivo .bib com as referências e o arquivo .tex com o texto do artigo utilizando o template da primeira revista mencionada a Information Processing & Management.

Na pasta files estão os arquivos relacionados à base de dados, como o arquivo .csv com os documentos jurídicos e o arquivo .json com as entidades extraídas utilizando o modelo NER:
    - files/datasets: dataset_completo.csv contendo os 2581 documentos jurídicos, com as seguintes colunas.
    - files/docs-condenacao: contendo os 25 documentos jurídicos classificados como condenação.
    - files/Documentos-Segmentados: contem os documentos de condenação segmentados pelo tipo, resultado da etapa de segmentação de sentenças.
    - files/NER: contendo os arquivos .json com as entidades extraídas utilizando o modelo NER.

## Tarefas pendentes

1. Organizar e realizar a limpeza do código para publicação. O esboço inicial do código foi feito usando o google colab, precisa ser convertido para uma estrutura de projeto mais organizada, com arquivos .py e uma estrutura de pastas mais clara.

2. Realizar a escrita do artigo utilizando o template da revista escolhida, inicialmente a Information Processing & Management.
    - Determinar a estrutura do artigo, quais seções serão utilizadas, quais informações serão apresentadas em cada seção, etc.
    - Escrever o texto do artigo, utilizando as informações obtidas nas etapas anteriores, como a metodologia, os resultados, a discussão, etc.
    - Realizar a revisão do texto para garantir que o rigor técnico e científico do trabalho seja digno de revistas de computação de algo nível (Qualis A1-A2).

3. Realizar um estudo comparativo com pelo menos três artigos semelhantes para comparar os resultados obtidos com os resultados de outros trabalhos na área, para destacar as contribuições do trabalho e identificar possíveis limitações.

## Ferramentas utilizadas

python: para produção do código
latex: para escrita do artigo
scopus: para obtenção dos artigos relacionados ao tema do trabalho

### Apendice

#### Query do Scopus para obtenção dos artigos com atividades semelhantes

query textual:
"machine learning" OR "learning of machine" OR "NLP" OR "natural language processing" OR "NER" OR "Named entity recognition"

AND

"court decision" OR "legal documents" OR "legal decisions"

AND

framework

critérios de busca:

- 2021 - 2026
- em ingles
- ordenados por quantidade de citações
- artigos de periódico
- remover artigos com < 50% de percentil ou < 100 citações

obtive 45 artigos

Especifiquei mais por NER, dado que quero comparar com minha abordagem obtive 3 artigos para análise.
