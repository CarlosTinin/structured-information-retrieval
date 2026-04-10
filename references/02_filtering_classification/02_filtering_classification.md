# 02 — Filtering and Classification of Legal Documents

---

## 02_01 — LegalVis: Exploring and Inferring Precedent Citations in Legal Documents

**Authors:** Lucas E. Resck, Jean R. Ponciano, Luis Gustavo Nonato, Jorge Poco  
**Journal:** IEEE Transactions on Visualization and Computer Graphics (2023), vol. 29, no. 6, pp. 3105–3120  
**DOI:** https://doi.org/10.1109/TVCG.2022.3152450

### Abstract

To reduce the number of pending cases and conflicting rulings in the Brazilian Judiciary, the National Congress amended the Constitution, allowing the Brazilian Supreme Court (STF) to create binding precedents (BPs), i.e., a set of understandings that both Executive and lower Judiciary branches must follow. The STF's justices frequently cite the 58 existing BPs in their decisions, and it is of primary relevance that judicial experts could identify and analyze such citations. To assist in this problem, we propose LegalVis, a web-based visual analytics system designed to support the analysis of legal documents that cite or could potentially cite a BP. We model the problem of identifying potential citations (i.e., non-explicit) as a classification problem. However, a simple score is not enough to explain the results; that is why we use an interpretability machine learning method to explain the reason behind each identified citation. For a compelling visual exploration of documents and BPs, LegalVis comprises three interactive visual components: the first presents an overview of the data showing temporal patterns, the second allows filtering and grouping relevant documents by topic, and the last one shows a document's text aiming to interpret the model's output by pointing out which paragraphs are likely to mention the BP, even if not explicitly specified. We evaluated our identification model and obtained an accuracy of 96%; we also made a quantitative and qualitative analysis of the results. The usefulness and effectiveness of LegalVis were evaluated through two usage scenarios and feedback from six domain experts.

### Introduction

The Brazilian Supreme Court (STF) is the highest court of law in Brazil, primarily responsible for guarding the rights in the Brazilian Constitution. In 2004, Brazil had more than 100 million pending cases in the Judiciary. To reduce those numbers and avoid conflicting decisions, the National Congress amended the Constitution, allowing the STF to create binding precedents. From 2004 to 2020, the Court ruled 58 of those precedents. STF's justices regularly cite these binding precedents in their decisions.

Given the large number of decisions in STF — more than 1 million from 2011 to 2020 — lawyers and other judicial experts face difficulties finding citations to BPs and analyzing them. Such difficulties arise from the lack of a computational tool to help the experts to find decisions that cite a particular BP of interest. Moreover, it is usual to have decisions with hundreds of pages, making it hard for experts to find parts where justices cite, quote, or implicitly mention a BP.

LegalVis was designed to explore and analyze the texts of STF's legal documents, particularly their explicit or implicit relationships with binding precedents. In contrast to an explicit citation to a BP, which can be easily found by searching the document's content, finding potential citations to a BP is not a straightforward task. To tackle this issue, LegalVis relies on a machine learning model to identify potential citations, building upon an interpretability mechanism to provide reliable explanations for the model's decisions.

### Conclusion

In this paper, the authors presented LegalVis, a web-based visual analytic system designed to assist lawyers and other judicial experts in analyzing legal documents that cite or could potentially cite binding precedents. LegalVis first identifies potential citations by implementing a simple yet powerful machine learning model based on classification. Then, an interpretability mechanism also incorporated into the system provides reliable explanations for the model's decisions.

Finally, all this information becomes accessible through the three interactive and linked views that compose the system. Qualitative and quantitative analyses validated the performance of the proposed model and two usage scenarios demonstrated the usefulness and effectiveness of the LegalVis system. Both scenarios were validated by six domain experts not involved in the system's development, who also reported positive feedback.

---

## 02_02 — Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models, and Challenges

**Authors:** Farid Ariai, Joel Mackenzie, Gianluca Demartini  
**Journal:** ACM Computing Surveys (2025), vol. 58, no. 6, Article 163  
**DOI:** https://doi.org/10.1145/3777009

### Abstract

Natural Language Processing (NLP) is revolutionising the way both professionals and laypersons operate in the legal field. The considerable potential for NLP in the legal sector, especially in developing computational assistance tools for various legal processes, has captured the interest of researchers for years. This survey follows the Preferred Reporting Items for Systematic Reviews and Meta-Analyses framework, reviewing 154 studies, with a final selection of 131 after manual filtering. It explores foundational concepts related to NLP in the legal domain, illustrating the unique aspects and challenges of processing legal texts, such as extensive document lengths, complex language, and limited open legal datasets. We provide an overview of NLP tasks specific to legal text, such as Document Summarisation, Named Entity Recognition, Question Answering, Argument Mining, Text Classification, and Judgment Prediction. Furthermore, we analyse both developed legal-oriented language models, and approaches for adapting general-purpose language models to the legal domain. Additionally, we identify sixteen open research challenges, including the detection and mitigation of bias in artificial intelligence applications, the need for more robust and interpretable models, and improving explainability to handle the complexities of legal language and reasoning.

### Introduction

Advancements in Natural Language Processing (NLP) have significantly impacted the legal domain by simplifying complex tasks, such as Legal Document Summarisation (LDS), Legal Argument Mining (LAM), enhancing legal text comprehension for laypersons, and improving Legal Question Answering (LQA) and Legal Judgment Prediction (LJP). These improvements are primarily attributed to advancements in Neural Network (NN) architectures, such as transformer models. NLP techniques now enable machines to generate text, answer legal questions, draft regulations, and simulate legal reasoning, which have the potential to revolutionise legal practices. Applications such as contract review and case prediction have been automated to a large extent, speeding up processes, reducing human error, and cutting operational costs.

Despite these advantages, the integration of NLP in the legal domain is not without challenges, especially in terms of fairness, bias, and explainability issues. The use of AI in legal applications must follow strict standards of accuracy, fairness, and transparency, given the potential impact on clients' lives and rights. Nonetheless, Large Language Models (LLMs) have demonstrated potential to enhance the efficiency, fairness, and precision of legal tasks.

This survey article explores the current landscape of NLP applications within the legal domain, discussing its potential benefits and the practical challenges it poses. It provides a comprehensive overview of the field, categorising research into several areas: LQA, LJP, Legal Text Classification (LTC), LDS, legal Named Entity Recognition (NER), LAM, legal corpora and legal Language Models. Notably, there is comparatively less research in NER and legal corpora, whereas LDS and LQA have seen extensive research activity.

### Conclusion

Advances in AI and NLP have improved legal NLP techniques and models, reducing the difficulty of engaging in legal processes for laypersons, and easing workloads and manual labour for professionals. This survey provides a comprehensive overview of the advancements in NLP techniques used in the legal domain, paying special attention to the unique characteristics of legal documents. Existing datasets and LLMs tailored for the legal domain were also reviewed.

Legal NER research spans multiple languages and utilises diverse methods, from rule-based to BERT-based models. LDS has largely focused on extractive and abstractive methods, ranging from TF-IDF to transformer-based models. LAM now automates the detection of claims, premises, and their links through domain-specific annotation schemes and graph-based or residual-network approaches, supporting legal reasoning tasks such as conflict resolution. In LTC, multi-class classification dominates, with DL architectures such as CNNs and Bi-LSTMs widely used. LJP primarily focuses on Chinese datasets with DL approaches such as CNNs. LQA often leverages IR techniques such as BM25, with a significant focus on statutory law.

Key open research challenges include the need for domain-specific fine-tuning strategies, addressing bias and fairness in legal datasets, and the importance of interpretability and explainability. Other challenges include the development of more robust pre-processing techniques, handling multilingual capabilities, and integrating ontology-based methods for more accurate legal reasoning.

---

## 02_03 — A systematical procedure to extracting legal entities from Indonesian judicial decisions

**Authors:** Eka Qadri Nuranti, Naili Suri Intizhami, Evi Yulianti, A. Muh. Iqbal Latief, Osama Iyad Al Ghozy  
**Journal:** MethodsX (2025), vol. 15, Article 103610  
**DOI:** https://doi.org/10.1016/j.mex.2025.103610

### Abstract

This article presents a systematic method of extracting legal entities from Indonesian judicial decisions with a well-structured named entity recognition (NER) approach. The procedure was implemented by gathering and annotating court decisions for theft cases at three court levels: first instance (2478 files), appeal (147 files), and cassation (62 files), amounting to 2687 annotated files. The data were harvested from the official website of the Supreme Court of the Republic of Indonesia using automated web scraping, followed by manual filtering for relevance and completeness.

Manual annotation was performed with the Label Studio platform by three independent annotators. Annotation consistency was considered using Fleiss' Kappa, yielding an average agreement score of 0.705 across all levels, indicating good inter-annotator reliability. The method uses a hierarchical structure and a BIO tagging scheme to tag >50 types of legal entities, including defendants, judges, legal articles, charges, and verdict decisions.

This approach is proper for text processes such as legal information extraction, classification, and legal analysis. From a legal perspective, this process will improve legal transparency and research on Indonesian judicial data.

### Introduction (Background)

The method outlined in this article aims to bridge a gap of particular concern in legal Natural Language Processing by developing a reproducible pipeline to retrieve legal entities from Indonesian court decisions. With legal documents increasingly accessible via the official website of the Supreme Court of the Republic of Indonesia, researchers can utilize rich, real-world data to reflect judicial processes. This open access is a significant step towards facilitating legal transparency and enabling computational legal analysis.

While earlier research has initiated legal NLP for Indonesian documents, most activities have been on limited domains. For example, prior work introduced an annotated dataset of 150 theft case decisions for rule-based entity recognition; subsequent research used a corpus of 1003 general criminal judgments with neural models such as BiLSTM+CRF; and more recent work applied large language models such as XLM-R and IndoRoBERTa on 993 labeled decisions for verdict-related entity recognition. These works have set foundations for Indonesian legal NLP but tend to deal with one-level cases (usually first instance) and are yet to be comprehensive across the judicial hierarchy.

To address these constraints, the approach devised in this study establishes an end-to-end pipeline spanning multiple judicial tiers — initial trial, appellate review, and cassation — and incorporates elements such as manual annotation, inter-annotator consistency evaluation, and strict transformation of outputs. The approach targets theft cases to ensure thematic consistency while incorporating a hierarchical judicial system within the annotation pipeline. The goal is to create an annotated corpus and provide a reproducible and scalable approach to curating and preparing legal decision text for NER tasks.

### Conclusion (Method Validation)

The manual annotation process resulted in the creation of the IDTheftCase-JudgmentCorpus, a rigorously structured dataset rich in legal information. The corpus contains 2687 documents relating to court judgments of three levels of the judiciary, thereby providing a representative and valuable tool to support research in NER, legal text analysis, and large-scale applications in NLP. The annotation quality was maintained through a rigorous process and then verified through Fleiss' Kappa, which found high inter-annotator agreement (averaging 0.705 across all court levels).

Inter-annotator agreement was consistently in the "Good" category across all three judicial levels. At the first instance level, the average Fleiss' Kappa was 0.746; at the appeal level, 0.671; and at the cassation level, 0.698. Some individual documents achieved "Very Good" scores (>0.81).

This dataset greatly expands available resources for legal natural language processing in the Indonesian context. It presents new opportunities to explore and harvest judicial documents for higher-order legal systems systematically. The annotation approach — using BIO tagging with >50 entity types and a hierarchical structure — can be reused, modified, and applied to other case types or jurisdictions, assisting in a broader endeavor to construct AI-ready legal corpora for low-resource linguistic settings such as Bahasa Indonesia.
