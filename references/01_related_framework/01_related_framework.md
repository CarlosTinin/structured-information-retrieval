# 01 — Related Frameworks

---

## 01_01 — An entity-centric approach to manage court judgments based on Natural Language Processing

**Authors:** Valerio Bellandi, Christian Bernasconi, Fausto Lodi, Matteo Palmonari, Riccardo Pozzi, Marco Ripamonti, Stefano Siccardi  
**Journal:** Computer Law & Security Review (2024), vol. 52, 105904  
**DOI:** https://doi.org/10.1016/j.clsr.2024.105904

### Abstract

In this paper, we present an entity-centric infrastructure to manage legal documents, especially court judgments, based on the organization of a textual document repository and on the annotation of these documents to serve a variety of downstream tasks. Documents are pre-processed and then iteratively annotated using a set of NLP services that combine complementary approaches based on machine learning and syntactic rules. We present a framework that has been designed to be developed and maintained in a sustainable way, allowing for multiple services and uses of the annotated document repository and considering the scarcity of annotated data as an intrinsic challenge for its development. The design activity is the result of a cooperative project where a scientific team, institutional bodies, and companies appointed to implement the final system are involved in co-design activities. We describe experiments to demonstrate the feasibility of the solution and discuss the main challenges to scaling the system at a national level. In particular, we report the results we obtained in annotating data with different low-resource methods and with solutions designed to combine these approaches in a meaningful way. An essential aspect of the proposed solution is a human-in-the-loop approach to control the output of the annotation algorithms in agreement with the organizational processes in place in Italian courts. Based on these results we advocate for the feasibility of the proposed approach and discuss the challenges that must be addressed to ensure the scalability and robustness of the proposed solution.

### Introduction

Legal documents, especially court judgments and similar resolutions such as orders, contain information that is valuable in multiple applications (from legal case retrieval to legal process analysis), for different purposes (from the comparison with similar cases for uniform judgments to discovering trends for specific legal subjects), and stakeholders (from judges to law administrators, lawmakers, lawyers, scholars, and the general public). Recent advances in data management, Natural Language Processing (NLP), and Machine Learning (ML) are dramatically transforming legal text processing, promising more powerful or completely novel functionalities in legal applications.

References to entities such as organizations, persons, locations, dates and money play an important role in these documents: the extraction, consolidation and storage of these references in the form of metadata and annotations can support many current and future applications. For example, considering entities in a faceted search application can help users retrieve legal cases they are interested in or locate specific entities in long judgments. Solutions to extract information from legal documents have a long tradition and are attracting increased interest.

### Conclusion

In this paper, the authors proposed an entity-centric framework designed for the effective management of legal documents, particularly court judgments. This framework revolves around structuring a repository of textual documents and enhancing their utility through meticulous annotation. These annotations cater to a diverse range of subsequent tasks. The documents undergo preliminary processing before undergoing iterative annotation. This annotation process is facilitated by a set of NLP services that synergistically combine machine learning and rule-based strategies to ensure comprehensive coverage and accuracy. The framework is designed to be developed and maintained in a sustainable way, allowing for multiple services and uses of the annotated document repository. The scarcity of annotated data was considered an intrinsic challenge for its development.

In the second part of the paper, experiments were described to demonstrate the feasibility of the solution and the main challenges to scaling the system at a national level. An essential aspect of the proposed solution is a human-in-the-loop approach to control the output of the annotation algorithms in agreement with the organizational processes in place in Italian courts. Based on the results, the feasibility of the proposed approach was advocated and the challenges that must be addressed to ensure the scalability and robustness of the proposed solution were discussed.

Future work plans to further explore the human-in-the-loop integration by incorporating mechanisms that learn from user feedback. The system can be adapted to handle legal documents from different countries and languages with minor adjustments — only the Service Systems need replacement, as they employ language-specific models and rules, and the Entity Registry metamodel should be reviewed to accommodate country-specific codes and conventions.

---

## 01_02 — Information extraction framework to build legislation network

**Authors:** Neda Sakhaee, Mark C. Wilson  
**Journal:** Artificial Intelligence and Law (2021), vol. 29, pp. 35–58  
**DOI:** https://doi.org/10.1007/s10506-020-09263-3

### Abstract

This paper concerns an information extraction process for building a dynamic legislation network from legal documents. Unlike supervised learning approaches which require additional calculations, the idea here is to apply information extraction methodologies by identifying distinct expressions in legal text in order to extract network information. The study highlights the importance of data accuracy in network analysis and improves approximate string matching techniques to produce reliable network data-sets with more than 98% precision and recall. The applications and the complexity of the created dynamic legislation network are also discussed and challenged.

### Introduction

The process of legal text retrieval for machine-readable structured documents is well explored in the literature. Prior work includes computer-based search-and-retrieval strategies that automated the process of finding citations in HTML files of U.S. Supreme Court cases, and context-free grammar approaches to find references in Dutch legal sources. Graph-based linking and visualisation processes for legal texts based on parsing structured or semi-structured XML files have also been proposed.

The main contribution of this study is the proposed information extraction framework which engages several processes and enables researchers to have access to the network information from historical documents. This framework makes it possible to study legislation networks as dynamic graphs. The case study covers all Acts in the New Zealand legislation corpus including historical, expired, repealed and consolidated Acts as at the end of September 2018 — a set of 23,870 PDF files of which about 87% are in scanned image format. The proposed framework suggests a high-performance procedure to derive network information from such poor quality documents.

### Conclusion

This study focused on time as an essential attribute in understanding and analyzing legislation. Legislation network has been discussed in recent years, but the importance of having access to the historic legislation was never discussed much. This paper underlined the value of studying legislation as dynamic networks and proposed a new information extraction process to achieve a highly accurate legislation network. The performance of the data extraction framework is examined, compared to previous studies and proved to be considerably high. This work contributed to the literature of network information extraction from old documents and insisted on the value and applications of the dynamic legislation network. The proposed process can be used not only in the legal domain but also in various research areas involving documented knowledge, facts, and cases.

Analyzing a dynamic legislation network is a novel approach to understand the underlying process behind the generation of the laws, and to study the behaviour, culture and growth of societies. Future directions include exploring growth behaviour to model network evolution mathematically, studying the betweenness measure and the impact of removing hubs, and comparing legislation networks of different jurisdictions.

---

## 01_03 — Design of Knowledge Graph Retrieval System for Legal and Regulatory Framework of Multilevel Latent Semantic Indexing

**Authors:** Guicun Zhu, Meihui Hao, Changlong Zheng, Linlin Wang  
**Journal:** Computational Intelligence and Neuroscience (2022), Article ID 6781043  
**DOI:** https://doi.org/10.1155/2022/6781043

### Abstract

Latent semantic analysis (LSA) is a natural language statistical model considered as a method to acquire, generalize, and represent knowledge. Compared with other retrieval models based on concept dictionaries or concept networks, the retrieval model based on LSA has the advantages of strong computability and less human participation. LSA establishes a latent semantic space through truncated singular value decomposition. This paper designs the system architecture of the public prosecutorial knowledge graph. Combining graph data storage technology and the characteristics of the public domain ontology, a knowledge graph storage method is designed. By building a prototype system, the functions of knowledge management, knowledge query, and knowledge push are realized. A named entity recognition method based on bidirectional long-short-term memory (bi-LSTM) combined with conditional random field (CRF) is proposed. The experimental results show that the LSTM-entity-context method proposed in this paper improves the representation ability of text semantics compared with other algorithms. The knowledge graph of legal documents of theft cases based on ontology can be updated and maintained in real time.

### Introduction

With the rapid development of Chinese society and the continuous improvement of the legal system, the number of cases heard by courts across the country has increased year by year, and judges accept more than 100 cases per year on average. Under multiple constraints, the work pressure on case investigators is increasing day by day. "Smart Court" is a concept proposed by the Supreme People's Court that mainly uses modern artificial intelligence technology, combines judicial laws with various reforms, and builds the judicial system with a high degree of informatization.

As a relatively cutting-edge technology in natural language processing tasks, knowledge graphs have great potential in the construction of court informatization. It is a technical method that uses graph models to describe knowledge and build relationships between things. Domain knowledge graphs are mainly aimed at specific fields, emphasizing the depth of knowledge. In view of the complex and rigorous knowledge characteristics of the judicial field, it is a better choice to build a knowledge graph in the field of legal documents. The maturity of machine learning and graph databases have provided strong support for this goal.

### Conclusion

Dimensions corresponding to large singular values in the latent semantic space represent more "commonality" and less "personality" of language elements, while dimensions corresponding to small singular values represent more "personality." From this, the approximate and implicit correspondence between the dimension of the latent semantic space and the conceptual granularity is obtained, which provides a new idea for understanding and utilizing the dimension of the latent semantic space.

By using ontology technology and Neo4j graph database to store data and building a knowledge graph system in the field of public prosecutorial affairs, functions such as knowledge query, knowledge recommendation, and knowledge management are realized. Implementing online import, natural language query, and similar case push for entity nodes and relationships not only diversifies search results, but also enriches knowledge connections and ensures the practicability of knowledge graphs in the public domain of procuratorial affairs.

In this paper, CRF is added to bi-LSTM to optimize named entity recognition (bi-LSTM-CRF model), which considers both contextual information and the correlation between labels. The text representation, word vector representation, and related knowledge entity vector representation of the legal text are taken as multichannel inputs, enabling the network to learn a more adequate text representation by fusing information from multiple channels. The method proposed achieves significant improvement in the task of crime prediction in different types of cases. This paper verifies the accuracy and feasibility of the system.
