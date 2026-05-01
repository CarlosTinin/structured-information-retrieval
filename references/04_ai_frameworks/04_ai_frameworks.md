# 04 — AI Frameworks

---

## 04_01 — A Frustratingly Easy Approach for Entity and Relation Extraction

**Authors:** Zexuan Zhong, Danqi Chen  
**Conference:** NAACL-HLT 2021, pages 50–61  
**Venue:** Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies  
**Code:** https://github.com/princeton-nlp/PURE

### Abstract

End-to-end relation extraction aims to identify named entities and extract relations between them. Most recent work models these two subtasks jointly, either by casting them in one structured prediction framework, or performing multi-task learning through shared representations. In this work, we present a simple pipelined approach for entity and relation extraction, and establish the new state-of-the-art on standard benchmarks (ACE04, ACE05 and SciERC), obtaining a 1.7%-2.8% absolute improvement in relation F1 over previous joint models with the same pre-trained encoders. Our approach essentially builds on two independent encoders and merely uses the entity model to construct the input for the relation model. Through a series of careful examinations, we validate the importance of learning distinct contextual representations for entities and relations, fusing entity information early in the relation model, and incorporating global context. Finally, we also present an efficient approximation to our approach which requires only one pass of both entity and relation encoders at inference time, achieving an 8-16× speedup with a slight reduction in accuracy.

### Introduction

Extracting entities and their relations from unstructured text is a fundamental problem in information extraction. This problem can be decomposed into two subtasks: named entity recognition and relation extraction. Early work employed a pipelined approach, training one model to extract entities and another model to classify relations between them. More recently, end-to-end evaluations have been dominated by systems that model these two tasks jointly. There has been a long held belief that joint models can better capture the interactions between entities and relations and help mitigate error propagation issues.

In this work, the authors re-examine this problem and present a simple approach which learns two encoders built on top of deep pre-trained language models. The two models — the entity model and relation model — are trained independently and the relation model only relies on the entity model to provide input features. The entity model builds on span-level representations and the relation model builds on contextual representations specific to a given pair of spans. Despite its simplicity, this pipelined approach is extremely effective: using the same pre-trained encoders, the model outperforms all previous joint models on three standard benchmarks (ACE04, ACE05 and SciERC), advancing the previous state-of-the-art by 1.7%–2.8% absolute in relation F1.

Key findings include: (1) the contextual representations for the entity and relation models essentially capture distinct information, so sharing their representations hurts performance; (2) it is crucial to fuse entity information (both boundary and type) at the input layer of the relation model; (3) leveraging cross-sentence information is useful in both tasks.

### Conclusion

In this paper, we present a simple and effective approach for end-to-end relation extraction. Our model learns two encoders for entity recognition and relation extraction independently and our experiments show that it outperforms previous state-of-the-art on three standard benchmarks considerably. We conduct extensive analyses to understand the superior performance of our approach and validate the importance of learning distinct contextual representations for entities and relations and using entity information as input features for the relation model. We also propose an efficient approximation, obtaining a large speedup at inference time with a small reduction in accuracy. We hope that this simple model will serve as a very strong baseline and make us rethink the value of joint training in end-to-end relation extraction.

---

## 04_02 — Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)

**Authors:** Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang  
**Conference:** RecSys '22, September 18–23, 2022, Seattle, WA, USA  
**DOI:** https://doi.org/10.1145/3523227.3546767  
**Code:** https://github.com/jeykigung/P5

### Abstract

For a long time, different recommendation tasks require designing task-specific architectures and training objectives. As a result, it is hard to transfer the knowledge and representations from one task to another, thus restricting the generalization ability of existing recommendation approaches. To deal with such issues, considering that language can describe almost anything and language grounding is a powerful medium to represent various problems or tasks, we present a flexible and unified text-to-text paradigm called "Pretrain, Personalized Prompt, and Predict Paradigm" (P5) for recommendation, which unifies various recommendation tasks in a shared framework. In P5, all data such as user-item interactions, user descriptions, item metadata, and user reviews are converted to a common format — natural language sequences. The rich information from natural language assists P5 to capture deeper semantics for personalization and recommendation. Specifically, P5 learns different tasks with the same language modeling objective during pretraining. Thus, it serves as the foundation model for various downstream recommendation tasks, allows easy integration with other modalities, and enables instruction-based recommendation. With adaptive personalized prompt for different users, P5 is able to make predictions in a zero-shot or few-shot manner and largely reduces the necessity for extensive fine-tuning. On several benchmarks, we conduct experiments to show the effectiveness of P5.

### Introduction

For the past decades, recommender systems have witnessed significant advancements and played an essential role in people's daily life. The development trend of modern RS is towards a more comprehensive system that accommodates diverse features and a wide spectrum of application scenarios. Feature engineering and learning in RS has evolved from simple to complex — from logistic regression or collaborative filtering to factorization machines and deep neural network models. More recommendation tasks have also emerged beyond classical rating prediction, including sequential recommendation, conversational recommendation, and explainable recommendation.

Inspired by recent progress in multitask prompt-based training, the authors propose P5: a unified "Pretrain, Personalized Prompt & Predict Paradigm". P5 integrates various recommendation related tasks into a shared conditional language generation framework by formulating these problems as prompt-based natural language tasks. The three main advantages are: (1) P5 deeply immerses recommendation models into a full language environment, exploiting abundant semantics and knowledge inside the training corpora; (2) P5 integrates multiple recommendation tasks into a shared text-to-text encoder-decoder architecture with the same language modeling loss; (3) trained with instruction-based prompts, P5 attains sufficient zero-shot performance when generalizing to novel personalized prompts or unseen items.

### Conclusion

In this paper, we present P5 which unifies various recommendation tasks into a shared language modeling and generation framework. Based on personalized prompts covering five task families, we transfer all raw data such as the user-item interactions, user descriptions, item metadata and user reviews to the same format – input-target text pairs. We then pretrain P5 to help it discover deeper semantics for various recommendation tasks. Experiments show that P5 can beat or achieve similar performance with several representative approaches on all five task families. Moreover, P5 shows the generalization ability on zero-shot transfer to new items, new domains, and new prompts. In the future, we will further explore larger model size of P5 and employ more powerful base models such as GPT-3, OPT, and BLOOM. Besides, P5 is a flexible paradigm and it is promising to extend P5 to diverse modalities and tasks such as conversational recommendation, comparative recommendation, cross-platform recommendation, or even search tasks by integrating user queries into P5.

---

## 04_03 — A strategic framework for artificial intelligence in marketing

**Authors:** Ming-Hui Huang, Roland T. Rust  
**Journal:** Journal of the Academy of Marketing Science (2021), vol. 49, pp. 30–50  
**DOI:** https://doi.org/10.1007/s11747-020-00749-9

### Abstract

The authors develop a three-stage framework for strategic marketing planning, incorporating multiple artificial intelligence (AI) benefits: mechanical AI for automating repetitive marketing functions and activities, thinking AI for processing data to arrive at decisions, and feeling AI for analyzing interactions and human emotions. This framework lays out the ways that AI can be used for marketing research, strategy (segmentation, targeting, and positioning, STP), and actions. At the marketing research stage, mechanical AI can be used for data collection, thinking AI for market analysis, and feeling AI for customer understanding. At the marketing strategy (STP) stage, mechanical AI can be used for segmentation (segment recognition), thinking AI for targeting (segment recommendation), and feeling AI for positioning (segment resonance). At the marketing action stage, mechanical AI can be used for standardization, thinking AI for personalization, and feeling AI for relationalization. We apply this framework to various areas of marketing, organized by marketing 4Ps/4Cs, to illustrate the strategic use of AI.

### Introduction

Artificial intelligence (AI) in marketing is currently gaining importance, due to increasing computing power, lower computing costs, the availability of big data, and the advance of machine learning algorithms and models. The academic literature on AI in marketing may be sorted into four main types: (1) technical AI algorithms for solving specific marketing problems, (2) customers' psychological reactions to AI, (3) effects of AI on jobs and society, and (4) managerial and strategic issues related to AI.

To facilitate the strategic use of AI in marketing, the authors develop a three-stage framework, from marketing research, to marketing strategy (STP), to marketing actions (4Ps/4Cs), for strategic marketing planning incorporating AI. This strategic AI framework is based on a more nuanced perspective of the technical development of AI, existing studies on AI and marketing, and current and future AI applications. The paper contributes to the strategic application of AI in marketing by developing a framework that guides the strategic planning of AI in marketing in a systematic and actionable manner.

The conceptual foundation introduces three AI intelligences: mechanical AI (automating repetitive tasks), thinking AI (processing data to arrive at new conclusions or decisions), and feeling AI (designed for two-way interactions involving humans and/or analyzing human feelings and emotions). Each intelligence delivers a unique benefit: mechanical AI provides standardization, thinking AI provides personalization, and feeling AI provides relationalization.

### Conclusion

The most disruptive aspect of AI is that it replaces and improves upon human thinking capability. One of the most revolutionary characteristics of modern thinking AI is its ability to personalize by analyzing big data in an automatic way. This creates a quantum leap in marketing's ability to target individual customers. Marketing primarily requires thinking intelligence and feeling intelligence. Until now there has been only limited ability of technology to help with those things. Now as thinking AI is advancing rapidly, it is gaining the ability to assume many of the thinking tasks in marketing. Eventually will even assume many of the feeling tasks in marketing, as AI develops further.

The authors note that marketers who cannot wait for technology to sufficiently advance use mechanical AI and thinking AI for feeling tasks, due to true feeling AI not being ready yet. They also observe that AI intelligences may not be used in the most effective way. Thus, this strategic framework is developed to help marketers leverage the benefits of multiple AI intelligences for marketing impact, laying out the ways in which various AI intelligences can be used in marketing research, marketing strategy (STP), and marketing action (4Ps/4Cs).

---

## 04_04 — An open source machine learning framework for efficient and transparent systematic reviews

**Authors:** Rens van de Schoot, Jonathan de Bruin, Raoul Schram, Parisa Zahedi, Jan de Boer, Felix Weijdema, Bianca Kramer, Martijn Huijts, Maarten Hoogerwerf, Gerbrich Ferdinands, Albert Harkema, Joukje Willemsen, Yongchao Ma, Qixiang Fang, Sybren Hindriks, Lars Tummers, Daniel L. Oberski  
**Journal:** Nature Machine Intelligence (2021), vol. 3, pp. 125–133  
**DOI:** https://doi.org/10.1038/s42256-020-00287-7

### Abstract

To help researchers conduct a systematic review or meta-analysis as efficiently and transparently as possible, we designed a tool to accelerate the step of screening titles and abstracts. For many tasks — including but not limited to systematic reviews and meta-analyses — the scientific literature needs to be checked systematically. Scholars and practitioners currently screen thousands of studies by hand to determine which studies to include in their review or meta-analysis. This is error prone and inefficient because of extremely imbalanced data: only a fraction of the screened studies is relevant. The future of systematic reviewing will be an interaction with machine learning algorithms to deal with the enormous increase of available text. We therefore developed an open source machine learning-aided pipeline applying active learning: ASReview. We demonstrate by means of simulation studies that active learning can yield far more efficient reviewing than manual reviewing while providing high quality. Furthermore, we describe the options of the free and open source research software and present the results from user experience tests. We invite the community to contribute to open source projects such as our own that provide measurable and reproducible improvements over current practice.

### Introduction

With the emergence of online publishing, the number of scientific manuscripts on many topics is skyrocketing. Scholars often develop systematic reviews and meta-analyses to develop comprehensive overviews of relevant topics. The process entails several explicit and reproducible steps, including identifying all likely relevant publications in a standardized way, extracting data from eligible studies and synthesizing the results. The process of systematic reviewing is error prone and extremely time intensive. The rapidly evolving field of machine learning has aided researchers by allowing the development of software tools that assist in developing systematic reviews.

Active learning is a type of machine learning in which a model can choose the data points it would like to learn from and thereby drastically reduce the total number of records that require manual screening. The term researcher-in-the-loop was introduced as a special case of human-in-the-loop with three unique components: (1) the primary output is a selection of the records, not a trained ML model; (2) all records in the relevant selection are seen by a human; (3) the use-case requires a reproducible workflow and complete transparency.

Existing tools have two main drawbacks: many are closed source with black box algorithms, and they lack the necessary flexibility to deal with the large range of possible concepts to be learned. ASReview addresses all these concerns: it is open source, uses active learning, allows multiple machine learning models, has a benchmark mode, and is easily extensible.

### Conclusion

We designed a system to accelerate the step of screening titles and abstracts to help researchers conduct a systematic review or meta-analysis as efficiently and transparently as possible. Our system uses active learning to train a machine learning model that predicts relevance from texts using a limited number of labelled examples. The classifier, feature extraction technique, balance strategy and active learning query strategy are flexible. We provide an open source software implementation, ASReview, with state-of-the-art systems across a wide range of real-world systematic reviewing applications.

Drawbacks remain: (1) active learning prevents straightforward evaluation of error rates without further labelling; (2) no empirical benchmarks of performance in non-reviewing situations yet exist; (3) machine learning-based screening systems automate only the screening step, which is just one part of a much larger process including initial search, data extraction, coding for risk of bias, and summarizing results.

Future research could focus on performance of identifying full text articles with different document lengths and domain-specific terminologies or even other types of text such as newspaper articles and court cases. The future of systematic reviewing will be an interaction with machine learning algorithms to deal with the enormous increase of available text.
