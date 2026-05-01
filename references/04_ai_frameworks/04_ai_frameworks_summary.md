# 04 — AI Frameworks: Article Summaries

---

## 04_01 — A Frustratingly Easy Approach for Entity and Relation Extraction

**Authors:** Zexuan Zhong, Danqi Chen  
**Conference:** NAACL-HLT 2021, pages 50–61  
**Code:** https://github.com/princeton-nlp/PURE

Zhong and Chen challenge the prevailing assumption that joint models are necessary for end-to-end entity and relation extraction by presenting PURE, a deliberately simple pipelined approach that learns two independent encoders built on pre-trained language models (BERT, ALBERT, SciBERT). The entity model predicts entity spans and types using standard span-level representations, while the relation model independently processes each candidate entity pair by inserting typed markers into the input sequence to highlight the subject and object spans. This design ensures that each entity pair receives a distinct contextual representation — a property the authors identify as critical, since shared representations across pairs are suboptimal for capturing pair-specific dependencies.

The approach achieves new state-of-the-art results on three benchmarks (ACE04, ACE05, SciERC), surpassing all previous joint models by 1.7%–2.8% absolute in relation F1. Careful ablations reveal three key findings: (1) entity and relation models capture fundamentally distinct contextual information, making parameter sharing counterproductive; (2) injecting entity type information at the input layer of the relation model (via typed markers) is substantially more effective than injecting it through auxiliary losses; (3) cross-sentence context improves both tasks. An efficient approximation batching multiple entity pairs in a single forward pass achieves 8–16× speedup with only ~1% F1 drop. The work serves as a strong argument that, given sufficiently powerful pre-trained representations, the simplicity and modularity of pipeline architectures can outperform more complex joint frameworks.

---

## 04_02 — Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)

**Authors:** Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang  
**Conference:** RecSys '22, Seattle, WA, USA  
**DOI:** https://doi.org/10.1145/3523227.3546767

Geng et al. propose P5, a unified text-to-text framework that reformulates diverse recommendation tasks — rating prediction, sequential recommendation, explanation generation, review summarization, and direct recommendation — as natural language generation problems. All heterogeneous recommendation data (user-item interactions, metadata, reviews) are converted into natural language sequences and processed through a single encoder-decoder Transformer (T5-based) with personalized prompt templates. This paradigm eliminates the need for task-specific architectures or objective functions and enables zero-shot generalization to unseen prompts, new items, and new domains.

The framework is motivated by the observation that language grounding is sufficiently flexible to express any recommendation input, making feature-specific encoders unnecessary. P5 is pretrained on a multitask collection of personalized prompts spanning five task families simultaneously. Experiments demonstrate competitive or superior performance compared to task-specific baselines across all five families, with notably strong zero-shot transfer capabilities. The work advances recommender systems from specialized deep models toward a universal recommendation engine paradigm, drawing a direct analogy to the "pre-train, prompt, predict" revolution in NLP. Future directions include scaling to larger base models (GPT-3, OPT, BLOOM) and extending to conversational and cross-platform recommendation.

---

## 04_03 — A strategic framework for artificial intelligence in marketing

**Authors:** Ming-Hui Huang, Roland T. Rust  
**Journal:** Journal of the Academy of Marketing Science (2021), vol. 49, pp. 30–50  
**DOI:** https://doi.org/10.1007/s11747-020-00749-9

Huang and Rust develop a conceptual three-stage strategic framework for incorporating AI into marketing planning, structured around the marketing research–strategy–action cycle. The framework introduces a taxonomy of three AI intelligences ordered by difficulty: mechanical AI (automating repetitive tasks, providing standardization), thinking AI (processing data to arrive at decisions, providing personalization), and feeling AI (analyzing interactions and human emotions, providing relationalization).

The framework maps these intelligences onto the three strategic marketing stages: at the research stage, mechanical AI handles data collection, thinking AI performs market analysis, and feeling AI enables customer understanding; at the strategy (STP) stage, mechanical AI supports segmentation, thinking AI targeting, and feeling AI positioning; at the action stage (4Ps/4Cs), mechanical AI drives standardization, thinking AI personalization, and feeling AI relationalization. The authors apply this framework systematically across product, price, place, and promotion decisions, identifying specific current and future AI applications at each intersection.

The paper acknowledges that true feeling AI does not yet exist — current practice uses thinking AI to analyze emotional data. The strategic contribution lies in providing a systematic, actionable guide for marketers to leverage AI capabilities appropriately, and in identifying research gaps at the intersection of AI and marketing strategy. This is a conceptual/theoretical paper without empirical evaluation, contributing primarily through its organizing framework.

---

## 04_04 — An open source machine learning framework for efficient and transparent systematic reviews

**Authors:** Rens van de Schoot, Jonathan de Bruin, Raoul Schram, et al.  
**Journal:** Nature Machine Intelligence (2021), vol. 3, pp. 125–133  
**DOI:** https://doi.org/10.1038/s42256-020-00287-7

Van de Schoot et al. present ASReview, an open-source machine learning pipeline that applies active learning to accelerate the title-and-abstract screening phase of systematic reviews. The tool addresses the fundamental bottleneck that systematic reviews require screening thousands of records manually, of which typically less than 5% are relevant — an extremely imbalanced classification problem. ASReview implements a researcher-in-the-loop active learning cycle: after minimal prior knowledge input (at least one relevant and one irrelevant record), the system iteratively trains a classifier, selects the most likely relevant record for human labeling, and retrains.

The software provides flexible, modular components: multiple classifiers (Naive Bayes, SVM, neural networks, logistic regression, LSTM, random forests), feature extraction methods (TF-IDF, Doc2Vec, sentence-BERT, embedding-IDF), query strategies (certainty-based, uncertainty-based, random, mixed), and balance strategies (dynamic resampling, undersampling). It runs locally to preserve data privacy — a key differentiator from competing tools. Simulation studies on four labeled datasets demonstrate that ASReview achieves recall of 95% of relevant records after screening only a fraction of the total corpus, with the default configuration (Naive Bayes + TF-IDF + certainty-based sampling + dynamic resampling) showing consistently strong performance across diverse domains.

The tool is positioned as directly relevant to the legal and judicial domain: the authors explicitly identify court cases as a future application domain. Key limitations include the difficulty of estimating error rates during active learning, the lack of benchmarks beyond systematic reviewing, and the fact that screening automation addresses only one step in the broader review pipeline. The paper emphasizes open science principles, with all code, data, and results publicly available.
