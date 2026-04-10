# 03 — Segmentation

---

## 03_01 — The use of information technology for international transfer pricing in multinational enterprises

**Authors:** Lars Hemling, Jacob Christian Plesner Rossing, Andreas Hoffjan  
**Journal:** International Journal of Accounting Information Systems (2022), vol. 44, 100546  
**DOI:** https://doi.org/10.1016/j.accinf.2021.100546

> **Note:** This article is thematically outside the NLP/legal-AI domain. It concerns ERP systems and tax compliance in multinational enterprises. It has no explicit "Conclusion" section; the "Discussion" section (Section 5) is used as the closest equivalent.

### Abstract

This paper studies the degree to which multinational enterprises (MNEs) use information technology for managing international transfer pricing (ITP). Based on 21 interviews conducted with in-house accounting and tax professionals in MNEs, we observed limited use of information technology for ITP management. However, some degree of ITP automation was observed in workflow management to produce transfer pricing documentation. The limited degree of automation observed was driven by both system- and individual-level barriers. Overall, we found that management accountants and information technology experts dominate the enterprise resource planning system design agenda, and the tax departments' ITP tax compliance objective plays a relatively limited role. This reduces the ability for ITP automation partly because the data segmentation that is prioritized for management reporting does not support the tax departments' needs for legal-entity data segmentation to document tax compliance.

### Introduction

The tax strategies deployed by multinational enterprises (MNEs) continue to receive public attention. Several high-profile legal disputes between MNEs and tax authorities have demonstrated the significance of the tax practices of MNEs, particularly those related to international transfer pricing (ITP). ITP entails opportunities for tax avoidance resulting from the ability to exploit tax rate differentials between countries hosting affiliated companies. For example, an MNE with a company located in a tax haven can decrease its group-level tax burden by transferring goods at artificially high prices to an affiliated company located in a high-tax jurisdiction.

In response to alleged tax avoidance by MNEs, prominent trade institutions, such as the European Union and the OECD, exert continuous efforts to enhance global tax policy and promote tax compliance. The BEPS action plan (OECD, 2015) addresses various tax issues, including revised guidelines for setting and documenting transfer prices based on the arm's length principle. Given the tightened tax regulatory environment combined with the increased attention to tax digitization, the issue of information technology and automation relating to ITP are becoming increasingly important.

Based on 21 interviews conducted with in-house accounting and tax professionals, we found only limited use of information technology for ITP management. The main automation observed relates to the management of intra-organizational workflows for developing local documentation. The limited use of information technology for ITP is significantly driven by system-level conditions, including fragmented ERP system environments. Functional hierarchies also play a role: ERP system design is based on the prioritization of management reporting needs, while ITP tax compliance is based on legal entity-level reporting. These differing data segmentation needs ultimately reduce the tax departments' ability to automate ITP processes.

### Conclusion (Discussion)

In this paper, we studied MNEs' use of information technology for managing ITP tax risks. While our respondents generally viewed ITP automation as an appealing idea, we observed that ERP system configurations in MNEs create numerous barriers at both the system and individual levels. At the system level, the main barriers were fragmented ERP systems and differences in the accounting information needs for management reporting versus tax reporting. ERP system designs were found to be based on the need for internal control and management reporting, while tax compliance objectives still received relatively limited priority. Ultimately, the prioritization of management reporting over tax compliance in ERP system design caused tax departments to gravitate towards traditional Excel solutions to handle ITP.

In contrast to the limited automation observed in transfer price setting, automation using ITP tools seemed more widespread in the documentation process, though mainly tied to workflow management rather than automated data importing. Tax departments in MNEs were found to use a highly manual documentation process based on in-house Excel models, reducing data integrity and consistency. As tax authorities continue to digitize their audit processes, MNEs will face increasing pressure to improve their own digitization for ITP. Our findings suggest that a standardized ITP policy across the value chain is a prerequisite for effectively automating intercompany transactions within ERP applications.

We also observed several individual-level barriers to ITP automation. Tax departments are increasingly recognizing the importance of the complex integration between ITP and ERP systems, particularly in the wake of BEPS. We believe this implies a significant need for CFOs and senior-level tax directors to initiate more focused efforts to ensure agility in information technology within tax departments. Limitations of this research include the focus on MNEs based primarily in Germany and the exclusion of external tax consultants and ITP software developers as interviewees. Future research could study how specific ERP applications allow for more efficient ITP automation, as well as exploring how organizational attributes affect ITP automation objectives.

---

## 03_02 — Adversarial training flat-lattice transformer for named entity recognition of Chinese legal texts

**Authors:** Jiabao Wang, Kaixuan Wang, Yang Weng, Xin Li  
**Journal:** Artificial Intelligence and Law (2025)  
**DOI:** https://doi.org/10.1007/s10506-025-09476-4

### Abstract

Judgment documents are the legally binding written conclusion made by the court based on the facts of the case and the law. Due to the use of professional terms and nested combinations of words, potential information of judgment documents has not been deeply excavated. Named Entity Recognition (NER) is a necessary task in Natural Language Processing (NLP), and has been widely introduced into Chinese texts processing for many years. However, the professional terms and nested words lead to the boundaries between entities being blurred, which cannot accurately divide entities. In this paper, a new NER model Adversarial Training Flat-Lattice Transformer (AT-Flat) which combines adversarial training and Flat-Lattice Transformer (Flat) is proposed to weaken these problems. In AT-Flat, the Flat combines character and word information to get sequence information, and the CRF is used to output the final entity prediction results. Moreover, the key point is an adversarial training framework introduced to integrate task-shared word boundary information from Chinese Word Segmentation (CWS) task into Chinese NER task. The framework is able to filter out the noise caused by CWS task and further enhance the effect of Chinese NER task. More importantly, experiments on NER task of Chinese traffic accident and financial lending judgment documents show that our method outperforms other state-of-the-art methods. These verify that our method can effectively alleviate the problem of poor NER effect caused by professional terms and nested words. In addition, three public Chinese NER datasets were also used to evaluate our method.

### Introduction

The amount of data in the judicial field has increased rapidly in recent years. Legal professionals such as judges, lawyers, and prosecutors not only handle a large number of cases, but also need to consult numerous references. Judgment documents are the legally binding written conclusion made by the court based on the facts of the case and the law. With the development of judicial online systems, nearly 100 million judgment documents have been published online in China, accumulating abundant judicial data of various cases. Judgment documents can help professionals get similar cases to assist in sentencing, reducing labor costs and improving legal work efficiency.

NER aims to extract words or expressions denoting specific entities from documents, composed of two parts: detecting entity boundaries and classifying entities into predefined categories such as Name, Judicial Organization, and Location. Chinese judicial NER presents significant challenges due to the absence of spaces between words, the lack of uppercase semantic features, dataset scarcity relative to English, and the multiple meanings of words in different contexts. Furthermore, judicial entities are mostly nested, making structures complicated—for example, a "Judicial Organization" entity may contain a "Location" sub-entity.

Although existing NER methods achieve great success in many domains, most of them normally ignore the nested nature of named entities. To address these problems, this paper proposes AT-Flat, a new NER model that combines adversarial training and the Flat-Lattice Transformer. The adversarial training framework extracts common features of the CWS and NER tasks, while filtering out noise from the CWS task. This makes the boundary of legal entities clearer and improves the effect of Chinese NER. To the best of the authors' knowledge, this is the first work introducing adversarial training to process nested NER of Chinese judgment documents.

### Conclusion

In this paper, we propose AT-Flat, an adversarial training model that integrates task-shared word boundary information from CWS to improve Chinese judgment documents NER performance, and combines character and word information to get more legal entity information. Extensive experiments based on two different Chinese judgment document datasets and three public Chinese NER datasets show that AT-Flat achieves the best performance compared to state-of-the-art models. For future work, the model will be adjusted to expand its application fields. Although AT-Flat achieves strong performance, it still has some limitations. One potential bottleneck is its limited capacity to model long-range dependencies, which may affect its effectiveness when processing lengthy legal documents. Additionally, the model's inference efficiency might degrade as input length increases, posing challenges for real-world deployment in legal or industrial scenarios.

---

## 03_03 — Legal Judgment Prediction in the Saudi Arabian Commercial Court

**Authors:** Ashwaq Almalki, Safa Alsafari, Noura M. Alotaibi  
**Journal:** Future Internet (2025), vol. 17, 439  
**DOI:** https://doi.org/10.3390/fi17100439

### Abstract

Legal judgment prediction is an emerging application of artificial intelligence in the legal domain, offering significant potential to enhance legal decision support systems. Such systems can improve judicial efficiency, reduce burdens on legal professionals, and assist in early-stage case assessment. This study focused on predicting whether a legal case would be Accepted or Rejected using only the Fact section of court rulings. A key challenge lay in processing long legal documents, which often exceeded the input length limitations of transformer-based models. To address this, we proposed a two-step methodology: first, each document was segmented into sentence-level inputs compatible with AraBERT—a pretrained Arabic transformer model—to generate sentence-level predictions; second, these predictions were aggregated to produce a document-level decision using several methods, including Mean, Max, Confidence-Weighted, and Positional aggregation. We evaluated the approach on a dataset of 19,822 real-world cases collected from the Saudi Arabian Commercial Court. Among all aggregation methods, the Confidence-Weighted method applied to the AraBERT-based classifier achieved the highest performance, with an overall accuracy of 85.62%. The results demonstrated that combining sentence-level modeling with effective aggregation methods provides a scalable and accurate solution for Arabic legal judgment prediction, enabling full-length document processing without truncation.

### Introduction

Legal Artificial Intelligence (LegalAI) refers to the use of artificial intelligence methods in the legal domain. One prominent task in this area is legal judgment prediction (LJP), which aims to predict the outcome of a case by learning patterns from previously adjudicated cases. Such systems have the potential to save time and effort, reduce the burden on judges, lawyers, and plaintiffs, and provide a preliminary review of a case before it is initiated. One of the main challenges in LJP is language diversity. Legal systems also vary significantly from one country to another, even among Arabic-speaking countries, as each country has its own legal rules, reasoning processes, and terminology.

This need is particularly critical within the Kingdom of Saudi Arabia, where the number of filed cases increased by 23.56% between 2021 and 2024. Several studies have explored NLP in the Arabic legal domain, but only a few have attempted LJP in the Saudi context. The present study advances beyond prior efforts in three ways: first, it builds the largest dataset to date of Saudi commercial court cases for LJP (19,822 judgments); second, it introduces a segmentation-and-aggregation framework that allows transformer models to process entire case documents without truncation, overcoming the 512-token input limit; and third, it provides an outcome-focused prediction system (Accepted vs. Rejected) with interpretable sentence-level insights.

### Conclusion

This study proposed a two-step framework for LJP in Arabic commercial court cases. The Fact section of each legal document was first segmented into sentence-level inputs and classified using AraBERT v2. Then, sentence-level predictions were aggregated to generate a final document-level decision. Experimental results demonstrated that all aggregation methods performed competitively, with Confidence-Weighted aggregation achieving the highest accuracy (85.62%) and F1-score (85.61%). Error analysis revealed higher precision for Rejected cases, increased error rates in longer documents, and performance variations across case types.

The proposed framework could serve as a reliable LJP assistant, enhancing judicial efficiency by reducing case screening time and providing early insights into likely verdicts. However, it should be viewed strictly as a decision-support tool rather than a replacement for human judgment. Future research should focus on developing models capable of capturing long-range context in Arabic legal texts, extending evaluation to diverse legal domains, addressing complex prediction tasks such as handling "Mixed" outcome cases, and incorporating domain-adaptive pretraining to further improve legal terminology understanding.

---

## 03_04 — LAWSUIT: a LArge expert-Written SUmmarization dataset of ITalian constitutional court verdicts

**Authors:** Luca Ragazzi, Gianluca Moro, Stefano Guidi, Giacomo Frisoni  
**Journal:** Artificial Intelligence and Law (2025), vol. 33, pp. 1151–1187  
**DOI:** https://doi.org/10.1007/s10506-024-09414-w

### Abstract

Large-scale public datasets are vital for driving the progress of abstractive summarization, especially in law, where documents have highly specialized jargon. However, the available resources are English-centered, limiting research advancements in other languages. This paper introduces LAWSUIT, a collection of 14K Italian legal verdicts with expert-authored abstractive maxims drawn from the Constitutional Court of the Italian Republic. LAWSUIT presents an arduous task with lengthy source texts and evenly distributed salient content. We offer extensive experiments with sequence-to-sequence and segmentation-based approaches, revealing that the latter achieve better results in full and few-shot settings. We openly release LAWSUIT to foster the development and automation of real-world legal applications.

### Introduction

Text summarization is a persistent pursuit of natural language processing (NLP). Recently, there has been a growing interest in abstractive summarization (AS), which involves paraphrasing the essential details of textual documents in a succinct and accessible language. One particularly impactful domain in real-world applications is law, where documents often consist of thousands of words filled with jargon and intricate expressions. The complexity of these documents makes their comprehension a time-consuming and labor-intensive process, even for legal experts. Therefore, legal AS is a practical, useful, and essential task to promote knowledge acquisition. Lamentably, current legal summarization corpora are almost entirely devoted to English, with no Italian datasets for legal AS, which limits research, access, and elaboration of lawful texts and their implications to Italian law practitioners.

To fill this gap, the authors present LAWSUIT, the first large-scale Italian legal AS dataset consisting of 14,000 source documents with expert-authored summaries drawn from the Constitutional Court of the Italian Republic. LAWSUIT's key features include: source and target texts significantly longer than existing Italian summarization datasets (+269% and +589%, respectively); salient content that is more uniformly distributed throughout the input rather than concentrated in specific sections; and inputs and targets authored by legal experts. These characteristics pose unique challenges for summarization tasks, requiring comprehensive processing of the entire source document rather than relying on localized content.

### Conclusion

In this paper, the authors introduced LAWSUIT, the first large-scale dataset for the abstractive summarization of long Italian legal verdicts. The challenges presented by LAWSUIT include lengthy sources, the uniform distribution of relevant information throughout the input, and the lower presence of formulaic patterns in the targets. Through an extensive series of experiments, the authors found that a text segmentation pipeline significantly outperforms other methods in both few-shot and full summarization settings. The authors anticipate that LAWSUIT will contribute to the development of real-world legal summarization systems and stimulate research towards effective long-range solutions for Italian legal documents. Future work will extend LAWSUIT to new tasks such as cross-domain classification, legal reasoning, open-domain question answering, and corpus-level knowledge extraction. Researchers could also explore efficient segmentation and summarization techniques based on graph sparsification by representing the source document as a graph.

---

## 03_05 — DeepRhole: deep learning for rhetorical role labeling of sentences in legal case documents

**Authors:** Paheli Bhattacharya et al.  
**Journal:** Artificial Intelligence and Law (2023), vol. 31, pp. 53–90  
**DOI:** https://doi.org/10.1007/s10506-021-09304-5

### Abstract

The task of rhetorical role labeling is to assign labels (such as Fact, Argument, Final Judgement, etc.) to sentences of a court case document. Rhetorical role labeling is an important problem in the field of Legal Analytics, since it can aid in various downstream tasks as well as enhances the readability of lengthy case documents. The task is challenging as case documents are highly various in structure and the rhetorical labels are often subjective. Previous works for automatic rhetorical role identification mainly used Conditional Random Fields over manually handcrafted features, and focused on certain law domains only (e.g., Immigration cases, Rent law), and a particular jurisdiction/country (e.g., US, Canada, India). In this work, we improve upon the prior works on rhetorical role identification by proposing novel Deep Learning models for automatically identifying rhetorical roles, which substantially outperform the prior methods. Additionally, we show the effectiveness of the proposed models over documents from five different law domains, and from two different jurisdictions—the Supreme Court of India and the Supreme Court of the UK. Through extensive experiments over different variations of the Deep Learning models, including Transformer models based on BERT and LegalBERT, we show the robustness of the methods for the task. We also perform an extensive inter-annotator study and analyse the agreement of the predictions of the proposed model with the annotations by domain experts. We find that some rhetorical labels are inherently hard/subjective and both law experts and neural models frequently get confused in predicting them correctly.

### Introduction

Rhetorical role labelling of sentences in a legal document refers to understanding what semantic function each sentence is associated with. Examples of these roles include facts of the case, arguments of the contending parties, the final judgement of the Court, and so on. Identifying the rhetorical roles of sentences in a legal case document can help in a variety of downstream tasks like semantic search, summarization, and case law analysis. However, legal case documents are highly varying in structure, and various themes often interleave with each other—for instance, the reason behind the judgment often interleaves with Precedents and Statutes, making it difficult even for human experts to understand the intricate differences between the rhetorical roles.

Prior attempts to automate the identification of rhetorical roles relied on hand-crafted features such as linguistic cue phrases and sequential label arrangements. These features have always been developed for legal documents of a particular jurisdiction and do not scale to other jurisdictions or domains. Recently developed deep learning models do not require hand-engineered features but can automatically learn them from sufficient training data, and have the ability of generalizing to any domain of law or jurisdiction. In this paper, the authors explore four neural network models: Hierarchical BiLSTM (Hier-BiLSTM), Hierarchical BiLSTM/BiGRU combined with CRF, Hierarchical BiLSTM with Attention, and Transformer-based models (Tf-BiLSTM/BiGRU-CRF). These models are used for supervised classification across seven rhetorical labels over legal case documents from the Indian and UK Supreme Courts.

### Conclusion

The objective of this work is to automate the task of rhetorical role labeling of legal case documents, working on Indian Supreme Court and UK Supreme Court case documents. The key take-aways are: (i) Neural models using Hierarchical BiLSTM architectures are much better in rhetorical role labeling compared to prior CRF methods, and generalize well across jurisdictions and legal domains. (ii) Some specific pairs of rhetorical roles—such as (Ratio of the decision, Precedents)—are inherently subjective and lead to disagreement even among law experts. (iii) Domain-specific pretraining is more beneficial; pretraining using Law2Vec word embeddings performs better than random initialization or Google News vectors. (iv) Transformer models—BERT and LegalBERT—are also well suited for the task when fine-tuned. (v) It is beneficial to use models pretrained on data from the same target jurisdiction; however, good results can also be obtained by adapting a model trained on a different jurisdiction using only small amounts of target-domain data.

Finally, neural models often lack transparency or explainability, which is very much desired in the legal domain. Future work plans to model the rhetorical role labelling task as a multi-label classification problem, apply the labeled documents to downstream summarization tasks, and explore whether the models can be applied in non-Common Law legal settings.

---

## 03_06 — CoTHSSum: Structured long-document summarization via chain-of-thought reasoning and hierarchical segmentation

**Authors:** Xiaoyong Chen, Zhiqiang Chen, Shi Cheng  
**Journal:** Journal of King Saud University – Computer and Information Sciences (2025), vol. 37, 40  
**DOI:** https://doi.org/10.1007/s44443-025-00041-2

### Abstract

Long-document summarization remains a challenging task for large language models (LLMs), which often suffer from input length constraints, semantic incoherence, and factual hallucinations when processing extensive and complex texts. In this paper, we propose a novel summarization framework that integrates hierarchical input segmentation with Chain-of-Thought (CoT) prompting to guide LLMs through structured, interpretable reasoning. Our method decomposes long documents into semantically coherent segments, applies CoT-based prompting for intermediate summary reasoning, and employs structure-guided decoding to compose high-quality final summaries. We evaluate our approach across five diverse datasets, including scientific, biomedical, governmental, literary, and legal domains, using strong LLM backbones such as Qwen, LLaMA, and Phi. Experimental results demonstrate that our method consistently outperforms state-of-the-art baselines across ROUGE, BLEU, BERTScore, and factual consistency metrics. Ablation and human evaluation further confirm the complementary benefits of CoT reasoning and hierarchical structure, offering a reliable and scalable solution for summarizing complex long-form content.

### Introduction

The rapid growth of online text in news, scholarly articles, and social media has made automatic summarization a crucial technology for information management. Summarizing lengthy documents remains especially challenging: as text length grows, models not only risk semantic drift but also struggle to preserve crucial content. Recent advances in LLMs such as GPT-3 and GPT-4 have substantially elevated summarization performance, but as input length increases, LLM-based summarization still faces limitations in factual consistency and coverage. Existing LLM-based summarizers often produce hallucinations, where generated content is factually incorrect or ungrounded, and may omit critical content when dealing with very long inputs. Even hierarchical or chunk-based approaches can yield disjointed results without careful coordination.

The paper proposes a new method that addresses these issues by combining chain-of-thought (CoT) guided reasoning with a hierarchical input/output structure. CoT prompting involves guiding the LLM to generate intermediate reasoning steps or sub-summaries, rather than directly jumping to the final summary. This is coupled with a hierarchical summarization framework, wherein the long input document is broken into smaller coherent segments. The model summarizes each segment with its own chain-of-thought, and these segment summaries are subsequently aggregated and refined. The three key contributions are: (1) a novel summarization framework that integrates CoT reasoning with hierarchical summarization—the first to combine stepwise logical reasoning and multi-level input segmentation for long-text summarization by LLMs; (2) demonstrated superiority over standard prompting techniques; and (3) extensive experiments on multiple long-text datasets with in-depth analysis.

### Conclusion

In this paper, we presented a novel summarization framework combining hierarchical input structuring and explicit Chain-of-Thought (CoT) prompting to address the challenges inherent in long-text summarization, such as information omission, semantic incoherence, and factual hallucination. Comprehensive experiments across diverse datasets (ArXiv, PubMed, GovReport, BookSum, and the CAILsfzy legal dataset) demonstrated that our approach significantly outperforms state-of-the-art baseline models in terms of ROUGE, BLEU, BERTScore, and FactCC metrics. Ablation studies highlighted the critical contributions of both hierarchical segmentation and CoT prompting, while human evaluations further validated the superior factual consistency, completeness, fluency, and structural clarity of our generated summaries. While effective, our approach is limited by computational efficiency and domain adaptability. Future work will refine segmentation techniques, explore automatic CoT prompting, and extend the framework to cross-lingual and multimodal summarization tasks.

---

## 03_07 — Legal sentence boundary detection using hybrid deep learning and statistical models

**Authors:** Reshma Sheik, Sneha Rao Ganta, S. Jaya Nirmala  
**Journal:** Artificial Intelligence and Law (2025), vol. 33, pp. 519–549  
**DOI:** https://doi.org/10.1007/s10506-024-09394-x

### Abstract

Sentence boundary detection (SBD) represents an important first step in natural language processing since accurately identifying sentence boundaries significantly impacts downstream applications. Nevertheless, detecting sentence boundaries within legal texts poses a unique and challenging problem due to their distinct structural and linguistic features. Our approach utilizes deep learning models to leverage delimiter and surrounding context information as input, enabling precise detection of sentence boundaries in English legal texts. We evaluate various deep learning models, including domain-specific transformer models like LegalBERT and CaseLawBERT. To assess the efficacy of our deep learning models, we compare them with a state-of-the-art domain-specific statistical conditional random field (CRF) model. After considering model size, F1-score, and inference time, we identify the Convolutional Neural Network Model (CNN) as the top-performing deep learning model. To further enhance performance, we integrate the features of the CNN model into the subsequent CRF model, creating a hybrid architecture that combines the strengths of both models. Our experiments demonstrate that the hybrid model outperforms the baseline model, achieving a 4% improvement in the F1-score. Additional experiments showcase the superiority of the hybrid model over SBD open-source libraries when confronted with an out-of-domain test set. These findings underscore the importance of efficient SBD in legal texts and emphasize the advantages of employing deep learning models and hybrid architectures to achieve optimal performance.

### Introduction

Sentence Boundary Detection (SBD) is an essential component of NLP, involving identifying the boundaries between sentences in a given text. While humans can easily identify and separate sentences within a given text, developing automatic SBD algorithms is a complex task. The task has been extensively studied in general domains, but detecting sentence boundaries in specialized corpora such as legal text poses significant challenges. Legal text is characterized by lengthy and intricate sentences with lists, citations, abbreviations, and law-specific keywords, often making it difficult to compose actual sentences from legal documents. In addition, legal documents of the same kind, like laws or court decisions, can vary significantly in structure and content, making it difficult to separate one sentence from another. Accurate SBD is a critical research area in NLP and is useful in many downstream applications like legal argument mining, legal text summarization, rhetorical role segmentation, and legal information retrieval.

One of the primary concerns with SBD for legal documents is the ambiguous nature of delimiters: for example, the period delimiter may indicate the boundary of a line, an abbreviation, a numerical value, or part of a citation in legal text. The study uses a deep learning framework that utilizes the contextual information surrounding delimiters for SBD, treating it as a sequence classification problem. A performance improvement is then demonstrated using a hybrid architecture incorporating the state-of-the-art statistical CRF model and the CNN. Key contributions include: a deep learning approach using domain-specific transformer models (LegalBERT and CaseLawBERT); a comparative evaluation with a state-of-the-art CRF model; a hybrid CNN+CRF architecture; and out-of-domain evaluation demonstrating robustness on Indian legal text.

### Conclusion

In this paper, we have presented a deep learning approach to address the challenging problem of detecting sentence boundaries in legal text. Our approach leverages the context surrounding the delimiter for detecting sentence boundaries in legal text as a sequence classification problem. We evaluated the performance of various deep learning models and identified the best-performing CNN model based on model size, F1 score, and inference time. Our evaluation results demonstrate the superiority of our proposed deep learning approach over the baselines, with significant improvements in the F1 score. To further improve performance, we incorporated the features of the best-performing CNN model into a statistical CRF model to obtain a hybrid architecture that leverages the strengths of both models. Our hybrid model achieved an F1 score improvement of over 4% compared to the individual models, demonstrating the effectiveness of combining deep learning models with traditional statistical models. The hybrid model also exhibits effectiveness and robustness in accurately detecting sentence boundaries in the context of Indian legal text.

Future work should assess how these tools impact downstream tasks, as a subtle enhancement in this preprocessing step can significantly impact the performance of subsequent NLP applications. The integration of rule-based approaches and ensemble techniques present promising opportunities to enhance SBD further. Extensions to multilingual legal corpora and languages lacking explicit sentence delimiters (such as Thai) are also noted as valuable future directions.

---

## 03_08 — A Multi-Modal Approach to Digital Document Stream Segmentation for Title Insurance Domain

**Authors:** Abhijit Guha, Abdulrahman Alahmadi, Debabrata Samanta, Mohammad Zubair Khan, Ahmed H. Alahmadi  
**Journal:** IEEE Access (2022), vol. 10, pp. 11341–11352  
**DOI:** https://doi.org/10.1109/ACCESS.2022.3144185

### Abstract

In the twenty-first century, storing and managing digital documents has become commonplace for all corporate and public sectors around the world. Physical documents are scanned in batches and stored in a digital archive as a heterogeneous document stream, referred to as a digital package. To make Robotic Process Automation (RPA) easier, it is necessary to automatically segment the document stream into a subset of independent, coherent multi-page documents by detecting the appropriate document boundary. The current study proposes, evaluates, and compares a multi-modal binary classification network incorporating text and image aspects of digital document pages to state-of-the-art baseline methodologies. Image and textual features are extracted simultaneously from the input document image by passing them through VGG16-CNN and a pre-trained Legal-BERTbase model through transfer learning respectively. Both features are finally fused and passed through a fully connected layer of Multi Layered Perceptron (MLP) to obtain the binary classification of the pages as First Page (FP) and Other Page (OP). Real-time document image streams from a production business process archive were obtained from a reputed Title Insurance (TI) company for the study. The obtained F1 scores of 97.37% and 97.15% are significantly higher than the accuracies of the two baseline models and well above the expected Straight Through Pass (STP) threshold.

### Introduction

Document Stream Segmentation (DSS) is the task of breaking a page stream into a set of documents. Multi-page digital documents arrive at the Document Management System (DMS) as an ordered set of digital images without any indication of document boundaries. Traditional DMSs needed human intervention to place page-break indicators at the source of digitization for machines to perform segmentation during real-time processing—a costly affair not feasible for all digitization sources. With AI and RPA integration, Automated Document Management Systems (ADMS) are fast replacing DMSs. Examination of digitized document packages comprised of multiple documents of varying length and quality is ubiquitous in a typical Title Insurance (TI) search and examination process. An efficacious DSS is necessary for any ADMS to be precise in the consecutive tasks, because any error occurring in the DSS has a rippling effect on subsequent modules.

The study was motivated by a real-time need for DSS in the TI industry. Prior state-of-the-art models achieved a maximum accuracy of only 86.7% on the proprietary TI data, which was significantly lower than expected. It has been empirically established that textual features have more prominence towards document image classification than image features alone. The proposed architecture replaces the text feature extractor with the pre-trained Legal-BERTbase model, adopting state-of-the-art transfer learning from both text and image modalities. To the authors' knowledge, no prior DSS study had been conducted to explore and validate data in the Title Insurance domain.

### Conclusion

We have proposed a multi-modal binary classification approach based on state-of-the-art transfer learning techniques involving images and NLP models to address the problem of DSS. Our study was motivated by a real-time need for DSS in the TI industry, and a proprietary dataset from a reputed TI company was considered. The proposed multi-modal approach has been proven to have performed significantly better than the present state-of-the-art model as well as the uni-modal NLP approach combined with transfer learning, achieving F1 scores of 97.37% and 97.15% for CP2020 and SP2020 archive data respectively. Adding the image modality improved the model performance by 1% and 0.15% for the two datasets respectively. The improvements are marginal compared to the computational complexity added by the second modality, but in a production environment where thousands of documents are processed daily, even a 1% upliftment in F1-score translates to a 2.32% improvement in STP, which cannot be ignored. Further research is in progress to confirm effectiveness of a deep RNN-based sequence model that exploits page-specific visual features such as font size, margin, font type, and logo presence for classifying document sequences.

---

## 03_09 — A domain-specific cross-lingual semantic alignment learning model for low-resource languages

**Authors:** Yurong Wang, Min Lin, Qitu Hu, Shuangcheng Bai, Yanling Li, Longjie Bao  
**Journal:** Neural Networks (2026), vol. 194, 108114  
**DOI:** https://doi.org/10.1016/j.neunet.2025.108114

### Abstract

Cross-lingual semantic alignment models facilitate the sharing and utilization of multilingual domain-specific data (e.g., medical, legal), offering cost-effective solutions for improving low-resource language tasks. However, existing methods are challenged by parallel data scarcity, semantic space heterogeneity, morphological complexity, and weak robustness—particularly for agglutinative languages. Therefore, this paper proposes CLWKD, a cross-lingual mapping and knowledge distillation framework. CLWKD leverages domain-specific pretrained models from high-resource languages as teachers and integrates multi-granularity alignment matrices with limited parallel data to guide cross-lingual knowledge transfer. CLWKD jointly learns multi-granularity semantic alignment mapping matrices at the token, word, and sentence levels from general-domain data. It eases domain data scarcity and helps bridge structural gaps caused by morphological and syntactic differences. To alleviate data sparsity and out-of-vocabulary issues in agglutinative languages, multilingual embedding sharing and morphological segmentation strategies are introduced. To improve the stability of unsupervised mapping training, generator pretraining is introduced and further combined with high-confidence word and sentence pairs to optimize the mapping matrix. To preserve alignment with fewer parameters, a parameter recycling and embedding bottleneck design is adopted. Experiments across the medical, legal, and educational domains on Mongolian-Chinese and Korean-Chinese language pairs demonstrate the effectiveness of CLWKD in three cross-lingual tasks.

### Introduction

In recent years, pretrained language models (PLMs) have achieved remarkable progress in open-domain tasks for high-resource languages. However, owing to the severe scarcity of both general-domain and domain-specific corpora, low-resource languages still exhibit limited capability in domain-specific text mining, such as in the medical and legal fields. Cross-lingual semantic alignment models facilitate the sharing and utilization of domain data across multiple languages, providing an effective and low-cost approach to improve the performance of deep learning models for low-resource languages in specific domains.

Current mainstream approaches to cross-lingual semantic alignment can be broadly categorized into three types: cross-lingual fine-tuning, cross-lingual mapping, and cross-lingual knowledge distillation. These challenges are particularly pronounced in China's multi-ethnic linguistic context, where languages such as Mongolian, Tibetan, Kazakh, and Uyghur suffer from severe data scarcity, strong agglutinative morphology, and flexible word order, which further complicate semantic modeling and cross-lingual knowledge transfer.

This paper proposes CLWKD, a cross-lingual mapping and knowledge distillation framework for multilevel semantic alignment in low-resource scenarios. CLWKD not only introduces a multi-granularity collaborative strategy for semantic alignment, but also, for the first time, combines a general-domain mapping matrix with domain-specific knowledge distillation to address modeling challenges in extremely low-resource language settings. The key components include: a domain-specific pretrained model from a high-resource language as the teacher network; multi-granularity alignment mapping matrices at the token, word, and sentence levels to bridge structural gaps; multilingual embedding sharing and morphological segmentation to address agglutinative language challenges; and a parameter recycling and embedding bottleneck design for efficient alignment.

### Conclusion

> **Note:** The conclusion section (Section 6) in the extracted PDF text was truncated due to a pagination issue in the source document. Only the opening sentence of the section is available.

In this paper, a cross-lingual multi-granularity semantic alignment model CLWKD for low-resource languages is proposed, which can be used for both word-level and sentence-level cross-lingual semantic alignment tasks. CLWKD is divided into two stages. In the first stage, multiple [*text truncated in source PDF*].
