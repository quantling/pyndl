---
title: 'pyndl: Naïve Discriminative Learning in Python'
tags:
  - Delta-rule
  - Rescorla-Wagner
  - NDL
  - Naïve Discrimination Learning
  - Language model
authors:
  - name: Konstantin Sering
    affiliation: 1
  - name: Marc Weitz
    affiliation: 2
  - name: Elnaz Shafaei-Bajestan
    affiliation: 1
  - name: David-Elias Künstle
    affiliation: 1, 3
affiliations:
  - name: University of Tübingen
    index: 1
  - name: UiT The Arctic University of Norway
    index: 2
  - name: International Max Planck Research School for Intelligent Systems
    index: 3
date: 31 March 2022
bibliography: paper.bib
repository: quantling/pyndl
---

# Summary

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience -->
*pyndl* implements Naïve Discriminative Learning (NDL) in Python. NDL is an incremental learning algorithm grounded in the principles of discrimination learning [@rescorla1972theory; @widrow1960adaptive] and motivated by animal and human learning research [e.g. @Rescorla1988PavlovianCI,@Baayen_2011]. Lately, NDL has become a popular tool in language research to examine large corpora and vocabularies, even with 750,000 spoken word tokens [@bla Elnaz] and a vocabulary size of 52,402 word types [@sering2018language]. *pyndl* implements NDL in an openly available Python package. In contrast to previous implementations, ours allows for a broader range of analysis, including non-English languages, adds further learning rules and provides better maintainability while having the same fast processing speed. Today, it supports multiple research groups in their work and led to several scientific publications.


# Statement of need

<!-- General problem -->
Naïve Discriminative Learning (NDL) is a computational modelling framework that phrases language comprehension as a multiple label classification problem [@Baayen_2011;@sering2018language]. It is grounded in simple but powerful principles of discrimination learning implemented on a 2-layer symbolic network combined with the Rescorla-Wagner learning rule [@rescorla1972theory].
This learning theory explains phenomena, where animals only associates co-occurring cues and outcomes, e.g. a flashing light and food, if the cue is effective in predicting the outcome [see @Baayen_2015 for an introduction].
Alternatively, the final state of the learning weights in the network is computable by equilibrium equations [@Danks_2003,@rescorla1972theory]. The underlying assumption is that language is learned over time. NDL models are trained on large corpora to investigate the language's structure and to provide insights into the learning process [please find an extensive list of examples in @Baayen_2015
, section 5].


<!-- Which implementations are out there? -->
Several implementations of NDL have been created over time but are struggling with modern challenges of linguistic research like multi-language support, increasing model sizes, and open science principles. The first implementation was the R package *ndl* [@ndl], which could solve the Danks equilibrium equations, but did not provide an exact iterative solver. This solver has been added to the R package *ndl2* [@ndl2]. However, the code of *ndl2* is only available upon request. One reason for this has been that it only runs on Linux and CRAN's guidelines make it difficult to publish single platform packages. A severe limitation of *ndl* and *ndl2* is that both packages have difficulties with non-ASCII input, causing problems in the analysis of non-English text corpora due to special characters or non-Latin alphabets. Furthermore, in *ndl2*, it is impossible to conveniently access huge weight matrices due to a size limitation of arrays in R.


# Implementation and use in research

<!-- Short description of pyndl -->
In *pyndl*, we reimplemented the learning rule of NDL mainly in Python with small code chunks outsourced to Cython to speed up the processing. We also implemented the processing of UTF-8 coded corpora enabling the analysis of many non-European languages and alphabets (e.g. Cyrillic [@milin2020keeping], Mandarin). Using the python ecosystem, the size of weight matrices in *pyndl* is only limited by the memory available. While previous packages were restricted in functionality and partially not openly available, *pyndl* was developed with hindsight for usability and maintainability. We also aimed to provide the same functionality as the previous R packages in Python. After installation, *pyndl* can be called from R or Julia scripts by convenient bridges.

<!-- WH extension of pyndl -->
The design decision to make *pyndl* maintainable allowed for an extension of the NDL learner to a learner that allows for continuous inputs either as cues or outcomes or both. When both cues and outcomes are continuous, the Rescorla Wagner learning rule changes to the Widrow-Hoff learning rule. This extension is added by keeping the API to the learner comparable to NDL and computationally exploiting the structure of the binary multi-hot features in the symbolic representation of language.

<!-- Pyndl in research -->
*pyndl* is used by several research groups to analyse language data and is regularly used in scholarly work.
These works use *pyndl* for models in a wide range of linguistic subfields and explicitly use the easy extensibility and UTF-8 support of *pyndl*.
@TOMASCHEK_2019 and @Baayen_2020 investigate morphological effects of the context of German and English languages, while @Shafaei_Bajestan_2018 and @Sering_2018
show auditory comprehension based on simple acoustic features, and @Romain_2022 model the learning of tenses.
@milin2020keeping profited from *pyndl*'s UTF-8 support when using Cyrillic cues to show different language phenomen with the simple Widrow-Hoff learning rule as a special case of the Rescorla-Wagner. @Tomaschek:Duran:2019 used *pyndl* to model the McGurk effect accross different lanuages.
@Shafaei_Bajestan_2021 presented a linearised version of the Rescorla-Wagner rule that they could add to *pyndl* and compare with the classic version. @Divjak_2020 showed the benefits of learning language models over probabilistic and rule-based models.


# Acknowledgements

The authors thank R. Harald Baayen for his support in creating and maintaining a scientific software package in the Python ecosystem.
Finally, the authors thank Lennart Schneider for his major contributions, as well as all [other contributors on GitHub](https://github.com/quantling/pyndl/graphs/contributors).

This research was supported by an ERC advanced Grant (no. 742545) and by the Alexander von Humboldt Professorship awarded to R. Harald Baayen and by the University of Tübingen.
**TODO: CURRENT FINANCIAL SUPPORT**


# References

