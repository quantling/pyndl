---
title: 'pyndl: Naïve Discriminative Learning in Python'
tags:
  - Delta-rule
  - Rescorla-Wagner
  - NDL
  - Naïve Discrimination Learning
  - Language model
authors:
  - name: PLACEHOLDER
    affiliation: 1
affiliations:
  - name: PLACEHOLER INSTITUTION
    index: 1    
date: 31 March 2022
bibliography: paper.bib
repository: quantling/pyndl
---

# Summary

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience --> 
*pyndl* implements Naïve Discriminative Learning (NDL) in Python. NDL is an incremental learning algorithm grounded in the principles of discrimination learning [@rescorla1972theory] and motivated by animal and human learning research [@bla]. Lately, NDL became a popular tool in language research to examine large corpora [@bla]. *pyndl* implements this algorithm in an openly available Python package. In contrast to previous implementations, our implementation allows for a wider range of analysis, adds further learning rules and provides a better maintainability while having a comparable processing speed. As of today, it supported multiple research groups in their work and led to several scientific publications.


# Statement of need

<!-- General problem --> 
Naïve Discriminative Learning (NDL) is a computational modeling framework that phrases language comprehension as a multiple label classification problem [@Baayen_2011;@sering2018language]. It is grounded in simple but powerful principles of discrimination learning [@bla]. In pratical terms, it is based on a 2-layer symbolic network motivated by the equilibrium equations of the Rescorla–Wagner model [@Danks_2003,@rescorla1972theory] and the assumption that language is learned over time. NDL models are trained on large corpora to investigate the language's structure [@bla] and to provide insights into the learning process [@bla].


<!-- Which implementations are out there? --> 
Several implemtations of NDL have been created over time. The first implementation that was available, was the R package *ndl* [@ndl] which could solve the Danks equilibrium equations, but did not provide an iterative exact solver. This feature has been added in the R package *ndl2* [@ndl2]. However, the code of *ndl2* is only available through the maintainer. One reason for this has been that it only runs on Linux and CRAN's guidlines make it difficult to publish single platform packages. A severe limitation of *ndl2* is that its input is restricted to ascii, causing problems in the analysis of non-english text corpora due to special characters or non-latin alphabets. Futhermore, with *ndl2* it is not possible to conveniantly compute huge weight matrices due to a size limitation of arrays in R.


# Implementation and use in research

<!-- Short description of pyndl --> 
In *pyndl*, we reimplemented the learning rule of NDL mainly in Python with small code junks outsourched to Cython to speed up the processing. We also implemented the processing of UTF-8 coded corpora enabling the analysis of many non-european languages and alphabets (e.g. Cyrilic [@milin2020keeping], Mandarin). Using the python ecosystem, the size of weight matrices is only limited by the memory available. While previous packages were restricted in functionality and partially not openly available, *pyndl* was developed with hindsight to usability and maintainability. We also aimed to provide the same functionality as the previous R packages in Python. If *pyndl* is installed it can be used in R or Julia and recepies on how to use it are documented.

<!-- Pyndl in research --> 
*pyndl* is used by several research groups to analyse language data and is regularly used in schorlaly work.
These works use *pyndl* to for models in a wide range of linguistic subfields  and explicitly make use of the easy extensibility and UTF-8 support of *pyndl*. 
@TOMASCHEK_2019 and @Baayen_2020 investigate morphological effects with context in German and English language, while @Shafaei_Bajestan_2018 and @Sering_2018 
show auditory comprehension based on simple acustic features, and @Romain_2022 model the learning of tenses. 
@milin2020keeping showed different language phenomena, among others using Cyrilic cues, with the simple Widrow-Hoff learnung rule as a special case of the Rescorla-Wagner rule. 
This Rescorla-Wagner rule was extended and compared with a linearized version by @Shafaei_Bajestan_2021, while @Divjak_2020 showed the benefits of learning language models over probabilistic and rule-based models.


# Acknowledgements

TODO: ADD MISSING


This research was supported by an ERC advanced Grant (no. 742545) and by the Alexander von Humboldt Professorship awarded to R. H. Baayen and by the University of Tübingen.

# References
