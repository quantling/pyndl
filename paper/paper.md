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
    affiliation: 1, 2
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

<!-- A summary describing the high-level functionality and purpose of the
software for a diverse, non-specialist audience -->
The *pyndl* package implements Naïve Discriminative Learning (NDL) in Python. NDL is an
incremental learning algorithm grounded in the principles of discrimination
learning [@rescorla1972theory; @widrow1960adaptive] and motivated by animal and
human learning research [e.g. @Baayen_2011; @Rescorla1988PavlovianCI]. Lately,
NDL has become a popular tool in language research to examine large corpora and
vocabularies, with 750,000 spoken word tokens [@Shafaei_2022] and a vocabulary
size of 52,402 word types [@Sering_2018]. In contrast to previous
implementations, *pyndl* allows for a broader range of analysis, including
non-English languages, adds further learning rules and provides better
maintainability while having the same fast processing speed. As of today, it
supports multiple research groups in their work and led to several scientific
publications.


# Statement of need

<!-- General problem -->
Naïve Discriminative Learning (NDL) is a computational modelling framework that
phrases language comprehension as a multiple label classification problem
[@Baayen_2011;@Sering_2018]. It is grounded in the simple but powerful principles
of discrimination learning implemented on a 2-layer symbolic network combined
with the Rescorla-Wagner learning rule [@rescorla1972theory].
This learning rule explains phenomena, where animals associate co-occurring
cues and outcomes, e.g. a flashing light and food, only if the cue is effective
in predicting the outcome [see @Baayen_2015, for an introduction].
<!-- Alternatively, the final state of the learning weights in the network is
computable by equilibrium equations [@Danks_2003,@rescorla1972theory]. The
underlying assumption is that language is learned over time. -->
In linguistics, NDL models are trained on large corpora to investigate the language's
structure and to provide insights into the learning process of life long language
acquisition [please find an extensive list of examples in @Baayen_2015, section 5].


<!-- Which implementations are out there? -->
Several implementations of NDL have been created over time but are struggling
with modern challenges of linguistic research like multi-language support,
increasing model sizes, and open science principles. The first implementation
was the R package *ndl* [@ndl], which could solve the Danks equilibrium
equations [@Danks_2003], but did not provide an exact iterative solver. An
iterative solver was added to the R package *ndl2* [@ndl2]. *ndl* and *ndl2*
made efficient implementations to learning algorithms available to language
researchers.


<!-- Differences to pyndl -->
<!-- Problems of ndl and ndl2 -->
However, the code of *ndl2* was only available upon request until end of the
year 2022.  One reason for this has been that it only runs on Linux and CRAN's
guidelines make it difficult to publish single platform packages [@R_project].
Another reason is the limited maintainability through low level C and C++ code
next to high level R code.  A severe limitation of *ndl* and *ndl2* is that
both packages have difficulties with non-ASCII input, causing problems in the
analysis of non-English text corpora due to special characters or non-Latin
alphabets. An example would be the processing of Arabic or Slavic languages;
even German umlauts are inconvenient to use in *ndl* and *ndl2*. Furthermore,
in *ndl2*, it is impossible to conveniently access huge weight matrices due to
a size limitation of arrays in the R programming language [@R_project]. This
limit does not allow for more than 46,340 word types in a word-type to
word-type model, which is too small to capture the full lexicon in most
languages.


# Implementation and use in research

<!-- Short description of pyndl -->
*pyndl* reimplements the learning rule of NDL mainly in Python with small code
chunks outsourced to Cython to speed up the processing. This allows the
processing of UTF-8 encoded corpora enabling the analysis of many non-European
languages and alphabets [e.g. Mandarin or Cyrillic, @milin2020keeping]. Using
the Python ecosystem, the size of weight matrices in *pyndl* is only limited by
the memory available. Computed weights using the *xarray* format
[@hoyer2017xarray] can be easily integrated into down-stream tasks like
analyzing the association strength between grapheme clusters a word types.

The input to *pyndl* is agnostic to the actual domain as long as it is
tokenized as Unicode character strings. Input sequences can consist of multiple
tokens separated by underscores which is together with the tab-character the
only special character in *pyndl*. While *pyndl* provides some basic
preprocessing for grapheme tokenization, the preprocessing of ideograms,
pictograms, logograms, and speech is possible using custom preprocessing. For
example, word classification using tokenized speech audio recordings was
investigated in @Arnold_2017. Inputs in this work consisted of around 50 tokens
per time slice, where each token encoded the pitch, loudness, and variability
into a string.

The input format is based on previous implementations of NDL. In contrast to
previous implementations, *pyndl* was open-source software from the beginning and
developed with usability and maintainability in mind.  The better
maintainability of *pyndl* does not come at the cost of performance: The
benchmark results in \autoref{fig:benchmark}, described in detail in our
package's documentation, shows that *pyndl* is faster than *ndl* and *ndl2*.
Memory is used efficiently by storing data records in compressed form and
loading data points as they are needed during learning.  *pyndl* provides the
same core functionality as the previous R packages in Python. After
installation, *pyndl* can be called from R or Julia scripts by convenient
bridges, like any Python library. An example on how to use *pyndl* from R can
be found in our documentation.

![Execution wall time for different implementations of the Rescorla-Wagner
learning rule for different number of learning events. The mean and
standard-error ($n$=10) for the wall time show that our implementation,
*pyndl*, is the fastest for larger numbers of
events.\label{fig:benchmark}](benchmark_result.png)

<!-- WH extension of pyndl -->
The improved maintainability of *pyndl* also allows for an easier addition of new 
features. For example, the NDL learner was extended to a learner for continuous 
inputs as cues, outcomes or both.  When both cues and outcomes are continuous, 
the Rescorla-Wagner learning rule changes to the Widrow-Hoff learning rule. This
extension is added by keeping the API to the learner comparable to NDL and
computationally exploiting the structure of the multi-hot encoded features in
the symbolic representation of language.

<!-- Pyndl in research -->
*pyndl* is used by several research groups to analyse language data and is
regularly used in scholarly work.  These works use *pyndl* for models in a wide
range of linguistic subfields and explicitly use the easy extensibility and
UTF-8 support of *pyndl*.  @TOMASCHEK_2019 and @Baayen_2020 investigate
morphological effects of the context of German and English languages, while
@Shafaei_Bajestan_2018 and @Sering_2018 show auditory comprehension based on
simple acoustic features, and @Romain_2022 model the learning of tenses.
@milin2020keeping profited from *pyndl*'s UTF-8 support when using Cyrillic
cues to show different language phenomena with the simple Widrow-Hoff learning
rule as a special case of the Rescorla-Wagner. @Tomaschek:Duran:2019 used
*pyndl* to model the McGurk effect across different languages.
@Shafaei_Bajestan_2021 presented a linearized version of the Rescorla-Wagner
rule that they could add to *pyndl* and compare with the classic version.
@Divjak_2020 showed the benefits of learning language models over probabilistic
and rule-based models.


# Acknowledgements

The authors thank R. Harald Baayen for his support in creating and maintaining
*pyndl* as a scientific software package in the Python ecosystem.
Furthermore, the authors thank Lennart Schneider for his major contributions, as
well as all [other contributors on
GitHub](https://github.com/quantling/pyndl/graphs/contributors).

This research was supported by an ERC advanced Grant (no. 742545) and by the
Alexander von Humboldt Professorship awarded to R. Harald Baayen and by the
University of Tübingen.

The authors thank the International Max Planck Research School for Intelligent
Systems (IMPRS-IS) for supporting David-Elias Künstle,
funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)
under Germany’s Excellence Strategy – EXC number 2064/1 – Project number
390727645 and the High North Population Studies at UiT The Arctic University of
Norway for supporting Marc Weitz.

Finally, this paper and the associated package benefited from constructive and
conscientious peer review. We would like to thank our three reviewers, Venktesh
V, Jinhang Jiang, and especially Frankie Robertson, for their constructive and
in-depth feedback and their suggestions on how to make the package more
user-friendly.

# References
