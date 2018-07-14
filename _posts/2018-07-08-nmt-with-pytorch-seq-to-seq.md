---
layout: post
current: post
navigation: True
title:  "Neural Machine Translation With PyTorch. Tutorial 1: Encoder-decoder"
date:   2018-07-08 15:37:21 +0300
tags: nmt, dl, PyTorch
class: post-template
subclass: 'post'
author: anatolii
---

**Recently I did a [3-day workshop about Deep Learning for Natural Language Processing](https://github.com/tsdaemon/dl-nlp-2018).
Clearly 3 days was not enough to cover all topics in this broad field, therefore
I decided to create a series of practical tutorials about Neural Machine Translation
in [PyTorch](https://pytorch.org/).**

*I will start with a simple neural translation model and gradually improve it using
modern neural methods and techniques. In this first tutorial, I am going to create
Ukrainian-German translator with an encoder-decoder model.*

# Introduction

For almost 25 years, mainstream translation systems used [Statistical Machine Translation](https://en.wikipedia.org/wiki/Statistical_machine_translation). SMT
methods were not outperformed until 2016, when Google AI released [results of their Neural Machine Translation model](https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html)
and started to use it in Google Translation for 9 languages.

Important to notice, that GMNT result was mostly defined by a huge amount of tranining data and
extensive computational power, which makes impossible to reproduce this results for
individual researchers. However, ideas and techniques, which were used in this architecture,
were reused to solve many other problems: question answering, natural database interface,
speech-to-text and text-to-speech and so on. Therefore, any deep learning expert
can benefit from understanding of how modern NMT works.

# Theory

Even though I am mostly practitioner but I still prefer to have a solid
mathematical representation of any model I am working with. This allows to maintain
a correct abstract understanding of problem which my model is solving. Therefore I will
start with formulation of NMT problem:

  Given a sentence in source language $\textbf{x}$, the task is to infer a
  sentence in target language $\textbf{y}$ which maximizes conditional
  probability $p(\textbf{y}|\textbf{x})$:

  \begin{equation}
  \textbf{y} = \underset{\hat{\textbf{y}}}{\mathrm{argmax}} p(\hat{\textbf{y}}|\textbf{x})
  \end{equation}
