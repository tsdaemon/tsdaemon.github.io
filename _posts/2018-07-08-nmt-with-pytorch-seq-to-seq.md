---
layout: post
current: post
navigation: True
title:  "Neural Machine Translation With PyTorch"
subtitle: "Tutorial 1: Encoder-decoder"
date:   2018-07-08 15:37:21 +0300
tags: nmt dl PyTorch
class: post-template
subclass: 'post'
author: anatolii
---

*Recently I did a [3-day workshop about Deep Learning for Natural Language Processing](https://github.com/tsdaemon/dl-nlp-2018).
Clearly 3 days was not enough to cover all topics in this broad field, therefore
I decided to create a series of practical tutorials about Neural Machine Translation
in [PyTorch](https://pytorch.org/). In this series I will start with a simple neural
translation model and gradually improve it using
modern neural methods and techniques. In the first tutorial I will create
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
a correct abstract understanding of problem which my model is solving.
You can skip this part, if you prefer to start coding already,
but I advise you to review it to have a deep understanding of the following implementation.

  >Given a sentence in source language $\textbf{x}$, the task is to infer a
  sentence in target language $\textbf{y}$ which maximizes conditional
  probability $p(\textbf{y}|\textbf{x})$:

  $$
  \textbf{y} = \underset{\hat{\textbf{y}}}{\mathrm{argmax}} p(\hat{\textbf{y}}|\textbf{x})
  $$

Clearly, this equation can not be used as is: there are an infinite number of all possible
$\hat{\textbf{y}}$. But since sentence $\textbf{y}$ is a sequence of words $y_1, y_2, \ldots ,y_{t-1}, y_t$, we can decompose
this probability as a product of probabilities for each word:

  $$
  p(\textbf{y}|\textbf{x}) = \prod_{i=1}^{t} p(y_i| \textbf{x})
  $$

Words probabilities in $\textbf{y}$ are not distributed independently; in the natural
language phrases and sentences are usually follow strict or non-strict rules for word selection. Therefore,
conditional probability for each word should also include other words from a target sentence:

  $$
  p(\textbf{y}|\textbf{x}) = \prod_{i=1}^{t} p(y_i|y_{t}, y_{t-1}, \ldots, y_{i+1}, y_{i-1}, \ldots, y_{2}, y_{1}, \textbf{x})
  $$
