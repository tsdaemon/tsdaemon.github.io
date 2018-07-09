---
layout: post
title:  "Neural Machine Translation with PyTorch. Part 1: Encoder-decoder"
date:   2018-07-08 15:37:21 0300
categories: neural machine translation, deep learning, natural language processing, seq2seq
---
***Recently I did a [3-day workshop about Deep Learning for Natural Language Processing](https://github.com/tsdaemon/dl-nlp-2018).
Clearly 3 days was not enough to cover all topics in this broad field, therefore
I decided to create a series of tutorials in [PyTorch](https://pytorch.org/),
in which I will start with a simple neural translation model and gradually improve it using
modern neural methods and techniques.***

*In this first tutorial, I am going to create Ukrainian-German translator with
simple encoder-decoder model.*

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
