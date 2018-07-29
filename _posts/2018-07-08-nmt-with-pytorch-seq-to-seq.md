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
toc: true
---

*Recently I did a [3-day workshop about Deep Learning for Natural Language Processing](https://github.com/tsdaemon/dl-nlp-2018).
Clearly, 3 days was not enough to cover all topics in this broad field, therefore
I decided to create a series of practical tutorials about Neural Machine Translation
in [PyTorch](https://pytorch.org/). In this series, I will start with a simple neural
translation model and gradually improve it using
modern neural methods and techniques. In the first tutorial, I will create
Ukrainian-German translator with an encoder-decoder model.*

{::nomarkdown}
{% jupyter_notebook encoder-decoder.ipynb %}
{:/nomarkdown}
