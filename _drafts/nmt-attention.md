---
layout: post
title:  "Neural Machine Translation With PyTorch"
subtitle: "Tutorial 2: Attention"
class: post-template
subclass: 'post'
comments: true
---

*In [the previous tutorial]({% post_url 2018-07-08-nmt-with-pytorch-encoder-decoder %}) I have explained Neural Machine Translation problem and described example of the most simple neural network to solve it: Encoder-Decoder with a thought vector. This approach is already obsolete now
(yeah, this industry is fast) and I used it only for educational purpose.*

*Modern Seq2seq networks use attention to pass information from encoder to decoder. This great technique significantly improves translation accuracy and, what is also important, it helps to interpret model results.*

# Introduction

Theoretically, a sufficiently large encoder-decoder model should be able to perform machine translation perfectly. However, to encode all words and their dependencies in the arbitrary-length sentences, the [thought vector](https://en.wikipedia.org/wiki/Thought_vector) should have enormous length. Such a model would require massive computational resources to train and to use, therefore this approach is ineffective.

This problem can be solved with attention technique. This wonderful idea was first presented in a work of Bahdanau, Cho and Bengio "Neural Machine Translation by Jointly Learning to Align and Translate"[^bengio2014]. The idea is to replace a single vector representation of an input sentence with references to representations of different words in this sentence.


[^bengio2014]: [Bahdanau et al, 2014](https://arxiv.org/abs/1409.0473)

## Theory

During encoding, each word representation $\textbf{h}_{x}^{(t)}$ is stored as a column of a matrix $\textbf{H}_x$. During a decoding step, each decoder input is extended with a context vector $\pmb{\phi}_t$:

\begin{equation}
\pmb{h}_y^{(t)}=rnn([\pmb{y}_t, \pmb{\phi}_t], \pmb{h}_y^{(t-1)})
\end{equation}

Context vector $\pmb{\phi}_t$ is calculated as a weighted sum of encoder representations:
\begin{equation}
	\pmb{\phi}_t = \pmb{H}_x\cdot \pmb{\alpha}_t
	\label{attn:phi}
\end{equation}

Weights for the attention vector $\pmb{\alpha}_t$ can be calculated using an attention score function. This can be any function, which accepts two vectors as inputs and outputs a single scalar value (for example, vector product). In this tutorial I use DNN with one hidden layer as suggested in *Bahdanau et al*[^bengio2014].

Attention function is applied to each pair of the decoder vector $\pmb{h}_y^{(t-1)}$ and an encoder vector $\pmb{h}_x^{(i)}, \forall i \in \pmb{X}$, where $\pmb{X}$ is the input sequence:


$$\hat{\pmb{\alpha}}_{(t,i)} = \textbf{W}_{attn1} \cdot [\pmb{h}_{x}^{(i)}, \pmb{h}_y^{(t-1)}] \\
  \alpha_{(t,i)} = \textbf{W}_{attn2} \cdot tanh(\hat{\pmb{\alpha}}_{(t,i)}) \\
  \pmb{\alpha}_t = softmax([\alpha_{(t,i)} \forall i \in \pmb{X}])$$

Above $W_{attn1}$ is a matrix of size $e+d \times n$ where $e$ is a size of encoder hidden vector $\pmb{h}_x$, and $d$ is a size of decoder hidden vector $\pmb{h}_y$. $\textbf{W}_{attn2}$ is a matrix of size $n \times 1$ and $\alpha_{(t,i)}$ is a single scalar value, which represent similarity measure between $\pmb{h}_{x}^{(i)}$ and $\pmb{h}_y^{(t-1)}$. $\pmb{\alpha}_t$ is a result of [softmax](https://en.wikipedia.org/wiki/Softmax_function) operation over all similarities. Softmax squeezes similarities, that they sum up to one, but still stores relative difference between their values. This way $\pmb{\phi}_t = \pmb{H}_x\cdot \pmb{\alpha}_t$ remains at the same scale as $\pmb{h}_{x}$.

Using context vector each decoder step can use information from any part of the encoded sequence. The input is not limited to the fixed length thought vector, and it can store information about any sequence size and any vocabulary size.

![Encoder-decoder-attention](/assets/images/nmt/seq2seq-attention.png){:width="800px"}
*Encoder-Decoder with attention*
