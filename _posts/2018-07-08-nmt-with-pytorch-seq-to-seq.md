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

{{ toc_only }}

# Introduction

For almost 25 years, mainstream translation systems used [Statistical Machine Translation](https://en.wikipedia.org/wiki/Statistical_machine_translation). SMT
methods were not outperformed until 2016 when Google AI released [results of their Neural Machine Translation model](https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html)
and started to use it in Google Translation for 9 languages.

Important to notice, that GNMT result was mostly defined by a huge amount of training data and
extensive computational power, which makes impossible to reproduce this results for
individual researchers. However, ideas and techniques, which were used in this architecture,
were reused to solve many other problems: question answering, natural database interface,
speech-to-text and text-to-speech and so on. Therefore, any deep learning expert
can benefit from an understanding of how modern NMT works.

# Theory

Even though I am mostly a practitioner but I still prefer to have a solid
mathematical representation of any model I am working with. This allows maintaining
a correct level of abstract understanding of a problem which my model is solving.
You can skip this part and go to a [model definition](#model) if you prefer to start coding.
But I advise you to review the theoretical part to have a deep understanding of
the following implementation.

  >Given a sentence in source language $\textbf{x}$, the task is to infer a
  sentence in target language $\textbf{y}$ which maximizes conditional
  probability $p(\textbf{y}|\textbf{x})$:

  \begin{equation}
  \textbf{y} = \underset{\hat{\textbf{y}}}{\mathrm{argmax}} p(\hat{\textbf{y}}|\textbf{x})
  \end{equation}

Clearly, this equation can not be used as is: there are an infinite number of all possible
$\hat{\textbf{y}}$. But since sentence $\textbf{y}$ is a sequence of words $y_1, y_2, \ldots ,y_{t-1}, y_t$, we can decompose
this probability as a product of probabilities for each word:

  \begin{equation}
  p(\textbf{y}|\textbf{x}) = \prod_{i=1}^{t} p(y_i| \textbf{x})
  \end{equation}

Words probabilities in $\textbf{y}$ are not distributed independently; in the natural
language phrases and sentences are usually follow strict or non-strict rules for word selection. Therefore,
conditional probability for each word should also include other words from a target sentence:

  \begin{equation}
  p(\textbf{y}|\textbf{x}) = \prod_{i=1}^{t} p(y_i|y_{t}, y_{t-1}, \ldots, y_{i+1}, y_{i-1}, \ldots, y_{2}, y_{1}, \textbf{x})\label{xy}
  \end{equation}

Using a predefined set of words or vocabulary $\textbf{V}$ we can infer from it
a target sentence $textbf{y}$ word by word using this probability: $p(y_i|y_{j\ne i}, \textbf{x})$.
The only question is how to get this probability? We can approximate it with a neural model,
training it weights.

>Let's denote weights of neural model as $\theta$ and training data as a set of
tuples $(\textbf{y}^{(x)}, \textbf{x})$, where each $\textbf{y}^{(x)}$ is a correct
translation of $\textbf{x}$. Optimal values of $\theta$ can be found by maximization
of a following equation:

  \begin{equation}
  \theta = \underset{\theta}{\mathrm{argmax}} \sum_{\textbf{y}^{(x)}, \textbf{x}} log p(\textbf{y}^{(x)}| \textbf{x}; \theta)
  \label{mle}
  \end{equation}

\ref{mle} is basically a [log-likelihood](http://mathworld.wolfram.com/Log-LikelihoodFunction.html) for a training data.
Likelihood maximization means that we train a neural network to produce probabilities, which makes training set $(\textbf{y}^{(x)}, \textbf{x})$ results more likely than any other results.


# Model

From equation \ref{xy} it is possible to see, what are the main inputs of a neural model for NMT.
To infer a word $y_i$ we need to provide a model information about previous words $y_{<i}$ (since we
don't have next words at the moment when we inferring $y_i$, we can only use previous words) and information about a source sentence
$\textbf{x}$.

The source sentence is encoded word for word by a [**Recurrent Neural Network**](https://en.wikipedia.org/wiki/Recurrent_neural_network), which is called **Encoder**. Then the last hidden state of the Encoder is used as the first hidden state of another RNN. This network decodes a target sentence word for word, and this network is obviously called **Decoder**. This way a shared hidden state between Encoder and Decoder is used to store all the information network need to produce a valid probability distribution for $y_i$.

![Encoder-decoder](/assets/images/nmt/Seq2Seq thought vector.png){:width="800px"}
*Encoder-Decoder with a thought vector*

# Implementation

## Data preparation

For this tutorial we will use bilingual datasets from [tatoeba.org](https://tatoeba.org/eng/downloads).
You can download language pairs data from http://www.manythings.org/anki/, or if there is no pair,
as for Ukrainian-German, you can use my script [get_dataset.py](https://github.com/tsdaemon/neural-experiments/blob/master/nmt/scripts/get_dataset.py). It will download raw data from
tatoeba and extract a bilingual dataset as a `csv` file.


To train neural network we need to turn the sentences into something the neural network can understand, which of course means numbers. Each sentence will be split into words and turned into a sequence of numbers. To do this, we will use a vocabulary – a class which will store word indices.  

```python
SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

class Vocab:
    def __init__(self):
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3: "PAD"}
        self.word2index = {v: k for k, v in self.index2word.items()}

    def index_words(self, words):
        for word in words:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            n_words = len(self)
            self.word2index[word] = n_words
            self.index2word[n_words] = word

    def __len__(self):
        assert len(self.index2word) == len(self.word2index)
        return len(self.index2word)

    def restore_words(self, indices):
        return ' '.join([self.index2word[i] for i in indices])
```

Define tokenizer functions for you languages to split sentences on words.
In this examples I'm using standard [`nltk.tokenize.WordPunctTokenizer`](https://kite.com/python/docs/nltk.tokenize.WordPunctTokenizer) for German and
a function `tokenize_words` from package [`tokenize_uk`](https://github.com/lang-uk/tokenize-uk) for Ukrainian.

Since there are a lot of example sentences and we want to train something quickly, we'll trim the data set to only relatively short and simple sentences. Here the maximum length is 8 words (that includes punctuation) and we're filtering to sentences that translate to the form "I am" or "He is" etc.

```python
source_tokenizer = tokenize_words
target_tokenizer = nltk.tokenize.WordPunctTokenizer().tokenize

MAX_LENGTH = 8

def read_langs(source_lang, source_tokenizer, target_lang, target_tokenizer, input_file):
    corpora = pd.read_csv(input_file, delimiter='\t')

    source_vocab = Vocab()
    target_vocab = Vocab()

    source_corpora = []
    target_corpora = []
    for i, row in tqdm(corpora.iterrows()):
        source_sent = row['text'+source_lang].lower().strip()
        target_sent = row['text'+target_lang].lower().strip()

        source_tokenized = source_tokenizer(source_sent)
        target_tokenized = target_tokenizer(target_sent)
        if len(source_tokenized) > MAX_LENGTH or \
           len(target_tokenized) > MAX_LENGTH:
            continue

        source_corpora.append(source_tokenized)
        target_corpora.append(target_tokenized)

        target_vocab.index_words(target_tokenized)
        source_vocab.index_words(source_tokenized)
    return source_vocab, target_vocab, list(zip(source_corpora, target_corpora))

source_vocab, target_vocab, corpora = read_langs(source_lang, source_tokenizer, target_lang, target_tokenizer, file_name)
```

PyTorch uses it's own format of data – Tensor. A Tensor is a multi-dimensional array of numbers with some type e.g. FloatTensor or LongTensor. Before we can use our training data, we need to convert it into tensors using previously defined word indices.
Additionaly, we need to add special tokens SOS (start of sentence) and EOS (end of sentence) to each sentence.
Also, all sentences should have the same length to make possible batch training, therefore we will extend them with token PAD if needed.

```python
MAX_SEQ_LENGTH = MAX_LENGTH+2 # 2 for EOS_token and SOS_token

def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    indexes.insert(0, SOS_token)
    # we need to have all sequences the same length to process them in batches
    if len(indexes) < MAX_SEQ_LENGTH:
        indexes += [PAD_token]*(MAX_SEQ_LENGTH-len(indexes))
    tensor = torch.LongTensor(indexes)
    if USE_CUDA: var = tensor.cuda()
    return tensor

def tensors_from_pair(source_sent, target_sent):
    source_tensor = tensor_from_sentence(source_vocab, source_sent).unsqueeze(1)
    target_tensor = tensor_from_sentence(target_vocab, target_sent).unsqueeze(1)

    return (source_tensor, target_tensor)

tensors = []
for source_sent, target_sent in corpora:
    tensors.append(tensors_from_pair(source_sent, target_sent))

x, y = zip(*tensors)
x = torch.transpose(torch.cat(x, dim=-1), 1, 0)
y = torch.transpose(torch.cat(y, dim=-1), 1, 0)
```
