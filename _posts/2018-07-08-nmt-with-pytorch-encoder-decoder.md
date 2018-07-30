---
layout: post
title:  "Neural Machine Translation With PyTorch"
subtitle: "Tutorial 1: Encoder-decoder"
date:   2018-07-08 15:37:21 +0300
class: post-template
subclass: 'post'
comments: true
---

*Recently I did a [workshop about Deep Learning for Natural Language Processing](https://github.com/tsdaemon/dl-nlp-2018).
Clearly, 3 days was not enough to cover all topics in this broad field, therefore
I decided to create a series of practical tutorials about Neural Machine Translation
in [PyTorch](https://pytorch.org/). In this series, I will start with a simple neural
translation model and gradually improve it using
modern neural methods and techniques. In the first tutorial, I will create
Ukrainian-German translator with an encoder-decoder model.*

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

![Encoder-decoder](/assets/images/nmt/Seq2Seq-thought-vector.png){:width="800px"}
*Encoder-Decoder with a thought vector*


# Implementation

## Data preparation

For this tutorial we will use bilingual datasets from [tatoeba.org](https://tatoeba.org/eng/downloads).
You can download language pairs data from http://www.manythings.org/anki/, or if there is no pair,
as for Ukrainian-German, you can use my script [get_dataset.py](https://github.com/tsdaemon/neural-experiments/blob/master/nmt/scripts/get_dataset.py). It will download raw data from
tatoeba and extract a bilingual dataset as a `csv` file.


```python
import pandas as pd
import os

source_lang = 'ukr'
target_lang = 'deu'
data_dir = '../../neural-experiments/nmt/data/'

corpus = pd.read_csv(os.path.join(data_dir, '{}-{}.csv'.format(source_lang, target_lang)), delimiter='\t')
```

To train neural network we need to turn the sentences into something the neural network can understand, which of course means numbers. Each sentence will be split into words and turned into a sequence of numbers. To do this, we will use a vocabulary – a class which will store word indexes.  


```python
SOS_token = '<start>'
EOS_token = '<end>'
UNK_token = '<unk>'
PAD_token = '<pad>'

SOS_idx = 0
EOS_idx = 1
UNK_idx = 2
PAD_idx = 3

class Vocab:
    def __init__(self):
        self.index2word = {
            SOS_idx: SOS_token,
            EOS_idx: EOS_token,
            UNK_idx: UNK_token,
            PAD_idx: PAD_token
        }
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

    def unidex_words(self, indices):
        return [self.index2word[i] for i in indices]

    def to_file(self, filename):
        values = [w for w, k in sorted(list(self.word2index.items())[5:])]
        with open(filename, 'w') as f:
            f.write('\n'.join(values))

    @classmethod
    def from_file(cls, filename):
        vocab = Vocab()
        with open(filename, 'r') as f:
            words = [l.strip() for l in f.readlines()]
            vocab.index_words(words)
```

Define tokenizer functions for you languages to split sentences on words.
In this examples I'm using standard [`nltk.tokenize.WordPunctTokenizer`](https://kite.com/python/docs/nltk.tokenize.WordPunctTokenizer) for German and
a function `tokenize_words` from package [`tokenize_uk`](https://github.com/lang-uk/tokenize-uk) for Ukrainian.
Also, replace input file with your

Since there are a lot of example sentences and we want to train something quickly, we'll trim the data set to only relatively short and simple sentences. Here the maximum length is 8 words (that includes punctuation) and we're filtering to sentences that translate to the form "I am" or "He is" etc.

Additionally, you might want to filter out rare words which occur only few times in corpus. This words could not be learned efficiently since there are not enough training examples for them. This will reduce vocabulary size and decrease training time.


```python
import nltk
from tokenize_uk import tokenize_words
import pandas as pd

max_length = 8
min_word_count = 2

tokenizers = {
    'ukr': tokenize_words,
    'deu': nltk.tokenize.WordPunctTokenizer().tokenize
}

def preprocess_corpus(sents, tokenizer, min_word_count):
    n_words = {}

    sents_tokenized = []
    for sent in sents:
        sent_tokenized = [w.lower() for w in tokenizer(sent)]

        sents_tokenized.append(sent_tokenized)

        for word in sent_tokenized:
            if word in n_words:
                n_words[word] += 1
            else:
                n_words[word] = 1

    for i, sent_tokenized in enumerate(sents_tokenized):
        sent_tokenized = [t if n_words[t] >= min_word_count else UNK_token for t in sent_tokenized]
        sents_tokenized[i] = sent_tokenized

    return sents_tokenized

def read_vocab(sents):
    vocab = Vocab()
    for sent in sents:
        vocab.index_words(sent)

    return vocab

source_sents = preprocess_corpus(corpus['text' + source_lang], tokenizers[source_lang], min_word_count)
target_sents = preprocess_corpus(corpus['text' + target_lang], tokenizers[target_lang], min_word_count)

source_sents, target_sents = zip(
    *[(s, t) for s, t in zip(source_sents, target_sents)
      if len(s) < max_length and len(t) < max_length]
)

source_vocab = read_vocab(source_sents)
target_vocab = read_vocab(target_sents)
target_vocab.to_file(os.path.join(data_dir, '{}.vocab.txt'.format(target_lang)))
source_vocab.to_file(os.path.join(data_dir, '{}.vocab.txt'.format(source_lang)))

print('Training corpus length: {}\nSource vocabulary size: {}\nTarget vocabulary size: {}'.format(
    len(source_sents), len(source_vocab.word2index), len(target_vocab.word2index)
))
```

    Training corpus length: 11582
    Source vocabulary size: 4131
    Target vocabulary size: 3201


Data for deep learning experiment is usually split into three parts:
* Training data is used for neural network training;
* Development data is used to select an optimal training stop point;
* Test data is used for final evaluation of experiment performance.

We will use 80% of the data as a training set, 6% of the data as a development set and 14% of the data as a test set.


```python
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

source_length = len(source_sents)
inidices = np.random.permutation(source_length)

training_indices = inidices[:int(x_length*0.8)]
dev_indices = inidices[int(x_length*0.8):int(x_length*0.86)]
test_indices = inidices[int(x_length*0.86):]

training_source = [source_sents[i] for i in training_indices]
dev_source = [source_sents[i] for i in dev_indices]
test_source = [source_sents[i] for i in test_indices]

training_target = [target_sents[i] for i in training_indices]
dev_target = [target_sents[i] for i in dev_indices]
test_target = [target_sents[i] for i in test_indices]
```

PyTorch uses its own format of data – Tensor. A Tensor is a multi-dimensional array of numbers with some type e.g. FloatTensor or LongTensor. Before we can use our training data, we need to convert it into tensors using previously defined word indices.
Additionally, we need to add special tokens SOS (start of a sentence) and EOS (end of a sentence) to each sentence.
Also, all sentences should have the same length to make possible batch training, therefore we will extend them with token PAD if needed.


```python
import torch

def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

def tensor_from_sentence(vocab, sentence, max_seq_length):
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(EOS_idx)
    indexes.insert(0, SOS_idx)
    # we need to have all sequences the same length to process them in batches
    if len(indexes) < max_seq_length:
        indexes += [PAD_idx] * (max_seq_length - len(indexes))
    tensor = torch.LongTensor(indexes)
    return tensor

def tensors_from_pair(source_sent, target_sent, max_seq_length):
    source_tensor = tensor_from_sentence(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    target_tensor = tensor_from_sentence(target_vocab, target_sent, max_seq_length).unsqueeze(1)
    return (source_tensor, target_tensor)

max_seq_length = max_length + 2  # 2 for EOS_token and SOS_token

training = []
for source_sent, target_sent in zip(training_source, training_target):
    training.append(tensors_from_pair(source_sent, target_sent, max_seq_length))

x_training, y_training = zip(*training)
x_training = torch.transpose(torch.cat(x_training, dim=-1), 1, 0)
y_training = torch.transpose(torch.cat(y_training, dim=-1), 1, 0)
torch.save(x_training, os.path.join(data_dir, 'x_training.bin'))
torch.save(y_training, os.path.join(data_dir, 'y_training.bin'))

x_development = []
for source_sent in dev_source:
    tensor = tensor_from_sentence(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    x_development.append(tensor)

x_development = torch.transpose(torch.cat(x_development, dim=-1), 1, 0)
torch.save(x_development, os.path.join(data_dir, 'x_development.bin'))

x_test = []
for source_sent in test_source:
    tensor = tensor_from_sentence(source_vocab, source_sent, max_seq_length).unsqueeze(1)
    x_test.append(tensor)

x_test = torch.transpose(torch.cat(x_test, dim=-1), 1, 0)
torch.save(x_test, os.path.join(data_dir, 'x_test.bin'))
```

## Encoder

The encoder of a Seq2seq network is a [**Recurrent Neural Network**](https://en.wikipedia.org/wiki/Recurrent_neural_network). A recurrent network can process model a sequence of related data (sentence in our case) using the same set of weights. To do this, RNN uses its output from a previous step as input along with input from the sequence.

A naive implementation of RNN is subject to problems with a gradient for long sequences; therefore, I use [**Long-Short Term Memory**](https://en.wikipedia.org/wiki/Long_short-term_memory) as a recurrent module. You should not care about its implementation since it already implemented in PyTorch: [nn.LSTM](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM). This module allows bi-directional sequence processing out-of-the-box – this allows to capture backward relations in a sentence as well as forward relations.

Additionally, I use embeddings module to convert word indices into dense vectors. This allows projecting discrete symbols (words) into continuous space which reflects semantical relations in spatial words positions. For this experiment, I will not use pre-trained word vectors and train this representation using machine translation supervision signal. But you may use the pre-trained word embeddings (for Ukrainian [lang-uk](http://lang.org.ua/en/models/#anchor4) project).

To not forget the meaning of dimensions for input vectors, I leave comments like `# word_inputs: (batch_size, seq_length)`. Above means that variable `word_inputs` contains a reference to tensor, which has shape `(batch_size, seq_length)`, e.g. it is an array of sequences of length `seq_length`, where array length is `batch_size`.


```python
import torch.nn as nn
import torch.nn.init as init

USE_CUDA = False

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(
            hidden_size,
            int(hidden_size/2),  # Bi-directional processing will ouput vectors of double size, therefore I reduced output dimensionality
            num_layers=n_layers,
            batch_first=True,  # First dimension of input tensor will be treated as a batch dimension
            bidirectional=True
        )

    # word_inputs: (batch_size, seq_length), h: (h_or_c, layer_n_direction, batch, seq_length)
    def forward(self, word_inputs, hidden):         
        # embedded (batch_size, seq_length, hidden_size)
        embedded = self.embedding(word_inputs)
        # output (batch_size, seq_length, hidden_size*directions)
        # hidden (h: (batch_size, num_layers*directions, hidden_size),
        #         c: (batch_size, num_layers*directions, hidden_size))
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batches):
        hidden = torch.zeros(2, self.n_layers*2, batches, int(self.hidden_size/2))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
```

## Decoder

Decoder module is similar to encoder with the difference in that it generates a sequence, therefore it will process inputs one by one; therefore it cannot be bidirectional.

```python
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)

    def forward(self, word_inputs, hidden):
        # Note: we run this one by one
        # embedded (batch_size, 1, hidden_size)
        embedded = self.embedding(word_inputs).unsqueeze_(1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden
```

## Test

To make sure the Encoder and Decoder models are working (and working together) we'll do a quick test with fake word inputs:


```python
vocab_size = 10
hidden_dim = 10
n_layers = 2

encoder_test = EncoderRNN(vocab_size, hidden_dim, n_layers)
print(encoder_test)

# Recurrent network requires initial hidden state
encoder_hidden = encoder_test.init_hidden(1)

# Test input of size (1x3), one sequence of size 3
word_input = torch.LongTensor([[1, 2, 3]])

if USE_CUDA:
    encoder_test.cuda()
    word_input = word_input.cuda()

encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

# encoder_outputs: (batch_size, seq_length, hidden_size)
# encoder_hidden[0, 1]: (n_layers*2, batch_size, hidden_size/2)
print(encoder_outputs.shape, encoder_hidden[0].shape, encoder_hidden[1].shape)
```

    EncoderRNN(
      (embedding): Embedding(10, 10)
      (lstm): LSTM(10, 5, num_layers=2, batch_first=True, bidirectional=True)
    )
    torch.Size([1, 3, 10]) torch.Size([4, 1, 5]) torch.Size([4, 1, 5])


`encoder_hidden` is a tuple for h and c components of LSTM hidden state. In PyTorch, tensors of LSTM hidden components have a following meaning of dimensions:
* First dimension is n_layers*directions, meaning that if we have a bidirectional network, then each layer will store two items in this direction;
* Second dimension is a batch dimension
* Third dimension is a hidden vector itself

The decoder uses single directional LSTM, therefore we need to reshape encoders h and c before sending them into decoder: concatenate all bi-directional vectors into single-direction vectors. This means, that every two vectors along `n_layers*directions` I combine into a single vector, increasing size of hidden vector dimension in two times and decreasing size of the first dimension to `n_layers`, which is two.


```python
decoder_test = DecoderRNN(vocab_size, hidden_dim, n_layers)
print(decoder_test)

word_inputs = torch.LongTensor([[1, 2, 3]])

decoder_hidden_h = encoder_hidden[0].reshape(2, 1, 10)
decoder_hidden_c = encoder_hidden[1].reshape(2, 1, 10)

if USE_CUDA:
    decoder_test.cuda()
    word_inputs = word_inputs.cuda()

for i in range(3):
    input = word_inputs[:, i]
    decoder_output, decoder_hidden = decoder_test(input, (decoder_hidden_h, decoder_hidden_c))
    decoder_hidden_h, decoder_hidden_c = decoder_hidden
    print(decoder_output.size(), decoder_hidden_h.size(), decoder_hidden_c.size())
```

    DecoderRNN(
      (embedding): Embedding(10, 10)
      (lstm): LSTM(10, 10, num_layers=2, batch_first=True)
    )
    torch.Size([1, 1, 10]) torch.Size([2, 1, 10]) torch.Size([2, 1, 10])
    torch.Size([1, 1, 10]) torch.Size([2, 1, 10]) torch.Size([2, 1, 10])
    torch.Size([1, 1, 10]) torch.Size([2, 1, 10]) torch.Size([2, 1, 10])


## Seq2seq

The logic to coordinate this two modules I have stored in a high-level module `Seq2seq`: it takes care of Encoder-Decoder coordination, a transformation of decoder results into word probability distribution.

Also, this module implements two `forward` functions: one for training time and second is for inference. The difference between these two functions is that during training I am using training `y` values (target sentence words) as decoder input; this is called [Teacher Forcing](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/). Obviously, during inference, I don't have `y` values.


```python
class Seq2seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, n_layers):
        super(Seq2seq, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(input_vocab_size, hidden_size, self.n_layers)
        self.decoder = DecoderRNN(output_vocab_size, hidden_size, self.n_layers)

        self.W = nn.Linear(hidden_size, output_vocab_size)
        self.softmax = nn.Softmax()

    def _forward_encoder(self, x):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(x, init_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden

        decoder_hidden_h = encoder_hidden_h.reshape(self.n_layers, batch_size, self.hidden_size)
        decoder_hidden_c = encoder_hidden_c.reshape(self.n_layers, batch_size, self.hidden_size)
        return decoder_hidden_h, decoder_hidden_c

    def forward_train(self, x, y):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        H = []
        for i in range(y.shape[1]):
            input = y[:, i]
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (batch_size, vocab_size)
            h = self.W(decoder_output.squeeze(1))
            # h: (batch_size, vocab_size, 1)
            H.append(h.unsqueeze(2))

        # H: (batch_size, vocab_size, seq_len)
        return torch.cat(H, dim=2)

    def forward(self, x):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        current_y = SOS_idx
        result = [current_y]
        counter = 0
        while current_y != EOS_idx and counter < 100:
            input = torch.tensor([current_y])
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (vocab_size)
            h = self.W(decoder_output.squeeze(1)).squeeze(0)
            y = self.softmax(h)
            _, current_y = torch.max(y, dim=0)
            current_y = current_y.item()
            result.append(current_y)
            counter += 1

        return result
```

# Training

To optimize neural network weights we need to have a model itself and optimizer. Model is already defined and the optimizer is usually available in the NN framework. I use [Adam](http://ruder.io/optimizing-gradient-descent/index.html#adam) from [torch.optim](https://pytorch.org/docs/stable/optim.html).


```python
from torch.optim import Adam

model = Seq2seq(len(source_vocab), len(target_vocab), 300, 1)
optim = Adam(model.parameters(), lr=0.001)
```

Since neural network training is a computationally expensive process, it is better to a train neural network for multiple examples at once. Therefore we need to split our training data on mini-batches.


```python
import math

def batch_generator(batch_indices, batch_size):
    batches = math.ceil(len(batch_indices)/batch_size)
    for i in range(batches):
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        if batch_end > len(batch_indices):
            yield batch_indices[batch_start:]
        else:
            yield batch_indices[batch_start:batch_end]
```

Previously I mentioned log-lieklyhood function which is used to optimize model parameters; in PyTorch this function is implemented in module `CrossEntropyLoss`:


```python
cross_entropy = nn.CrossEntropyLoss()
```

Finally, we can start to train our model. Each training epoch includes forward propagation, which yields some training results for training target sentences; then `cross_entropy` loss is calculated and `loss.backward()` calculates gradient with respect to the loss for each model parameter. After that, `optim.step()` uses the gradient to adjust model parameters and minimize loss.

After each training epoch, the development set is used to evaluate model performance with [BLEU](https://en.wikipedia.org/wiki/BLEU) score. I use `early_stop_counter` to stop the training process if BLEU is not improving for 10 epochs.

Module `tqdm` is optional to use, it is a handy and simple way to create a progress bar for a long operations.


```python
from tqdm import tqdm_notebook as tqdm
from nltk.translate.bleu_score import sentence_bleu

BATCH_SIZE = 100
total_batches = int(len(x_training)/BATCH_SIZE)
early_stop_after = 10
early_stop_counter = 0

best_bleu = 0.0

for epoch in range(10000):
    total_loss = 0
    for batch in tqdm(batch_generator(list(range(len(x_training))), BATCH_SIZE),
                      desc='Training epoch {}'.format(epoch+1),
                      total=total_batches):
        x = x_training[batch, :]
        # y for teacher forcing is all sequence without a last element
        y_tf = y_training[batch, :-1]
        # y for loss calculation is all sequence without a last element
        y_true = y_training[batch, 1:]
        # (batch_size, vocab_size, seq_length)
        H = model.forward_train(x, y_tf)
        loss = cross_entropy(H, y_true)
        assert loss.item() > 0

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
    print('Epoch {} training finished, loss: {}'.format(epoch+1, total_loss/total_batches))

    bleu = 0.0
    dev_length = len(dev_target)
    for x, reference in tqdm(zip(x_development, dev_target),
                             desc='Validating epoch {}'.format(epoch+1),
                             total=dev_length):
        y = model(x.unsqueeze(0))
        hypothesis = target_vocab.unidex_words(y[1:-1])  # Remove SOS and EOS
        bleu += sentence_bleu([reference], hypothesis)

    bleu /= dev_length
    source = ' '.join(source_vocab.unidex_words(x.tolist()[0]))
    print ('Epoch {} validation finished, BLEU: {}\nTranslation example: source "{}", ref "{}", hypothesis "{}"'.format(
        epoch+1, bleu, source, ' '.join(reference), ' '.join(hypothesis)
    ))

    if bleu > best_bleu:
        early_stop_counter = 0
        print('The best model found, resetting eraly stop counter.')
        best_bleu = bleu
    else:
        early_stop_counter += 1
        print('No improvement, early stop counter: {}.'.fromat(early_stop_counter))
        if early_stop_counter >= early_stop_after:
            print('Early stop!')
            break
```
