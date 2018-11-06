---
title: Linear Discriminant Analysis
parent: bayesian-statistics
context: data-science
---

<islr|p139>

Stated to be bayesian, but not exactly "Bayesian approach" in terms of <map> instead of <mle>.
Estimate posterior distribution of $P(Y=y|X=x)$ given prior $P(Y=y)$ and conditional $P(X=x|Y=y).

Seems to be more stable for <classification> then <logistic-regression>.

> Q: What is the difference with Naive Bayes?
> PA: Naive Bayes is a particular case of LDA when $f_k(x)$ is uses only single
> dimensional probabilities (?).

<qda> - non linear decision boundary because covariance matrix is separate for
each class. LDA uses a single covariance matrix for all classes, and this makes
decision boundary linear. This is pretty fantastic. %%toinvestigate
