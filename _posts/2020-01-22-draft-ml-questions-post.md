---
layout: post
title: ML/Data Science Interview Questions
date: 2020-01-22-22:25
summary: A frequently updated list of commonly asked ML interview questions.
categories: draft
---

### 1. What is the difference between supervised and unsupervised machine learning?

**Supervised Learning:**

 The dataset is a collection of labelled examples, each example **x** is a feature vector. The goal of a supervised machine learning algorithm is to use this labelled dataset to train a model that takes a feature vector x as input and output a label that best matches this feature vector.

**Unsupervised Learning:** 

The dataset is a collection of unlabelled examples, where again **x** is a feature vector. The goal of an unsupervised machine learning algorithm is to use the unlabelled dataset to find some previously unknown pattern, ways this can be used are: clustering, anomaly (outlier) detection and dimensionality reduction. For clustering, the model returns the ID of the cluster for each feature in the dataset, for anomaly detection the model's output is a value describing how different the example is from the usual examples, and finally, in dimensionality reduction, the output vector has fewer features than the input vector.

### 2. What is the bias-variance trade-off?

**Bias:**

The **bias** in a model is how well the model fits the training data, i.e. the training accuracy. 

A model has **low bias** if it predicts the training data labels well, with low error. 

A model has **high bias** if it predicts the training data labels poorly, with a high error. 

**Variance:**

The **variance** in a model is its sensitivity to small fluctuations in the training data. 

A model has **low variance** if it predicts a little better on the validation data than the training data.

A model has **high variance** if it predicts poorly on the validation/test data but very well on the training data. 

**Bias-variance trade-off:**

The problem of finding a model that has a low bias (won’t underfit) and low variance (won’t overfit) so your model can generalise well to new unseen data. The difficulty is finding the point at which you have low bias and low variance, commonly known as the **bias-variance trade-off**.

### 3. Explain the differences between L1 and L2 regularisation 

**L1 regularisation**:

Also known as lasso regularisation, is a way of regularising a model's weights by adding a penalising term to the loss function. This term contains a hyperparameter that is multiplied by the absolute value of the weights. When this hyperparameter is greater than zero it tends to lead to less important features being essentially turned off with their weights getting close to zero, leading to a sparse, less complex model. L1 regression is essentially a feature selector that decides which features are important for correct prediction and which are not.

**L2 regularisation**:

Also known as ridge regularisation, is a way of regularising a model's weights by adding a penalising term to the loss function. This term again contains a tuneable hyperparameter that is multiplied by the square of the weights. L2 regularisation is also differentiable so it can be used with gradient descent during training to optimise the loss function. 

### 4. How Does the Self-attention mechanism work?

*The following comes from Peter Bloems* [*blog post*](http://www.peterbloem.nl/blog/transformers) *on Transformers (the best I have found)*

**Self-attention** is a sequence-to-sequence operation: a sequence of vectors go in and a sequence of vectors come out. The input vectors can be X1 to Xn and the corresponding output vectors can be Y1 to Yn.  The output vector Yn is produced by applying the self-attention operation of a weighted average across all of the input vectors. There is a weight W that is not a traditional weight parameter like in a neural network but is rather derived from a function over Xi and Xj. The simplest option for the function is the dot product. The dot product can give a value anywhere from negative to positive infinity, so a softmax is applied to scale the values between [0, 1], ensuring they sum up to 1 over the whole sequence. 

### 5. How can you deal with class imbalance?

There are a few main ways to account for class imbalance:

1. Collect more data for the underrepresented class 
2. Resample your data so that your classes are around 50:50:

- - Oversample the minority class by making multiple copies of the same example of that class.
  - Downsample the majority class by randomly removing some examples of the majority class.

1. Some algorithms allow you to provide a weighting to a specific class for the algorithm to take into account when training
2. Creating synthetic examples from your minority class using the algorithms:

- - Synthetic Minority Oversampling Technique (SMOTE)
  - Adaptive Synthetic Sampling Method (ADASYN)

### 6. How would you detect whether a model is underfitting/overfitting?

**Underfitting:** 

Your model is underfitting when you have a high bias, the model is predicting poorly the labels of the training data, to determine whether your model is underfitting you can compare your training accuracy against your validation accuracy, if both are showing high error then your model is most likely underfitting to the data. 

**Overfitting:**

Your model is overfitting when you have high variance, the model predicts well the labels of the training data but predicts poorly on your validation/test data, to determine whether your model is overfitting you can compare your training error against your validation error, if the training error appears to be much lower than the validation error, then your model is most likely overfitting to the training data.

### 7. How would you deal with a model that appears to be overfitting? 

**Dealing with underfitting:**

If you know your model is underfitting then there are a few things you can do to change it: 

- Find a more complex model, usually, the current model is too simple for the data.
- Find more informative features from your data, perhaps your current features aren't powerful enough to let the model extract meaningful insight. 

**Dealing with overfitting:** 

If you have confirmed your model is overfitting then you can try the following to change it:

- Regularisation - your model is too complex for the data, it is learning patterns in the data and fitting to the noise. Examples of regularisation are: 

- - L1 and L2 regularisation
  - Batch Normalisation (this is not actually a form of regularisation, but routinely has the effect of it)
  - Dropout
  - Early stopping (don't do this one, it's just an example).

- Get more training data, your model may be the right one but has so little data it learns its features so well. 

- Reduce the complexity of your model, i.e. for a deep neural network use fewer layers or hidden units. 

- Reduce the dimensionality of your data with Principal Component Analysis (PCA) or UMAP.