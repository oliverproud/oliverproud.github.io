---
layout: post
title: ML/Data Science Interview Questions
date: 2020-01-22-11:13
summary: A frequently updated list of commonly asked ML interview questions.
categories: draft
---

### 1. What is the difference between supervised and unsupervised machine learning?

In **supervised learning** the dataset is a collection of labelled examples, each example x is a feature vector. The goal of a supervised machine learning algorithm is to use this labelled dataset to train a model that takes a feature vector x as input and output a label that best matches this feature vector. 

In **unsupervised learning**, the dataset is a collection of unlabelled examples, where again x is a feature vector. The goal of an unsupervised machine learning algorithm is to use the unlabelled dataset to find some previously unknown pattern, ways this can be used are: clustering, anomaly (outlier) detection and dimensionality reduction. For clustering, the model returns the ID of the cluster for each feature in the dataset, for anomaly detection the model's output is a value describing how different the example is from the usual examples, and finally, in dimensionality reduction, the output vector has fewer features than the input vector.

### 2. What is the bias-variance trade-off?

**High bias** in machine learning is the error when a model performs weakly in fitting to the dataset, in other words, high bias is when a model underfits the data because the model is too simple for the given data or the features aren’t powerful enough to extract any meaningful insight from.

**High variance** is the error when a model fits so well to the given dataset that it fits the noise in that dataset, meaning it will not generalise well to new unseen data. High variance is also known as overfitting the model to the data because either the model is too complex, your data contains too few examples or usually both. 

**Bias-variance trade-off** is the problem of finding a model that has a low bias (won’t underfit) and low variance (won’t overfit) so your model can generalise well to new unseen data. The difficulty is finding the point at which you have low bias and low variance. 

### 3. Explain the differences between L1 and L2 regularisation 

**L1 regularisation**, also known as Lasso Regularisation, is a way of regularising a model's weights by adding a penalising term to the loss function. This term contains a hyperparameter that is multiplied by the absolute value of the weights. When this hyperparameter is greater than zero it tends to lead to less important features being essentially turned off with their weights getting close to zero, leading to a sparse, less complex model. 

**L2 regularisation** also known as Ridge regularisation is a way of regularising a model's weights by adding a penalising term to the loss function. This term again contains a tuneable hyperparameter that is multiplied by the square of the weights. L2 regularisation is also differentiable so it can be used with gradient descent during training to optimise the loss function. 

### 4. How Does the Self-attention mechanism work?

_The following comes from Peter Bloems blog post on Transformers (the best I have found)_

**Self-attention** is a sequence-to-sequence operation: a sequence of vectors go in and a sequence of vectors come out. The input vectors can be X1 to Xn and the corresponding output vectors can be Y1 to Yn. 
The output vector Yn is produced by applying the self-attention operation of a weighted average across all of the input vectors. There is a weight W that is not a traditional weight parameter like in a neural network but is rather derived from a function over Xi and Xj. The simplest option for the function is the dot product. The dot product can give a value anywhere from negative to positive infinity, so a softmax is applied to scale the values between [0, 1], ensuring they sum up to 1 over the whole sequence. 

### 5. How can you deal with class imbalance?

There are a few main ways to account for class imbalance:
1. Collect more data for the underrepresented class 
2. Resample your data so that your classes are around 50:50:
    * Oversample the minority class by making multiple copies of the same example of that class.
    * Downsample the majority class by randomly removing some examples of the majority class.
3. Some algorithms allow you to provide a weighting to a specific class for the algorithm to take into account when training
4. Creating synthetic examples from your minority class using the algorithms:
    * Synthetic Minority Oversampling Technique (SMOTE)
    * Adaptive Synthetic Sampling Method (ADASYN)
