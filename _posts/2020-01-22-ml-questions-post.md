---
layout: post
title: ML/Data Science Interview Questions
date: 2020-01-22
summary: A frequently updated list of commonly asked ML interview questions.
categories: posts
---

***Below is a list of common data science/machine learning questions you may be asked in an interview. A question about the self-attention mechanism may be rare, however, I believe it will become more common as it is such an important operation integral to many architectures***

### 1. What is the difference between supervised and unsupervised machine learning?

**Supervised Learning:**

Supervised learning involves training a machine learning algorithm on a dataset of labeled examples, with each example being a feature vector $$x_i$$ with a corresponding label $$y_i$$. The trained algorithm will take a feature vector $$x_i$$ as input and produce a label or labels that best match this input vector. 

**Unsupervised Learning:**

Unsupervised learning involes using a dataset of unlabled examples, where again $$x_i$$ is a feature vector that does not have a corresponding label and the goal of the unsupervised algorithm is to find some pattern in the data that was previously unkown. Ways this can be used are [1]:

- Clustering
  - For clustering, the model returns the ID of the cluster for each feature in the dataset
- Anomaly (outlier) Detection
  - In anomaly detection, the model's output is a value describing how different the example is from the usual examples
- Dimensionality Reduction
  - In dimensionality reduction, the output vector has fewer features than the input vector. 

### 2. What is the bias-variance trade-off?

**Bias:**

The **bias** in a model is how well the model fits the training data, i.e. the training accuracy.

A model has **low bias** if it has high accuracy in predicting the labels of the training data.

A model has **high bias** if it has low accuracy in predicting the labels of the training data.

**Variance:**

The **variance** in a model is its sensitivity to small fluctuations [1] in the training data.

A model has **low variance** if it has a higher accuracy in predicting the labels of the validation/test data compared to the training data.

A model has **high variance** if it has a lower accuracy in predicting the labels of the validation/test data compared to the training data.

**Bias-variance trade-off:**

The problem of finding a model that has a low bias (won’t underfit) and low variance (won’t overfit) so your model can generalise well to new unseen data. The difficulty is finding the point at which you have low bias and low variance, commonly known as the **bias-variance trade-off**.

### 3. Explain the differences between L1 and L2 regularisation

**L1 regularisation**:

Also known as lasso regression, is a way of regularising a model's weights by adding a penalising term to the loss function. This term contains a hyperparameter that is multiplied by the absolute value of the weights. When this hyperparameter is greater than zero it tends to lead to less important features being essentially turned off with their weights getting close to zero, leading to a sparse, less complex model. L1 regression is essentially a feature selector [1] that decides which features are important for correct prediction and which are not.

$$ min(\mathbf w,b)\ \left[\frac{1}n \sum_{i=1}^n (f(\mathbf X_i) - y_i)^2 \ + \ \lambda |\mathbf w_i| \right],  $$

$$ where \ |\mathbf w| = \sum_{j=1}^D |w^{(j)}| $$

**L2 regularisation**:

Also known as ridge regression, is a way of regularising a model's weights by adding a penalising term to the loss function. This term again contains a tuneable hyperparameter that minimises coefficients using least squares, because of this the coefficients are minimised but do not minimise to zero, unlike L1 regularisation. L2 regularisation is also differentiable so it can be used with gradient descent during training to optimise the loss function.

$$ min(\mathbf w,b)\ \left[\frac{1}n \sum_{i=1}^n (f(\mathbf X_i) - y_i)^2 \ + \ \lambda ||\mathbf w_i||^2 \right], $$

$$ where \ ||\mathbf w||^2 = \sum_{j=1}^D (w^{(j)})^2 $$

### 4. How Does the Self-attention mechanism work?

*The following comes from Peter Bloems blog post [2] on Transformers (the best I have found)*

**Self-attention** is a sequence-to-sequence operation: a sequence of vectors goes in and a sequence of vectors comes out. The input vectors can be $$x_1, \ x_2,\ ..., \ x_n $$  and the corresponding output vectors can be $$y_1, \ y_2, \ ..., \ y_n $$. The output vector $$y_i$$ is produced by applying the self-attention operation of a weighted average across all of the input vectors:

$$y_i = \sum_j w_{ij}x_j$$

$$j$$ indexes over the whole sequence and the weights sum to $$\frac{1}{j}$$

 The weight $$w_{ij}$$ is not a traditional weight parameter like in a neural network but derived from a function over $$x_i$$ and $$x_j$$. The simplest option for the function is the dot product:

$$w'_{ij} = x_i \cdot x_j$$

The dot product can give a value anywhere from negative and positive infinity, so a softmax is applied to scale the values between [0, 1], ensuring they sum to 1 over the whole sequence:

$$w_{ij} = \frac{\exp{w'_{ij}}}{\sum_j \exp{w'_{ij}}}$$

And that is the basic operation of self-attention

### 5. How can you deal with class imbalance?

There are a few main ways to account for class imbalance:

1. Collect more data for the underrepresented class
2. Resample your data so that your classes are around 50:50:

    - Oversample the minority class by making multiple copies of the same example of that class.
    - Downsample the majority class by randomly removing some examples of the majority class.

1. Some algorithms allow you to provide a weighting to a specific class for the algorithm to take into account when training
2. Creating synthetic examples from your minority class using the algorithms:

    - Synthetic Minority Oversampling Technique (SMOTE)
    - Adaptive Synthetic Sampling Method (ADASYN)

### 6. How would you detect whether a model is underfitting/overfitting?

**Underfitting:**

Your model is underfitting when you have a high bias, the model is predicting poorly the labels of the training data. To determine whether your model is underfitting you can compare your training accuracy against your validation accuracy, if both are showing low accuracy then your model is most likely underfitting to the data.

**Overfitting:**

Your model is overfitting when you have high variance, the model does a good job at predicting the labels of the training data but predicts poorly on your validation/test data. To determine whether your model is overfitting you can compare your training error against your validation error, if the training error appears to be much lower than the validation error, then your model is most likely overfitting to the training data.

<figure>
	<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_underfitting_overfitting_001.png" alt="3 graphs showing a polynomial regression with different polynomial degrees, resulting in underfitting, almost perfect fit and overfitting" title="Underfitting vs Overfitting" style="zoom:72%;" />
	<figcaption style="font-size:10px; text-align:center;">Image from sci-kit learn documentation </figcaption>
</figure>

The above image shows three graphs [3], the first is a linear function trying to approximate to part of a cosine function, it isn't fitting very well - we can say it is underfitting (high bias). The linear function (polynomial of degree 1) is not complex enough to fit to the training examples. 

The second graph shows a polynomial function, of degree 4, trying to approximate to the part of the cosine function, we can see it fits well - almost perfectly. 

The third graph shows another polynomial function, of degree 15, trying to approximate to the part of the cosine function, we can see it fits poorly, it is learning the noise in the data and overfitting (high variance), trying to fit to every single point. The polynomial of degree 15 is too complex for the training data it is being trained on.

### 7. How would you deal with a model that appears to be overfitting?

**Dealing with underfitting:**

If you know your model is underfitting then there are a few things you can do to change it:

- Find a more complex model, usually, the current model is too simple for the data.
- Find more informative features from your data, perhaps your current features aren't powerful enough to let the model learn any meaningful insight.

**Dealing with overfitting:**

If you have confirmed your model is overfitting then you can try the following to change it:

- Regularisation - your model is too complex for the data, it is learning patterns in the data and fitting to the noise. Examples of regularisation are:

  - L1 and L2 regularisation
  - Batch Normalisation (standardises the output activations of a layer, therefore has a regularising effect)
  - Dropout
  - Early stopping
- Get more training data, your network may be the right one but has so little data it learns the noise in the data.
  - Data augmentation is a cheap way of getting more training data by adapting the features of the original example so that you end up with a new example that is slightly different from the original but different enough that the model can learn from it. 
- Reduce the complexity of your model, i.e. for a deep neural network use fewer layers/hidden units.
- Reduce the dimensionality of your data by removing features that are not important for prediction - Principal Component Analysis (PCA) , UMAP, t-SNE or Random Forests and Gradient Boosting Machines (built-in feature importance).

### 8. What is exploding and vanishing gradients?

**Exploding gradients:**

The problem of exploding gradients can arise when training deep neural networks, the parameters of these networks are updated during gradient descent using backpropagation. This problem arises because of the nature of backpropagation, the calculated gradients are backpropagated through the network to the first layer,  this means the gradients go through many matrix multiplications, if the gradients are large (>1) then these multiplications cause the gradients to get exponentially bigger with the number of layers until they explode, resulting in NaN values. 

**Vanishing Gradients:**

The problem of vanishing gradients, again, can occur when training deep neural networks. Vanishing gradients can occur when the calculated gradients are very small (<1) and these small gradients are, again, getting multiplied through the layers which cause the gradient to diminish exponentially with the number of layers, causing early layers to train very slowly because the gradient that reaches them is very small. 

### 9. How can you deal with exploding and vanishing gradients?

**Dealing with exploding gradients:**

You can apply simple techniques such as:

- Gradient clipping - a threshold on the gradients
- L1 or L2 regularisation

**Dealing with vanishing gradients:**

Vanishing gradients used to be an intractable problem for deep neural networks but these days most network and activation function implementations are effective at training very deep neural networks. 

Such improvements include:

- ReLU activation function
- LSTMs, GRUs
- Skip connections in residual networks
- Transformers
- New optimisers such as Adam, RAdam and AdaMod



### 10. What is cross-validation and when would you use it?

Cross-validation is a resampling technique that can be seen as a stand-in for a validation set when you have too few data to split into a training, validation and test set. Cross-validation can be used to determine which hyperparameters work best for your network. With cross-validation you will have a training set and a test set, leaving most of the data for the training set. 

Cross-validation involves the use of $$k$$ folds - subsets of your training data, each of the same size. Five-fold cross-validation is the most common value for $$ k $$, with five-fold cross-validation, the training data is split randomly into five folds $$F_1, F_2, ..., F_5$$. From here five new models are trained, the first model $$ f_1 $$ is trained on $$k-1$$ folds $$ F_2, F_3, F_4 $$ and $$ F_5 $$ and fold $$F_1$$ is used as the validation set for the model. Then model two $$ f_2 $$ is trained on folds $$ F_1, F_3, F_4 $$ and $$ F_5 $$, with fold $$F_2$$ being used as the validation set. This continues for the remaining models and subsets until you have trained and validated all five on your metric or metrics of choice. Then you can average over the k validation results and compute the final value.  



References: 

[1] ([The Hundred Page Machine Learning Book](http://themlbook.com)) 

[2] ([Transformers From Scratch - Peter Bloem](http://www.peterbloem.nl/blog/transformers)) 

[3] ([Underfitting vs. Overfitting - scikit-learn documentation](https://bit.ly/3aMY73s))

[4] ([Least squares - Wikipedia](https://en.wikipedia.org/wiki/Least_squares#Regularization))
