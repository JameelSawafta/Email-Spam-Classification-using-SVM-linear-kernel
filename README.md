# Email Spam Classification using SVM linear kernel

This project is an implementation of a **Support Vector Machine (SVM)** classifier to classify emails as spam or non-spam

## Overview
The aim of this project is to develop a machine learning model that can accurately classify emails as spam or non-spam using the SVM algorithm. The model is trained on a dataset of labeled emails.

The project consists of the following components:

* Data Loading and Exploration
* Data Preprocessing
* Feature Extraction
* Model Training and Evaluation
* Model Testing

## Requirements
To get started with this project, you will need to install Python and the following libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
* nltk

## Data preprocessing
In this section, we preprocess the data to prepare it for training the model. We perform tasks such as removing stop words.

## Data Splitting
We split the dataset into a training set and a testing set with a ratio of 80:20. The training set was used to train the SVM model, while the testing set was used to evaluate its performance.

## Training
We trained an SVM model with a linear kernel on the training set using scikit-learn's **svm.SVC** function. We used a C value of 1.0 and set the class_weight parameter to 'balanced' to account for any class imbalance in the dataset.

## Evaluation
We evaluated the performance of the SVM model on the testing set using scikit-learn's metrics.accuracy_score function. We obtained an accuracy score of **0.9845**, indicating that the model is able to correctly classify almost 98.5% of the emails as either spam or non-spam.

## Conclusion
Based on our results, we can conclude that the SVM algorithm with a linear kernel is a promising approach for email spam classification. With an accuracy score of **0.9845**, the model is able to effectively distinguish between spam and non-spam emails. Further improvements can be made by tuning the model's hyperparameters or using different preprocessing techniques. Nonetheless, this report demonstrates that machine learning can be a powerful tool for addressing the problem of email spam.


