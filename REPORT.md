# Credit Risk Classification Report

## Contents
1. [Overview](#1-overview)
2. [Results](#2-results)
3. [Summary](#3-summary)

## 1. Overview

This analysis aims to evaluate a supervised machine learning model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company is used to build a model that can identify the creditworthiness of borrowers.

The model is used to learn and predict the loan_status of loan activities, with `0` representing Healthy loans and `1` representing High-risk loans. The model is based on the following information:
- loan_size
- interest_rate
- borrower_income
- debt_to_income
- num_of_accounts
- derogatory_marks
- total_debt


In the dataset, there are 75036 counts of Healthy loans and 2500 counts of High-risk loans. The dataset is highly imbalanced because in normal situations, there should be significantly more Healthy loans than the High-risk ones. However, in machine learning, sufficient training data can help improve the accuracy of the model.

The dataset was first split into training dataset and test dataset.

* y_train: loan_status
* X_train: features in the dataset, including loan_size, interest_rate, etc.

* y_test: loan_status
* X_test: features in the dataset, including loan_size, interest_rate, etc.

A Logistic Regression model was fitted to the training dataset and then used to predict the loan_status of the test dataset.

Since the sample was unbalanced (75036 Healthy loans vs. 2500 High-risk loans), another Logistic Regression Model was trained with Resampled Training Data. The original training data was resampled using the imblearn.over_sampling.RandomOverSampler() function. The resampled data (56271 Healthy loans vs. 56271 High-risk loans) was then used to train a new Logistic Regression Model.

The results of the two models are compared to decide which one is more suitable for predicting loan_status.


## 2. Results

* Machine Learning Model 1 (Logistic Regression - Original Data):
  * Balanced Accuracy Score: 0.952
  *	Precision: Healthy Loans `0` - 1.00; High-Risk Loans `1` - 0.85
  *	Recall: Healthy Loans `0` - 0.99; High-Risk Loans `1` - 0.91
  
- The logistic regression model is highly accurate overall. The balanced accuracy score is 0.952. The balanced accuracy is defined as the average of recall obtained on each class. As the best value is 1, it shows that the model is highly accurate.

- According to the classification report, the classification model achieves high performance. It shows excellent precision and recall for both healthy and high-risk loans. The model can distinguish between the Healthy loans and High-risk loans accurately, with minimal misclassification.

- The values of precision and recall are slightly lower when it comes to High-risk loans when compared to predicting Healthy loans.


* Machine Learning Model 2 (Logistic Regression - Resampled Data):
  * Balanced Accuracy Score: 0.994
  *	Precision: Healthy Loans `0` - 1.00; High-Risk Loans `1` - 0.84
  *	Recall: Healthy Loans `0` - 0.99; High-Risk Loans `1` - 0.99

- The model performs slightly better when using the resampled data.

- For Healthy loans `0`, though there is less training data, the precision and recall do not suffer a lot. This is because the amount of training data, though reduced, is still sufficient to train an accurate model.

- For High-risk loans `1`, with more training data, we can see improvements in the prediction results.

- Overall balanced accuracy score of the prediction is improved from 0.952 to 0.994.


## 3. Summary

Although accuracy of both model is high, Model 2 (Logistic Regression - Resampled Data) is recommended. The overall accuracy may only be slightly higher on surface. However, Model 2 is significantly better at predicting High-risk loans, which makes it a more suitable model for creditors. In real-world application, risk classification model is developed mainly to predict High-risk loans instead of healthy ones. With better prediction of High-risk loans, creditors can reduce potential risks in future lending activities.
