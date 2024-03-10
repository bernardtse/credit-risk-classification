# Module 12 Report

## Overview of the Analysis

This analysis aims to evaluate a supervised machine learning model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company is used to build a model that can identify the creditworthiness of borrowers.

The model is used to learn and predict the loan_status of loan activities, with 0 representing Healthy loans and 1 representing High-risk loans. The model is based on the following information:
- loan_size
- interest_rate
- borrower_income
- debt_to_income
- num_of_accounts
- derogatory_marks
- total_debt


In the dataset, there are 75036 counts of Healthy loans and 2500 counts of High-risk loans. The dataset is highly imbalanced because in normal situations, there should be significantly more Healthy loans than the High-risk ones. However, in machine learning, sufficient training data can help improve the accuracy of the model.

The dataset was first split into training dataset and test dataset.

- y_train: loan_status
- X_train: features in the dataset, including loan_size, interest_rate, etc.

- y_test: loan_status
- X_test: features in the dataset, including loan_size, interest_rate, etc.

A Logistic Regression model was fitted to the training dataset and then used to predict the loan_status of the test dataset.

Since the sample was unbalanced (75036 Healthy loans vs. 2500 High-risk loans), another Logistic Regression Model was trained with Resampled Training Data. The original training data was resampled using the imblearn.over_sampling.RandomOverSampler() function. The resampled data (56271 Healthy loans vs. 56271 High-risk loans) was then used to train a new Logistic Regression Model.

The results of the two models are compared to decide which one is more suitable for predicting loan_status.


## Results

* Machine Learning Model 1 (Logistic Regression - original data):
  * Balanced Accuracy Score: 0.952
  *	Precision: Healthy Loan (0) - 1.00, High-Risk Loan (1) - 0.85
  *	Recall: Healthy Loan (0) - 0.99, High-Risk Loan (1) - 0.91
  
- The logistic regression model is highly accurate overall. The balanced accuracy score is 0.95. The balanced accuracy is defined as the average of recall obtained on each class. As the best value is 1, it shows that the model is highly accurate.

- According to the classification report, the classification model achieves high performance, with an balanced accuracy score of 0.952. It shows excellent precision, recall, and F1-score for both healthy and high-risk loans. The model can distinguish between the Healthy loans and High-risk loans accurately, with minimal misclassification.

- The values of precision, recall and F1-score are slightly lower when it comes to High-risk loans when compared to predicting Healthy loans.


* Machine Learning Model 2 (Logistic Regression - resampled data):
  * Balanced Accuracy Score: 0.993  
  *	Precision: Healthy Loan (0) - 1.00, High-Risk Loan (1) - 0.84
  *	Recall: Healthy Loan (0) - 0.99, High-Risk Loan (1) - 0.99

- The model performs slightly better when using the resampled data.

- For Healthy loan, though there is less training data, the precision, recall and f1-score do not suffer a lot. This is because the amount of training data, though reduced, is still sufficient to train an accurate model.

- On the other hand, with more training data for High-risk loan, we can see improvements in the prediction results.

- Overall balanced accuracy score of the prediction is improved from 0.95 to 0.99.

## Summary

Although accuracy of both model is high, Model 2 (Logistic Regression - resampled data) is recommended. The overall accuracy may only be slightly higher on surface. However, Model 2 is significantly better at predicting High-risk loans, which makes it a more suitable model for creditors. In real-world application, risk classification model is developed mainly to predict High-risk loans instead of healthy ones. With better prediction of High-risk loans, creditors can reduce potential risks in future lending activities.
