# Credit_Risk_Analysis
Apply machine learning to predict credit card risk using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

## Overview of Credit_Risk_Analysis:
Fast Lending a peer to peer lending services company wants to use machine learning to predict credit risk. Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, oversampling of the data is done using the RandomOverSampler and SMOTE algorithms, and undersampling of the data is done using the ClusterCentroids algorithm. A comparision of the two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, is done to predict credit risk. Our job is to  Assist the lead Data Scientist Jill using the imbalanced-learn and scikit-learn libraries to evaluate the performance of these models and make a written recommendation on whether these models should be used to predict credit risk.

## Resources:
* Data: LoanStats_2019Q1.csv
* Software: Jupyter Notebook 6.4.8 , Python 3.7.11 

## Results:
The code for the machine learning algortithms can be found in the notebooks [credit_risk_resampling.ipynb](https://github.com/sucharita1/Credit_Risk_Analysis/blob/41fa81b2ef10b35d67966a44622cc3a0eecee954/credit_risk_resampling.ipynb) and [credit_risk_ensemble.ipynb](https://github.com/sucharita1/Credit_Risk_Analysis/blob/41fa81b2ef10b35d67966a44622cc3a0eecee954/credit_risk_ensemble.ipynb).

* First we will import dependencies, ignore warning, perform basic data cleaning.
![warning](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/warning.png?raw=true)
![read](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/read.png?raw=true)
![load](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/load.png?raw=true)
![df](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/df.png?raw=true)

* Split the data into training and testing. This included encoding the strings into numerical data, and identifying the features, X and target, y.

![split](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/split.png?raw=true)
![test](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/test.png?raw=true)


#### Use Resampling Models to Predict Credit Risk
We can see outright that our data is imbalanced that is the loan status is as follows:
* low_risk     68470
* high_risk      347

So when we split the data the y_train counter is as follows:
* Counter({'low_risk': 51366, 'high_risk': 246})

We perform oversampling, SMOTE, undersampling and SMOTEENN methods to balance the data before logistic regression and find out which one gives the best results.
* Oversampling 
Counter({'low_risk': 51366, 'high_risk': 51366})

![oversampling](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/oversampling.png?raw=true)

* SMOTE oversampling
Counter({'low_risk': 51366, 'high_risk': 51366})

![SMOTE](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/SMOTE.png?raw=true)

* Undersampling
Counter({'high_risk': 246, 'low_risk': 246})

![undersampling](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/undersampling.png?raw=true)


#### Use the SMOTEENN Algorithm to Predict Credit Risk
Counter({'high_risk': 68458, 'low_risk': 62022})

![combination](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/combination.png?raw=true)

We can see that Logistic Regression model after performing all kinds of resampling also does not give us high accuracy, the maximum accuracy is 68 % for SMOTEENN and the f1 score for high risk is a dismal 2% and  it is 73% only for low risk which is also not too good. So, we need to explore other alogrithms like ensemble classifiers to see if we can get better results.

#### Use Ensemble Classifiers to Predict Credit Risk
* Balanced Random Forest classifier

![balanced_random_forest](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/balanced_random_forest.png?raw=true)

* Easy Ensemble Ada boost Classifier

![easy_ensemble](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/easy_ensemble.png?raw=true)

We can see that Ensemble Classifers perform far better, the Balanced Random Forest classifier has an accuracy of 79% and a f1 score of 6% for high risk and 93% for low risk. While Easy Ensemble Ada boost Classifier has an accuracy of 93% and F1 score of 16% for high risk and 97% for low risk.

## Summary:

* Summary of all six machine learning Models:
![summary](https://github.com/sucharita1/Credit_Risk_Analysis/blob/cf647cc5d22c232068e0526aa18635bc6402fc9e/Images/summary.png?raw=true)

Here, we can clearly see that the Ensemble Algorithms have provided much better results than logistic regression coupled with resampling. Easy Ensemble Ada boost Classifier has the best performance overall.

* Our dataset is an unbalanced one. So, the precison for high risk loans is very low below 10% for all the six models. Typically, for unbalanced data instead of concentrating on recall or precision, F1 score and accuracy can give an overall balanced picture. Easy Ensemble Ada boost Classifier has the best numbers in the whole bunch:

| Property | Value |
| --- | --- |
Accuracy       |         0.93 |
High-risk Precision	 |    .09	
Low-risk Precision  |       1				
High-risk Recall	   |    0.92
Low-risk Recall	    |    0.94
High-risk F1 Score |	    0.16
Low-risk F1 Score  |     0.97

Easy Ensemble Ada boost Classifier with an accuracy of 93% alongwith an F1 score for low-risk at 97% and high-risk at 16% and is the one that is recommended.

 


