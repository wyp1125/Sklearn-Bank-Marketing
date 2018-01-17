# Sklearn-ML-Bank-Marketing
Build multiple machine learning models for bank marketing data using sklearn and pandas.

Dataset: https://archive.ics.uci.edu/ml/datasets/bank+marketing.

  The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

  The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

Source code: multi-classifier.py

  Written in Python with scikit-learn and pandas libraries.

  Data are imported as pandas dataframe, filtered by intact records, encoded on categorical variables using onehot scheme, normalized using min-max scaler, and divided into training and testing datasets. 
  
  For linear SVM and Random Forest models, importance of variables are sorted and top 10 are visualized using bar plots.

  Accuracy of prediction is printed for all the machine learning models including Nearest Neighbors, Linear SVM, RBF SVM, Decision Tree, Random Forest, Neural Net, AdaBoost and Naive Bayes.
