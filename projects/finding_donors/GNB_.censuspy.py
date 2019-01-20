# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:32:33 2018

@author: Krishna
# 
# https://www.kaggle.com/uciml/adult-census-income
# there are 32,562 total entries
# features are: "age","workclass","fnlwgt","education","education.num",
# "marital.status","occupation", "relationship","race","sex","capital.gain",
# "capital.loss","hours.per.week","native.country","income"
# label is "income"
#
# used algorithm: Gaussian Naive Bayes (GaussianNB)
# 
# accuracy ~80%
#
"""
import os
import pandas as pd
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data_set():
    csv_path = os.path.join("adult.csv")
    return pd.read_csv(csv_path)

def clean_data_set(data_set):
	for column in data_set.columns:
		if data_set[column].dtype == type(object):
			le = LabelEncoder()
			data_set[column] = le.fit_transform(data_set[column])
	return data_set

def split_features_labels(data_set, feature):
	features = data_set.drop(feature, axis=1)
	labels = data_set[feature].copy()
	return features, labels

data_set = load_data_set()

data_set = clean_data_set(data_set)
train_set, test_set = train_test_split(data_set, test_size=0.15, random_state=3)
train_features, train_labels = split_features_labels(train_set, "income")
test_features, test_labels = split_features_labels(test_set, "income")
clf = GaussianNB()
print("Start training...")
tStart = time()
clf.fit(train_features, train_labels)
print("Training time: ", round(time()-tStart, 3), "s")
a= accuracy_score(clf.predict(test_features), test_labels)
print("Accuracy: ",a)