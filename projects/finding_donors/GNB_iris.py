# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:05:44 2018

@author: Krishna
"""

from time import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def split_features_labels(data_set):
	features = data_set.data
	labels = data_set.target
	return features, labels

def split_train_test(features, labels, test_size):
	total_test_size = int(len(features) * test_size)
	np.random.seed(2)
	indices = np.random.permutation(len(features))
	train_features = features[indices[:-total_test_size]]
	train_labels = labels[indices[:-total_test_size]]
	test_features  = features[indices[-total_test_size:]]
	test_labels  = labels[indices[-total_test_size:]]
	return train_features, train_labels, test_features, test_labels

data_set=load_iris()
features , labels = split_features_labels(data_set)
train_features, train_labels, test_features, test_labels = split_train_test(features, labels, 0.15)
clf = GaussianNB()
tStart = time()
clf.fit(train_features, train_labels)
print("Training time: ", round(time()-tStart, 3), "s")
print(accuracy_score(clf.predict(test_features), test_labels))