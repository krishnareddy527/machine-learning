# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 10:51:19 2018

@author: Krishna
"""

# Import required libraries: (some of them are not available in Anaconda!)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from IPython.display import Image
import pydotplus
import pandas

# Load Data
data = pandas.read_csv('titanic_data.csv', sep=',')

# Define outcome, drop non used features and generate a binary variable for Sex:
outcomes = data['Survived']
data = data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)
data.loc[:, 'Sex'] = data['Sex'].apply(lambda x: 1. if x == 'female' else 0.)