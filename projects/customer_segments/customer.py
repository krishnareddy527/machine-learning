# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:56:54 2018

@author: Krishna
"""

print(__doc__)
import numpy as np
import visuals as vs
import matplotlib.pyplot as plt
from IPython.display import display

"""reading the data"""
import pandas as pd
customers_data = pd.read_csv("customers.csv")
data=customers_data.drop(['Channel', 'Region'],axis=1)
#print("data has {} rows of data with {} columns".format(*data.shape))

"""description of the data"""
description = data.describe()
#print(description)

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [50 , 200 , 401]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
#print("Chosen samples of wholesale customers dataset:")
#display(samples)

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
Feature="Grocery"
new_data = data.drop(Feature , axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, data[Feature] , test_size=0.25 , random_state = 42)
# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(X_train , y_train)
# TODO: Report the score of the prediction using the testing set
y_pred=regressor.predict(X_test)

from sklearn.metrics import accuracy_score , r2_score
score=accuracy_score(y_pred , y_test)

score_r2= r2_score(y_pred , y_test)
#print(Feature,"accuracy_score=",score ,"R2 score",score_r2 )

# Produce a scatter matrix for each pair of features in the data
#pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
#print(data.corr())
#feature_scaling
log_data = np.log(data)
log_samples = np.log(samples)
#pd.plotting.scatter_matrix(log_data,alpha=0.3 , figsize=(14,8) , diagonal = 'kde')
##############################################################################
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1) * 1.5
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [65 , 66 , 128 , 128, 154 , 94 ,167 ,159]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

from sklearn.decomposition import PCA
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA()
pca.fit(good_data)
# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))
#############################################################################
# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)
# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))
# Create a biplot
vs.biplot(good_data, reduced_data, pca)
#############################################################################
from sklearn.cluster import KMeans
# TODO: Apply your clustering algorithm of choice to the reduced data 
clusterer = KMeans(n_clusters=2)
clusterer.fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.cluster_centers_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

from sklearn.metrics import silhouette_score
# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data , preds)
print(score)
# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)"""