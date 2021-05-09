# TASK 02 : Prediction using Unsupervised ML
# By JERIN PHILIP

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("FIRST FIVE DATASET AS SAMPLE:")
print(iris_df.head())

# PART 01 : Finding the optimum number of clusters for k-means classification
# We will be using elbow method to determine the same.
x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph, allowing us to observe 'The elbow'

font1 = {'family':'serif','color':'red','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.plot(range(1, 10), wcss)
plt.title('The Elbow method', fontdict=font1)
plt.xlabel('Number of clusters', fontdict=font2)  # No. of Clusters
plt.ylabel('WCSS', fontdict=font2)                # Within cluster sum of squares
plt.show()

# PART 02 : Representation of the clusters

# Applying kmeans to the dataset / Creating the kmeans classifier

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_pred = kmeans.fit_predict(x)
print()
print("Centroids of the Clusters:")
print(kmeans.cluster_centers_)
print()
print("Predicted Points afters applying k-means :")
print(y_pred)
# Cluster 01
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s=50, c='red', label='Iris-setosa')
# Cluster 02
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s=50, c='blue', label='Iris-versicolour')
# Cluster 03
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s=50, c='green', label='Iris-virginica')

# Plotting the centroids of the cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='yellow', label='Centroids')

plt.show()
