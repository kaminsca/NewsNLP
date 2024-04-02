import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# generating samples with 4 classes
# X: ndarray of shape (n_samples, n_features) The generated samples
# y: ndarray of shape (n_samples,) the integer labels for cluster membership of each sample.
X, y = make_blobs(n_samples = 500, n_features = 2, centers = 4,cluster_std = 1.5)

# show our data
plt.figure(figsize = (10,10))
plt.scatter(X[:,0], X[:,1], c=y,s=100,edgecolors='black')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)

# finding the right k value can be really important
knn5 = KNeighborsClassifier(n_neighbors = 5)
knn1 = KNeighborsClassifier(n_neighbors=1)

knn5.fit(X_train, y_train)
knn1.fit(X_train, y_train)

y_pred_5 = knn5.predict(X_test)
y_pred_1 = knn1.predict(X_test)

print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
print("Accuracy with k=1", accuracy_score(y_test, y_pred_1)*100)