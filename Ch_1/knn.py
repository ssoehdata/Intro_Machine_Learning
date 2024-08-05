# page 21 Intro to ML w Python
import sklearn
from sklearn.datasets import load_iris
iris_dataset = load_iris()
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
        metric_params=None, n_jobs=None, n_neighbors=1, p=2, 
        weights='uniform')

X_new = np.array([5, 2.9, 1, 0.2])
print("X_new.shape:", X_new.shape)

X_new.shape:(1,4)

prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
    iris_dataset['target_names'][prediction])

Prediction:  [0]
Predicted target name: ['setosa']

