import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("iris.csv", header=None)

print(iris.head())

# data = iris[iris.columns[2:6]]

data = iris.iloc[:,2:6]

print(data.head())

target = iris.iloc[:,1].values

print(target)

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(data, target)

X = [
    [4.9, 1.1, 1.1, 1.2],
    [3.5, 2.0, 1.3, 4.8],
    [6.0, 3.0, 2.4, 2.0],
    [6.8, 3.2, 5.9, 2.3],
]

print(X)

results = knn.predict(X)

print(results)












