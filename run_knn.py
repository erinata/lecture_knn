import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("iris.csv", header=None)

print(iris.head())

# data = iris[iris.columns[2:6]]

data = iris.iloc[:,2:6]

print(data.head())

target = iris.iloc[:,1]

print(target)

