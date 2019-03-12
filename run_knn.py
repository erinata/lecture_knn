import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("iris.csv", header=None)

print(iris.head())



