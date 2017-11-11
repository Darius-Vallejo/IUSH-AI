import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data= pd.read_csv(url,names=names)
print(data.head())
print(data.describe())
print(data.describe().transpose())

y=data.values[:,8]
X=data.values[:,0:8]

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500,alpha=0.02)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

print(predictions[1])
print(y_test[1])
print(classification_report(y_test,predictions))
print(mlp.score(X_test,y_test))
