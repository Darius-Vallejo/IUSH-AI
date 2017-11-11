import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

db = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
data = pd.read_csv(db)

def clean(row):
    return [0 if item == "?" else item for item in row]

values = list(map(lambda r: clean(r), data.values))
values = np.array(values, dtype=np.float)

attributes = len(values.transpose()) - 1
x = values[:, 1: attributes]
y = [item - 1 for item in values[:, 0]]

x_train, x_test,  y_train, y_test = train_test_split(x, y)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def classifier(layerSizesTuple = (13, 13, 13), maxIter = 500, alpha = 0.02):
    mlp = MLPClassifier(hidden_layer_sizes=layerSizesTuple, max_iter=maxIter, alpha=alpha, early_stopping=True)
    mlp.fit(x_train, y_train)
    predictions = mlp.predict(x_test)
    classification_report(y_test, predictions)
    print("score ", mlp.score(x_test, y_test))
    return mlp.score(x_test, y_test)



scores = {}
while True:
    layer0 = np.random.randint(1, 11)
    layer1 = np.random.randint(7, 11)
    layer2 = np.random.randint(1, 11)
    layer = 10, layer1
    alpha = np.random.random_sample()
    maxInter = np.random.randint(100, 200)
    tuple = (layer, alpha, maxInter)
    score = classifier(layerSizesTuple=layer, alpha=alpha, maxIter=maxInter)
    scores[str(score)] = tuple
    if score > 0.9: break

tupleWithScore = max(scores), max(scores.values()), len(scores)
print("The best score was '%s' for values %s with iterations %s"%(tupleWithScore))