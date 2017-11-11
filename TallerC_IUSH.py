import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from  sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

flower = load_iris()
names = flower.feature_names
#print(names)

targets = flower.target_names
print(targets)

print(flower.data)

#print(flower.target)


x = flower.data
y = flower.target
#plt.scatter(x, y)

n_neighbors = 10

def normalScores(n_neighbors=2):
    scores = []
    for n in range(n_neighbors):
        knn = KNeighborsClassifier(n_neighbors=n+1)
        trainTest = train_test_split(x, y)
        x_train, x_test, y_train, y_test = trainTest

        knn.fit(x_train, y_train)
        score = knn.score(x_test, y_test)
        #print(score)
        scores.append(score)
    return scores

normalScores = normalScores(10)


plt.plot(normalScores)
#plt.show()

def crossScores(n_neighbors=2):
    scores = []
    for n in range(n_neighbors):
        knn = KNeighborsClassifier(n_neighbors=n+1)
        score = cross_val_score(knn, x, y, scoring='accuracy')
        mean = score.mean()
        #print(mean)
    scores.append(mean)
    return scores

def logisticScores(n_neighbors=2):
    scores = []
    for n in range(n_neighbors):
        # LOGISTIC REGRETION
        knn = KNeighborsClassifier(n_neighbors=n+1)
        trainTest = train_test_split(x, y)
        x_train, x_test, y_train, y_test = trainTest
        logisticRegretion = LogisticRegression()
        logisticRegretion.fit(x_train, y_train)
        score = logisticRegretion.score(x_test, y_test)
        print("Logistic Regretion ", score)
    scores.append(score)
    return scores


crossScores = crossScores(10)
logisticScores = logisticScores(10)
scores = {"cross validation": max(crossScores), "split": max(normalScores), "logistic": max(logisticScores)}
tupleWithScore = max(scores, key=scores.get), max(scores.values())
print("The best score was '%s' with %s of score"%(tupleWithScore))

"""Conclusion: Usually the best score is 1.0 by split algorithm"""

