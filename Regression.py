import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("home_data.csv")
data.head(10)
print(data.tail())

home = data.set_index(["id", "price"])
x = data.sqft_living.values
y = data.price.values

print("------- ", data.sqft_living.values)
trainTest = train_test_split(x,y)
x_train, x_test, y_train, y_test = trainTest
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
model = LinearRegression()
model.fit(x_train.reshape(-1, 1), y_train.reshape(16209, 1))
print(x_test[0])
print(y_test[0])

model.predict(x_test[0])
model.predict(y_test[0])
a = model.score(x_test.reshape(-1,1), y_test.reshape(-1,1))
print(a)

coef = model.coef_
intercept = model.intercept_

plt.show()
"""
home = data.set_index(["id", "price"])
x = np.array([data.sqft_living.values, data.id.values])
y = np.array([data.price.values, data.price.values])

trainTest = train_test_split(x,y)
x_train, x_test, y_train, y_test = trainTest
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

rX = x_train.reshape(21613, 1)
rY = y_train.reshape(-1, 1)
model = LinearRegression()
print(" x ", data.id.values, "y ", y)
print(" HOmbre ", rX)
model.fit(rX, rY)
print("-----> ",  model.predict(x_test[0][0]))
print(y_test[0][0])
a = model.score(rX, rY)
print(a)
coef = model.coef_
intercept = model.intercept_
plt.show()
model.predict(x_test[0])
model.predict(y_test[0])
a = model.score(x_test.reshape(5404,1), y_test.reshape(5404,1))
print(a)

coef = model.coef_
intercept = model.intercept_

plt.show()
"""


"""
tarea
Linear regression()
multi variable
"""