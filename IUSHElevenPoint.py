"""
Please notice that each import is starting the file to avoid the noise nearby the another logic or code

"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2)
visitas = np.random.normal(3.0, 1.0, 1000)

promCompras = np.random.normal(50.0, 10.0, 1000) / visitas

x = visitas
y = promCompras

def sort(x):
    try:
        c = x.shape[1]
    except:
        c = 1
    x = x.reshape(x.shape[0], c)
    return x

def log(name="", *options):
    print(name, "-"*10, ">", options)

plt.scatter(visitas, promCompras)
#plt.show()

#PARTIR LOS DATOS CON LA FUNCION TRAIN_TEST_SPLIT CON UN 33% EN LOS DATOS DE TESTING
trainTest = train_test_split(visitas, promCompras, test_size=0.33)
x_train, x_test, y_train, y_test = trainTest

y_train = sort(y_train)
x_train = sort(x_train)
x_test = sort(x_test)
y_test = sort(y_test)

def getMaxScoreFor(degrees = 2):
    scores = []
    for degree in range(degrees - 1):
        poly = PolynomialFeatures(degree=degree+1)
        x_train_pol = poly.fit_transform(x_train)
        x_test_pol = poly.fit_transform(x_test)
        model = LinearRegression()
        model.fit(x_train_pol, y_train)

        #CUALES SON LOS COEFICIENTES?
        #log("the coefficients are", model.coef_)

        #CUAL ES EL INTERCEPTO?
        #log("the intercepts are", model.intercept_)

        #plot
        xx = np.linspace(0, 7, 100)
        xx_quadratic = poly.transform(xx.reshape(xx.shape[0], 1))
        plt.plot(xx, model.predict(xx_quadratic), c='r', linestyle='--')
        plt.scatter(x, y)
        #plt.show()

        #cual es la exactitud del modelo
        score = model.score(x_test_pol, y_test.reshape(-1, 1))
        log("score:", score, "degree: ", degree)
        scores.append(score)
    return max(scores)


"""
SEGUN LO ANTERIOR CREAR UN CICLO 'FOR' QUE ME MUESTRE EL MEJOR SCORE
VARIANDO EL ORDEN DEL POLINOMIO DE LA REGRESION
"""

log("best score: ", getMaxScoreFor(20))
