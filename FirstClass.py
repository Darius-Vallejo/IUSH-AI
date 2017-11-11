from numpy import  *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import pylab
import PyplotChar as PLT

x = np.linspace(0,30)
print(x)

numbers = np.random.random(10)
print (numbers)

# print("Hi I'm an string with {0} {1}".format(20, "things"))


a = np.array([20, 9, 2., 1., 40.443])
# print(a[0])


# np.lookfor("plus")

np.e
np.pi
np.log(9)

# print a.dtype

# print map(lambda number: number * .2, a)

stringArray = np.array(["2", 34], dtype=str)

# print stringArray.astype(int)

# PLT.PLT.show([[2, 30], [4, 9]])

dimensions = (2, 5)

empty = empty(dimensions)

def printForFunction(func, args=""):
    format = "%s ---> %s"
    tuple = (func, args)
    if args == "":
        line = "-"*10
        format = line + " %s " + line
        tuple = func
    print(format % tuple)

printForFunction("zeros", np.zeros(((3, 9))))
printForFunction("zeros", np.zeros(2))
printForFunction("unos", np.ones(((3, 9))))
identify = np.identity(3)
printForFunction("identify", identify)
printForFunction("identify shape", identify.shape)
onesForShape = np.ones(identify.shape)
printForFunction("ones for shape", onesForShape)
onesForMatrix = np.ones_like(identify)
printForFunction("onesForMatrix", onesForMatrix)

printForFunction("RANGES")
printForFunction("range", np.linspace(1, 5, 5))
printForFunction("log in array", np.logspace(4, 5, num=5))
printForFunction("normal array ", np.array([1, 5, 5]))
x = np.linspace(0, 1, 5)


