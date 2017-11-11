import pandas as pd
import matplotlib.pyplot as plt
import  PyplotChar as pltc
#pd.set_option('display.mpl_style', 'default')
#plt.rcParams['figure.figsize'] = (25, 5)
# data = pd.read_excel("examples.") csv
#data = pd.read_csv("examples.csv")
#data.head(15)
#data.colum2[data.column1 >=5]
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', ".", 'Milner', 'Cooze'],
        'age': [42, 52, 36, 24, 73],
        'preTestScore': [4, 24, 31, ".", "."],
        'postTestScore': ["25,000", "94,000", 57, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])

df.to_csv('example.csv')
data = pd.read_csv("example.csv")
print(data.head(1))
pltc.PLT.show(data.head(1))
"""scikitlearn"""
