import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

data = pd.read_excel('wslp_f29.xlsx', sheet_name=0)

wslp101 = pd.read_excel('WSLP-101.xlsx', header=8, sheet_name=2, skiprows=[9, 10, 11])

wslp108 = pd.read_excel('B-WSLP-108.xlsx', header=10, sheet_name=1, skiprows=[11, 12])

wslp109 = pd.read_excel('C-WSLP-109.xlsx', header=8, sheet_name=1, skiprows=[9, 10, 11])

goodDataWslp101 = wslp101.loc[:, ["Depth ", "qc", 'fs', 'u2', 'qt/pa', 'Rf', "s 'p"]]
goodDataWslp101 = goodDataWslp101.rename(columns={"s 'p": "prec"})
goodDataWslp101 = goodDataWslp101.dropna()

goodDataWslp108 = wslp108.loc[:, ["Depth", "q_c", "fs", "Pw", "q_t", "Rf", "Total Stress"]]
goodDataWslp108 = goodDataWslp108.rename(columns={"q_c": "qc","Pw": "u2", "q_t": "qt", "Total Stress": "prec"})
goodDataWslp108 = goodDataWslp108.dropna()

goodDataWslp109 = wslp109.loc[:, ["Depth ", "qc", "fs", "u2", "qt/pa", "Rf", "sp"]]
goodDataWslp109 = goodDataWslp109.rename(columns={"sp": "prec"})
goodDataWslp109 = goodDataWslp109.dropna()

# formatting numbers
goodDataWslp101.iloc[:, [1]] *= 2000
goodDataWslp101.iloc[:, [2]] *= 2000
goodDataWslp101.iloc[:, [3]] *= 2000

goodDataWslp101['qt/pa'] = np.log10(goodDataWslp101['qt/pa'])
goodDataWslp101['Rf'] = np.log10(goodDataWslp101['Rf'])

# creating and inserting geology column
geoValuesWslp101 = np.zeros((len(goodDataWslp101), 1))
goodDataWslp101.insert(6, "geology", geoValuesWslp101, True)

# formatting numbers
goodDataWslp108.iloc[:, [1]] *= 2000
goodDataWslp108.iloc[:, [2]] *= 2000
goodDataWslp108.iloc[:, [3]] *= 144

goodDataWslp108['prec'] = .33 * (goodDataWslp108['qt'] * 2000 - goodDataWslp108['prec'])
goodDataWslp108.iloc[:, [4]] /= 1.0581

goodDataWslp108['qt'] = np.log10(goodDataWslp108['qt'])
goodDataWslp108['Rf'] = np.log10(goodDataWslp108['Rf'])

# creating and inserting geology column
geoValuesWslp108 = np.zeros((len(goodDataWslp108), 1))
goodDataWslp108.insert(6, "geology", geoValuesWslp108, True)

# formatting numbers
goodDataWslp109.iloc[:, [1]] *= 2000
goodDataWslp109['qt/pa'] = np.log10(goodDataWslp109['qt/pa'])
goodDataWslp109['Rf'] = np.log10(goodDataWslp109['Rf'])

# creating and inserting geology column
geoValuesWslp109 = np.zeros((len(goodDataWslp109), 1))
goodDataWslp109.insert(6, "geology", geoValuesWslp109, True)

# print dataframes
print(goodDataWslp101)
print(goodDataWslp108)
print(goodDataWslp109)

# pulling organic/not organic from charts (useless now)
'''''
testAnswers = wslp101.loc[:, ["Classification.1"]]
testAnswers.rename(columns={"Classification.1": "Classification"}, inplace = True)
testAnswers = testAnswers.dropna()
uniqueValues = pd.unique(testAnswers["Classification"].values.ravel())
valueMapping = {}
for i in uniqueValues:
    if i == "Organic":
        valueMapping[i] = 1
    else:
        valueMapping[i] = 0
testAnswers.Classification = [valueMapping[item] for item in testAnswers.Classification]
for i in uniqueValues:
    testAnswers.replace({i: valueMapping})
print(testAnswers)
'''''
dataLengthWslp101 = len(goodDataWslp101)
dataLengthWslp108 = len(goodDataWslp108)
dataLengthWslp109 = len(goodDataWslp109)

X = data.loc[:, ['Depth ', 'qc', 'fs', 'u2', 'qt/pa', 'Rf', 'geology', 'prec']]  # 8 feature columns
Y = data.loc[:, ['organic']]  # Output column
m = len(Y)



# Randomizing data
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=15)
# np.random.seed(0)
p = .8

idx = np.random.permutation(m)
idxTest = np.random.permutation(len(goodDataWslp101))


xtr = X.loc[idx[1:round(p*m)]]

ytr = Y.loc[idx[1:round(p*m)]]

xte = X.loc[idx[round(p*m)+1:len(idx)-1]]
xteTest = goodDataWslp101.loc[idxTest[round(p*dataLengthWslp101)+1:len(idxTest)-1]]

yte = Y.loc[idx[round(p*m)+1:len(idx)-1]]
#yteTest = testAnswers.loc[idxTest[round(p*dataLengthWslp101)+1:len(idxTest)-1]]


# Create a random forest model
Mdl = sklearn.ensemble.RandomForestClassifier()
Mdl.fit(xtr, np.ravel(ytr))

hte = Mdl.predict(xte)

accuracy = accuracy_score(yte, hte)

confusionMatrix = confusion_matrix(yte, hte)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()

htr = Mdl.predict(xtr)
accuracy = accuracy_score(ytr, htr)

confusionMatrix = confusion_matrix(ytr, htr)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()

htAll = Mdl.predict(X)
accuracy = accuracy_score(Y, htAll)

confusionMatrix = confusion_matrix(Y, htAll)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()


hteTest = Mdl.predict(xteTest)
confusionMatrix = confusion_matrix(xte, hteTest)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()

print(accuracy)
plt.show()

'''''
'''''