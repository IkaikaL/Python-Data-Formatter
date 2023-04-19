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

newData = pd.read_excel('WSLP-101.xlsx', header=8, sheet_name=1, skiprows=[9])
#index_col= [9, 10, 13, 15, 20, 43, 44],


X = data.loc[:, ['Depth(ft)', 'qc(psf)', 'fs(psf)', 'u2(psf)', 'log(qt/Pa)', 'log(Rf)', 'geology', 'prec']]  # 8 feature columns
Y = data.loc[:, ['organic']]  # Output column
m = len(Y)

print(newData)

depthValues = newData.iloc[:, [9]]
print(depthValues)

qcValues = newData.iloc[:, [10]]
qcValues = qcValues * 2000
print(qcValues)

fsValues = newData.iloc[:, [13]]
print(fsValues)

u2Values = newData.iloc[:, [15]]
print(u2Values)

preConsolidationValues = newData.iloc[:, [20]]
preConsolidationValues = preConsolidationValues * 2000
print(preConsolidationValues)

qtPaValues = newData.iloc[:, [43]]
#qtPaValues = np.log10(qtPaValues)
print(qtPaValues)

rfValues = newData.iloc[:, [44]]
print(rfValues)

newDataFinished = depthValues
newDataFinished = np.append(qcValues, axis=1)
print(newDataFinished)


'''''
PredictorNames = ["depth", "qc", "fs", "u2", "log(qt/Pa)", "log(Rf)", "geology", "preconsolidation"]


# Randomizing data
p = .8

idx = np.random.permutation(m)

xtr = X.loc[idx[1:round(p*m)]]

ytr = Y.loc[idx[1:round(p*m)]]

xte = X.loc[idx[round(p*m)+1:len(idx)-1]]

yte = Y.loc[idx[round(p*m)+1:len(idx)-1]]

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

plt.show()
'''''