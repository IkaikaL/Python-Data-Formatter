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

newData = pd.read_excel('WSLP-101.xlsx', header=8, sheet_name=1, skiprows=[9, 10, 11])


goodData = newData.loc[:, ["Depth ", "qc", 'fs', 'u2', 'qt/pa', 'Rf', "s 'p"]]
goodData = goodData.dropna()

testAnswers = newData.loc[:, ["Classification.1"]]
testAnswers.rename(columns={"Classification.1": "Classification"}, inplace = True)
testAnswers = testAnswers.dropna()
uniqueValues = pd.unique(testAnswers["Classification"].values.ravel())
valueMapping = {uniqueValues[0]: 0, uniqueValues[1]: 1, uniqueValues[2]: 0, uniqueValues[3]: 0, uniqueValues[4]: 0, uniqueValues[5]: 0}
testAnswers.Classification = [valueMapping[item] for item in testAnswers.Classification]
testAnswers.replace({uniqueValues[0]: valueMapping, uniqueValues[1]: valueMapping, uniqueValues[2]: valueMapping, uniqueValues[3]: valueMapping, uniqueValues[4]: valueMapping, uniqueValues[5]: valueMapping})
print(testAnswers)

X = data.loc[:, ['Depth(ft)', 'qc(psf)', 'fs(psf)', 'u2(psf)', 'log(qt/Pa)', 'log(Rf)', 'geology', 'prec']]  # 8 feature columns
Y = data.loc[:, ['organic']]  # Output column
m = len(Y)

goodData.iloc[:, [1]] *= 2000

goodData.iloc[:, [2]] *= 2000

goodData.iloc[:, [3]] *= 2000

goodData['qt/pa'] = np.log10(goodData['qt/pa'])

goodData['Rf'] = np.log10(goodData['Rf'])

geoValues = np.zeros((len(goodData), 1))
goodData['geology'] = geoValues

print(goodData)

# Randomizing data
p = .8

idx = np.random.permutation(m)
idxTest = np.random.permutation(len(goodData))


xtr = X.loc[idx[1:round(p*m)]]
xtrTest = goodData.loc[idxTest[1:round(p*m)]]

ytr = Y.loc[idx[1:round(p*m)]]
#ytrTest =

xte = X.loc[idx[round(p*m)+1:len(idx)-1]]
xteTest = goodData.loc[idxTest[round(p*m)+1:len(idxTest)-1]]

yte = Y.loc[idx[round(p*m)+1:len(idx)-1]]
#yteTest =


print(idx)
print(xtr)
print(ytr)
print(xte)
print(yte)
'''''
'''''
# Create a random forest model
Mdl = sklearn.ensemble.RandomForestClassifier()
Mdl.fit(xtr, np.ravel(ytr))

hte = Mdl.predict(xte)

accuracy = accuracy_score(yte, hte)

confusionMatrix = confusion_matrix(yte, hte)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()

htr = Mdl.predict(xtr)
htrTest = Mdl.predict(xtrTest)
accuracy = accuracy_score(ytr, htr)

confusionMatrix = confusion_matrix(ytr, htr)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()

htAll = Mdl.predict(X)
accuracy = accuracy_score(Y, htAll)

confusionMatrix = confusion_matrix(Y, htAll)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()


hteTest = Mdl.predict(xteTest)

#plt.show()
'''''
'''''