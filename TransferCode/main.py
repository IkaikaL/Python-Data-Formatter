import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import openpyxl

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

elevationWslp101 = wslp101.loc[:, ["Elevation"]]
elevationWslp101 = elevationWslp101.dropna()

goodDataWslp108 = wslp108.loc[:, ["Depth", "q_c", "fs", "Pw", "q_t", "Rf", "Total Stress"]]
goodDataWslp108 = goodDataWslp108.rename(columns={"Depth": "Depth ","q_c": "qc","Pw": "u2", "q_t": "qt/pa", "Total Stress": "prec"})
goodDataWslp108 = goodDataWslp108.dropna()

elevationWslp108 = wslp108.loc[:, ["Elevation"]]
elevationWslp108 = elevationWslp108.dropna()

goodDataWslp109 = wslp109.loc[:, ["Depth ", "qc", "fs", "u2", "qt/pa", "Rf", "sp"]]
goodDataWslp109 = goodDataWslp109.rename(columns={"sp": "prec"})
goodDataWslp109 = goodDataWslp109.dropna()

elevationWslp109 = wslp109.loc[:, ["Elevation"]]
elevationWslp109 = elevationWslp109.dropna()


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
goodDataWslp108.iloc[:, [4]] /= 1.0581
goodDataWslp108['prec'] = .33 * (5 - goodDataWslp108['qt/pa']) * 2000 - goodDataWslp108['prec']

goodDataWslp108['qt/pa'] = np.log10(goodDataWslp108['qt/pa'])
goodDataWslp108['Rf'] = np.log10(goodDataWslp108['Rf'])

# creating and inserting geology column
geoValuesWslp108 = np.zeros((len(goodDataWslp108), 1))
goodDataWslp108.insert(6, "geology", geoValuesWslp108, True)

# formatting numbers
goodDataWslp109.iloc[:, [1]] *= 2000
goodDataWslp109['qt/pa'] = np.log10(goodDataWslp109['qt/pa'])

# creating and inserting geology column
geoValuesWslp109 = np.zeros((len(goodDataWslp109), 1))
goodDataWslp109.insert(6, "geology", geoValuesWslp109, True)

'''''
# print dataframes
print(goodDataWslp101)
print(goodDataWslp108)
print(goodDataWslp109)
'''''

'''''
# pulling organic/not organic from charts (useless now)
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
'''''
dataLengthWslp101 = len(goodDataWslp101)
dataLengthWslp108 = len(goodDataWslp108)
dataLengthWslp109 = len(goodDataWslp109)

X = data.loc[:, ['Depth ', 'qc', 'fs', 'u2', 'qt/pa', 'Rf', 'geology', 'prec']]  # 8 feature columns
Y = data.loc[:, ['organic']]  # Output column
m = len(Y)



# Randomizing data
np.set_printoptions(suppress=True, precision=15)
np.random.seed(0)
p = .8

idx = np.random.permutation(m)
idxWslp101 = np.random.permutation(len(goodDataWslp101))
idxWslp108 = np.random.permutation(len(goodDataWslp108))
idxWslp109 = np.random.permutation(len(goodDataWslp109))

xtr = X.loc[idx[1:round(p*m)]]

ytr = Y.loc[idx[1:round(p*m)]]

xte = X.loc[idx[round(p*m)+1:len(idx)-1]]
newDataTestWslp101 = goodDataWslp101.loc[idxWslp101[round(p*dataLengthWslp101)+1:len(idxWslp101)-1]]
newDataTestWslp108 = goodDataWslp108.loc[idxWslp108[round(p*dataLengthWslp108)+1:len(idxWslp108)-1]]
newDataTestWslp109 = goodDataWslp109.loc[idxWslp109[round(p*dataLengthWslp109)+1:len(idxWslp109)-1]]

trimmedElevationWslp101 = elevationWslp101.loc[idxWslp101[round(p*dataLengthWslp101)+1:len(idxWslp101)-1]]
trimmedElevationWslp108 = elevationWslp108.loc[idxWslp108[round(p*dataLengthWslp108)+1:len(idxWslp108)-1]]
trimmedElevationWslp109 = elevationWslp109.loc[idxWslp109[round(p*dataLengthWslp109)+1:len(idxWslp109)-1]]


yte = Y.loc[idx[round(p*m)+1:len(idx)-1]]
#newDataTestSampleAnswers = testAnswers.loc[idxTest[round(p*dataLengthWslp101)+1:len(idxTest)-1]]


# Create a random forest model
Mdl = sklearn.ensemble.RandomForestClassifier()
Mdl.fit(xtr, np.ravel(ytr))

hte = Mdl.predict(xte)

accuracy = accuracy_score(yte, hte)

confusionMatrix = confusion_matrix(yte, hte)
#ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()

htr = Mdl.predict(xtr)
accuracy = accuracy_score(ytr, htr)

confusionMatrix = confusion_matrix(ytr, htr)
#ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()

htAll = Mdl.predict(X)
accuracy = accuracy_score(Y, htAll)

confusionMatrix = confusion_matrix(Y, htAll)
#ConfusionMatrixDisplay(confusion_matrix=confusionMatrix).plot()


newDataPredictionWslp101 = Mdl.predict(newDataTestWslp101)
#accuracy = accuracy_score(newDataPrediction, newDataTestSampleAnswers)
print(newDataPredictionWslp101)

newDataPredictionWslp108 = Mdl.predict(newDataTestWslp108)
#accuracy = accuracy_score(newDataPrediction, newDataTestSampleAnswers)
print(newDataPredictionWslp108)

newDataPredictionWslp109 = Mdl.predict(newDataTestWslp109)
#accuracy = accuracy_score(newDataPrediction, newDataTestSampleAnswers)
print(newDataPredictionWslp109)

predictionDataFrameWslp101 = pd.DataFrame(newDataPredictionWslp101, index=trimmedElevationWslp101.index)
predictionDataFrameWslp101 = predictionDataFrameWslp101.rename(columns={0: 'pred'})
outputDataWslp101 = pd.concat([trimmedElevationWslp101, predictionDataFrameWslp101], axis=1)

predictionDataFrameWslp108 = pd.DataFrame(newDataPredictionWslp108, index=trimmedElevationWslp108.index)
predictionDataFrameWslp108 = predictionDataFrameWslp108.rename(columns={0: 'pred'})
outputDataWslp108 = pd.concat([trimmedElevationWslp108, predictionDataFrameWslp108], axis=1)

predictionDataFrameWslp109 = pd.DataFrame(newDataPredictionWslp109, index=trimmedElevationWslp109.index)
predictionDataFrameWslp109 = predictionDataFrameWslp109.rename(columns={0: 'pred'})
outputDataWslp109 = pd.concat([trimmedElevationWslp109, predictionDataFrameWslp109], axis=1)


with pd.ExcelWriter('outputData.xlsx') as writer:
    outputDataWslp101.to_excel(writer, sheet_name='WSLP-101')
    outputDataWslp108.to_excel(writer, sheet_name='WSLP-108')
    outputDataWslp109.to_excel(writer, sheet_name='WSLP-109')

'''''
'''''