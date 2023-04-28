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

wslp101 = pd.read_excel('WSLP-101.xlsx', header=8, sheet_name=None, skiprows=[9, 10, 11])

wslp108 = pd.read_excel('B-WSLP-108.xlsx', header=10, sheet_name=None, skiprows=[11, 12])

wslp109 = pd.read_excel('C-WSLP-109.xlsx', header=8, sheet_name=None, skiprows=[9, 10, 11])


wslp101DataFrames = {}
wslp101ElevationDataFrames = {}

for value in wslp101.keys():
    # Get values from every sheet in WSLP-101 using headers
    individualSheetWslp101 = wslp101[value].loc[:, ["Depth ", "qc", 'fs', 'u2', 'qt/pa', 'Rf', "s 'p"]]

    # Rename columns to be uniform
    individualSheetWslp101 = individualSheetWslp101.rename(columns={"s 'p": "prec"})
    individualSheetWslp101 = individualSheetWslp101.dropna()

    # Add dataframe to dictionary
    wslp101DataFrames[value] = individualSheetWslp101

    # Get elevation values
    elevationWslp101 = wslp101[value].loc[:, ["Elevation"]]
    elevationWslp101 = elevationWslp101.dropna()
    wslp101ElevationDataFrames[value] = elevationWslp101

    # Format Numbers
    wslp101DataFrames[value].iloc[:, [1]] *= 2000
    wslp101DataFrames[value].iloc[:, [2]] *= 2000
    wslp101DataFrames[value].iloc[:, [3]] *= 144

    wslp101DataFrames[value]['qt/pa'] = np.log10(wslp101DataFrames[value]['qt/pa'])
    wslp101DataFrames[value]['Rf'] = np.log10(wslp101DataFrames[value]['Rf'])

    # creating and inserting geology column
    wslp101GeoValues = np.zeros((len(wslp101DataFrames[value]), 1))
    wslp101DataFrames[value].insert(6, "geology", wslp101GeoValues, True)


wslp108DataFrames = {}
wslp108ElevationDataFrames = {}

for value in wslp108.keys():
    # Get values from every sheet in WSLP-101 using headers
    individualSheetWslp108 = wslp108[value].loc[:, ["Depth", "q_c", "fs", "Pw", "q_t", "Rf", "Total Stress"]]

    # Rename columns to be uniform
    individualSheetWslp108 = individualSheetWslp108.rename(columns={"Depth": "Depth ", "q_c": "qc", "Pw": "u2", "q_t": "qt/pa", "Total Stress": "prec"})
    individualSheetWslp108 = individualSheetWslp108.dropna()

    # Add dataframe to dictionary
    wslp108DataFrames[value] = individualSheetWslp108

    # Get elevation values
    elevationWslp108 = wslp108[value].loc[:, ["Elevation"]]
    elevationWslp108 = elevationWslp108.dropna()

    wslp108ElevationDataFrames[value] = elevationWslp108

    # Format Numbers
    wslp108DataFrames[value].iloc[:, [1]] *= 2000
    wslp108DataFrames[value].iloc[:, [2]] *= 2000
    wslp108DataFrames[value].iloc[:, [3]] *= 144
    wslp108DataFrames[value].iloc[:, [4]] /= 1.0581
    wslp108DataFrames[value]['prec'] = .33 * (5 - wslp108DataFrames[value]['qt/pa']) * 2000 - wslp108DataFrames[value]['prec']

    wslp108DataFrames[value]['qt/pa'] = np.log10(wslp108DataFrames[value]['qt/pa'])
    wslp108DataFrames[value]['Rf'] = np.log10(wslp108DataFrames[value]['Rf'])

    # creating and inserting geology column
    wslp108GeoValues = np.zeros((len(wslp108DataFrames[value]), 1))
    wslp108DataFrames[value].insert(6, "geology", wslp108GeoValues, True)


wslp109DataFrames = {}
wslp109ElevationDataFrames = {}

for value in wslp109.keys():
    # Get values from every sheet in WSLP-101 using headers
    individualSheetWslp109 = wslp109[value].loc[:, ["Depth ", "qc", 'fs', 'u2', 'qt/pa', 'Rf', "sp"]]

    # Rename columns to be uniform
    individualSheetWslp109 = individualSheetWslp109.rename(columns={"sp": "prec"})
    individualSheetWslp109 = individualSheetWslp109.dropna()

    # Add dataframe to dictionary
    wslp109DataFrames[value] = individualSheetWslp109

    # Get elevation values
    elevationWslp109 = wslp109[value].loc[:, ["Elevation"]]
    elevationWslp109 = elevationWslp109.dropna()

    wslp109ElevationDataFrames[value] = elevationWslp109

    # Format Numbers
    wslp109DataFrames[value].iloc[:, [1]] *= 2000
    wslp109DataFrames[value]['qt/pa'] = np.log10(wslp109DataFrames[value]['qt/pa'])

    # creating and inserting geology column
    wslp109GeoValues = np.zeros((len(wslp109DataFrames[value]), 1))
    wslp109DataFrames[value].insert(6, "geology", wslp109GeoValues, True)

'''''
# print dataframes
print(wslp101DataFrames)
print(wslp108DataFrames)
print(wslp109DataFrames)
'''''

dataLengthWslp101 = len(wslp101DataFrames[list(wslp101DataFrames.keys())[0]])
dataLengthWslp108 = len(wslp108DataFrames[list(wslp108DataFrames.keys())[0]])
dataLengthWslp109 = len(wslp109DataFrames[list(wslp109DataFrames.keys())[0]])


X = data.loc[:, ['Depth ', 'qc', 'fs', 'u2', 'qt/pa', 'Rf', 'geology', 'prec']]  # 8 feature columns
Y = data.loc[:, ['organic']]  # Output column
m = len(Y)


# Randomizing data
np.set_printoptions(suppress=True, precision=15)
np.random.seed(0)
p = .8

idx = np.random.permutation(m)

xtr = X.loc[idx[1:round(p*m)]]

ytr = Y.loc[idx[1:round(p*m)]]

xte = X.loc[idx[round(p*m)+1:len(idx)-1]]

'''''
newDataTestWslp101 = goodDataWslp101.loc[idxWslp101[round(p*dataLengthWslp101)+1:len(idxWslp101)-1]]
newDataTestWslp108 = goodDataWslp108.loc[idxWslp108[round(p*dataLengthWslp108)+1:len(idxWslp108)-1]]
newDataTestWslp109 = goodDataWslp109.loc[idxWslp109[round(p*dataLengthWslp109)+1:len(idxWslp109)-1]]

trimmedElevationWslp101 = elevationWslp101.loc[idxWslp101[round(p*dataLengthWslp101)+1:len(idxWslp101)-1]]
trimmedElevationWslp108 = elevationWslp108.loc[idxWslp108[round(p*dataLengthWslp108)+1:len(idxWslp108)-1]]
trimmedElevationWslp109 = elevationWslp109.loc[idxWslp109[round(p*dataLengthWslp109)+1:len(idxWslp109)-1]]
'''''

yte = Y.loc[idx[round(p*m)+1:len(idx)-1]]


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

wslp101PredictionDataFrames = {}
wslp101OutputDataFrames = {}

print(wslp101DataFrames["002Cy"])
print(len(wslp101DataFrames["002Cy"]))
print(wslp101DataFrames["002Cy"].dropna)
# need to find Nan value inside of 002Cy and remove the line along with the line in elevation

for value in wslp101DataFrames:
    wslp101PredictionDataFrames[value] = Mdl.predict(wslp101DataFrames[value])

for value in wslp101PredictionDataFrames:
    wslp101OutputDataFrames[value] = pd.DataFrame(data=wslp101ElevationDataFrames[value])
    wslp101OutputDataFrames[value] = pd.concat([wslp101OutputDataFrames[value], pd.DataFrame(wslp101PredictionDataFrames[value], index=wslp101OutputDataFrames[value].index)], axis=1)


'''''
wslp101OutPutDataFrames[value] = pd.DataFrame(wslp101PredictionDataFrames, index=elevationWslp101.index)
wslp101OutPutDataFrames[value] = wslp101OutPutDataFrames[value].rename(columns={0: 'pred'})
wslp101OutPutDataFrames[value] = pd.concat([elevationWslp101, wslp101OutPutDataFrames[value]], axis=1)

wslp108PredictionDataFrames = {}

for value in wslp108DataFrames:
    wslp108PredictionDataFrames[value] = Mdl.predict(wslp108DataFrames[value].dropna())


wslp109PredictionDataFrames = {}

for value in wslp109DataFrames:
    wslp109PredictionDataFrames[value] = Mdl.predict(wslp109DataFrames[value].dropna())




predictionDataFrameWslp108 = pd.DataFrame(wslp108PredictionDataFrames, index=elevationWslp108.index)
predictionDataFrameWslp108 = predictionDataFrameWslp108.rename(columns={0: 'pred'})
outputDataWslp108 = pd.concat([elevationWslp108, predictionDataFrameWslp108], axis=1)

predictionDataFrameWslp109 = pd.DataFrame(wslp109PredictionDataFrames, index=elevationWslp109.index)
predictionDataFrameWslp109 = predictionDataFrameWslp109.rename(columns={0: 'pred'})
outputDataWslp109 = pd.concat([elevationWslp109, predictionDataFrameWslp109], axis=1)

print(outputDataWslp101)

with pd.ExcelWriter('outputData.xlsx') as writer:
    outputDataWslp101.to_excel(writer, sheet_name='WSLP-101')
    outputDataWslp108.to_excel(writer, sheet_name='WSLP-108')
    outputDataWslp109.to_excel(writer, sheet_name='WSLP-109')

'''''