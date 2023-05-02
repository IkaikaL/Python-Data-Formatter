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

    # Add dataframe to dictionary
    wslp101DataFrames[value] = individualSheetWslp101

    # Get elevation values
    elevationWslp101 = wslp101[value].loc[:, ["Elevation"]]
    wslp101ElevationDataFrames[value] = elevationWslp101

    # Remove any NaN values before formatting
    nanValueIndexes = wslp101DataFrames[value][np.isnan(wslp101DataFrames[value]['prec'])]
    if not nanValueIndexes.empty:
        for indexes in nanValueIndexes.index:
            wslp101DataFrames[value] = wslp101DataFrames[value].drop(labels=indexes, axis=0)
            wslp101ElevationDataFrames[value] = wslp101ElevationDataFrames[value].drop(labels=indexes, axis=0)

    # Format Numbers
    wslp101DataFrames[value].iloc[:, [1]] *= 2000
    wslp101DataFrames[value].iloc[:, [2]] *= 2000
    wslp101DataFrames[value].iloc[:, [3]] *= 144

    wslp101DataFrames[value]['qt/pa'] = np.log10(wslp101DataFrames[value]['qt/pa'])
    wslp101DataFrames[value]['Rf'] = np.log10(wslp101DataFrames[value]['Rf'])

    # creating and inserting geology column
    wslp101GeoValues = np.zeros((len(wslp101DataFrames[value]), 1))
    wslp101DataFrames[value].insert(6, "geology", wslp101GeoValues, True)

    # Remove any Nan values after calculating Rf
    nanValueIndexes = wslp101DataFrames[value][np.isnan(wslp101DataFrames[value]['Rf'])]
    if not nanValueIndexes.empty:
        for indexes in nanValueIndexes.index:
            wslp101DataFrames[value] = wslp101DataFrames[value].drop(labels=indexes, axis=0)
            wslp101ElevationDataFrames[value] = wslp101ElevationDataFrames[value].drop(labels=indexes, axis=0)


wslp108DataFrames = {}
wslp108ElevationDataFrames = {}

for value in wslp108.keys():
    # Get values from every sheet in WSLP-101 using headers
    individualSheetWslp108 = wslp108[value].loc[:, ["Depth", "q_c", "fs", "Pw", "q_t", "Rf", "Total Stress"]]

    # Rename columns to be uniform
    individualSheetWslp108 = individualSheetWslp108.rename(columns={"Depth": "Depth ", "q_c": "qc", "Pw": "u2", "q_t": "qt/pa", "Total Stress": "prec"})

    # Add dataframe to dictionary
    wslp108DataFrames[value] = individualSheetWslp108

    # Get elevation values
    elevationWslp108 = wslp108[value].loc[:, ["Elevation"]]
    wslp108ElevationDataFrames[value] = elevationWslp108

    # Remove any NaN values before formatting
    nanValueIndexes = wslp108DataFrames[value][np.isnan(wslp108DataFrames[value]['prec'])]
    if not nanValueIndexes.empty:
        for indexes in nanValueIndexes.index:
            wslp108DataFrames[value] = wslp108DataFrames[value].drop(labels=indexes, axis=0)
            wslp108ElevationDataFrames[value] = wslp108ElevationDataFrames[value].drop(labels=indexes, axis=0)

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

    # Remove any Nan values after calculating Rf
    nanValueIndexes = wslp108DataFrames[value][np.isnan(wslp108DataFrames[value]['Rf'])]
    if not nanValueIndexes.empty:
        for indexes in nanValueIndexes.index:
            wslp108DataFrames[value] = wslp108DataFrames[value].drop(labels=indexes, axis=0)
            wslp108ElevationDataFrames[value] = wslp108ElevationDataFrames[value].drop(labels=indexes, axis=0)


wslp109DataFrames = {}
wslp109ElevationDataFrames = {}

for value in wslp109.keys():
    # Get values from every sheet in WSLP-101 using headers
    individualSheetWslp109 = wslp109[value].loc[:, ["Depth ", "qc", 'fs', 'u2', 'qt/pa', 'Rf', "sp"]]

    # Rename columns to be uniform
    individualSheetWslp109 = individualSheetWslp109.rename(columns={"sp": "prec"})

    # Add dataframe to dictionary
    wslp109DataFrames[value] = individualSheetWslp109

    # Get elevation values
    elevationWslp109 = wslp109[value].loc[:, ["Elevation"]]
    wslp109ElevationDataFrames[value] = elevationWslp109

    # Remove any NaN values before formatting
    nanValueIndexes = wslp109DataFrames[value][np.isnan(wslp109DataFrames[value]['prec'])]
    if not nanValueIndexes.empty:
        for indexes in nanValueIndexes.index:
            wslp109DataFrames[value] = wslp109DataFrames[value].drop(labels=indexes, axis=0)
            wslp109ElevationDataFrames[value] = wslp109ElevationDataFrames[value].drop(labels=indexes, axis=0)

    # Format Numbers
    wslp109DataFrames[value].iloc[:, [1]] *= 2000
    wslp109DataFrames[value]['qt/pa'] = np.log10(wslp109DataFrames[value]['qt/pa'])

    # creating and inserting geology column
    wslp109GeoValues = np.zeros((len(wslp109DataFrames[value]), 1))
    wslp109DataFrames[value].insert(6, "geology", wslp109GeoValues, True)

    # Remove any Nan values after calculating Rf
    nanValueIndexes = wslp109DataFrames[value][np.isnan(wslp109DataFrames[value]['Rf'])]
    if not nanValueIndexes.empty:
        for indexes in nanValueIndexes.index:
            wslp109DataFrames[value] = wslp109DataFrames[value].drop(labels=indexes, axis=0)
            wslp109DataFrames[value] = wslp109ElevationDataFrames[value].drop(labels=indexes, axis=0)


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

stations = pd.read_excel('WSLP_stations.xlsx', sheet_name=0)
stations = stations.loc[:, ["Stations", "CPT"]]

for value in wslp101DataFrames:
    wslp101PredictionDataFrames[value] = Mdl.predict(wslp101DataFrames[value])


for value in wslp101PredictionDataFrames:
    cpt = ""
    for char in value:
        if char.isdigit():
            cpt = cpt + char
    currentStation = stations.loc[stations["CPT"] == cpt]
    stationArray = np.repeat(a=int(currentStation.iloc[0]["Stations"]), repeats=len(wslp101ElevationDataFrames[value]))
    wslp101OutputDataFrames[value] = pd.DataFrame(data=stationArray)
    wslp101OutputDataFrames[value] = pd.concat([wslp101OutputDataFrames[value], pd.DataFrame(wslp101ElevationDataFrames[value], index=wslp101OutputDataFrames[value].index)], axis=1)
    wslp101OutputDataFrames[value] = pd.concat([wslp101OutputDataFrames[value], pd.DataFrame(wslp101PredictionDataFrames[value], index=wslp101OutputDataFrames[value].index)], axis=1)
    wslp101OutputDataFrames[value].columns = ['X', 'Y', 'I/O']

print(wslp101OutputDataFrames)


wslp108PredictionDataFrames = {}
wslp108OutputDataFrames = {}

for value in wslp108DataFrames:
    wslp108PredictionDataFrames[value] = Mdl.predict(wslp108DataFrames[value])
    cpt = ""
    for char in value:
        if char.isdigit():
            cpt = cpt + char
    currentStation = stations.loc[stations["CPT"] == cpt]
    stationArray = np.repeat(a=int(currentStation.iloc[0]["Stations"]), repeats=len(wslp108ElevationDataFrames[value]))
    wslp108OutputDataFrames[value] = pd.DataFrame(data=stationArray)
    wslp108OutputDataFrames[value] = pd.concat([wslp108OutputDataFrames[value], pd.DataFrame(wslp108ElevationDataFrames[value], index=wslp108OutputDataFrames[value].index)], axis=1)
    wslp108OutputDataFrames[value] = pd.concat([wslp108OutputDataFrames[value], pd.DataFrame(wslp108PredictionDataFrames[value], index=wslp108OutputDataFrames[value].index)], axis=1)
    wslp108OutputDataFrames[value].columns = ['X', 'Y', 'I/O']

print(wslp108OutputDataFrames)


wslp109PredictionDataFrames = {}
wslp109OutputDataFrames = {}

for value in wslp109DataFrames:
    wslp109PredictionDataFrames[value] = Mdl.predict(wslp109DataFrames[value])
    cpt = ""
    for char in value:
        if char.isdigit():
            cpt = cpt + char
    currentStation = stations.loc[stations["CPT"] == cpt]
    stationArray = np.repeat(a=int(currentStation.iloc[0]["Stations"]), repeats=len(wslp109ElevationDataFrames[value]))
    wslp109OutputDataFrames[value] = pd.DataFrame(data=stationArray)
    wslp109OutputDataFrames[value] = pd.concat([wslp109OutputDataFrames[value], pd.DataFrame(wslp109ElevationDataFrames[value], index=wslp109OutputDataFrames[value].index)], axis=1)
    wslp109OutputDataFrames[value] = pd.concat([wslp109OutputDataFrames[value], pd.DataFrame(wslp109PredictionDataFrames[value], index=wslp109OutputDataFrames[value].index)], axis=1)
    wslp109OutputDataFrames[value].columns = ['X', 'Y', 'I/O']

print(wslp109OutputDataFrames)


with pd.ExcelWriter('outputData.xlsx') as writer:
    current = 0
    for wslp101Sheet in wslp101OutputDataFrames:
        wslp101OutputDataFrames[wslp101Sheet].to_excel(writer, sheet_name='WSLP-101', index=False, startrow=current, header=False)
        current += len(wslp101OutputDataFrames[wslp101Sheet])
        print(current)
    current = 0
    for wslp108Sheet in wslp108OutputDataFrames:
        wslp108OutputDataFrames[wslp108Sheet].to_excel(writer, sheet_name='WSLP-108', index=False, startrow=current, header=False)
        current += len(wslp108OutputDataFrames[wslp108Sheet])
        print(current)
    current = 0
    for wslp109Sheet in wslp109OutputDataFrames:
        wslp109OutputDataFrames[wslp109Sheet].to_excel(writer, sheet_name='WSLP-109', index=False, startrow=current, header=False)
        current += len(wslp109OutputDataFrames[wslp109Sheet])
        print(current)
    current = 0

