import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import openpyxl
import xlsxwriter


# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

class DeleteNans:
    def __init__(self, valuesDataFrame=None, elevationDataFrame=None, output=None):
        self.valuesDataFrame = valuesDataFrame
        self.elevationDataFrame = elevationDataFrame
        self.output = output

    def removenansprec(self):
        nanValueIndexes = self.valuesDataFrame[np.isnan(self.valuesDataFrame['prec'])]
        if not nanValueIndexes.empty:
            for indexes in nanValueIndexes.index:
                self.valuesDataFrame = self.valuesDataFrame.drop(labels=indexes, axis=0)
                self.elevationDataFrame = self.elevationDataFrame.drop(labels=indexes, axis=0)

        return self.valuesDataFrame, self.elevationDataFrame

    def removenansrf(self):
        nanValueIndexes = self.valuesDataFrame[np.isnan(self.valuesDataFrame['Rf'])]
        if not nanValueIndexes.empty:
            for indexes in nanValueIndexes.index:
                self.valuesDataFrame = self.valuesDataFrame.drop(labels=indexes, axis=0)
                self.elevationDataFrame = self.elevationDataFrame.drop(labels=indexes, axis=0)

        return self.valuesDataFrame, self.elevationDataFrame

    def removenansy(self):
        nanValueIndexes = self.output[np.isnan(self.output['Y'])]
        for indexes in nanValueIndexes.index:
            self.output = self.output.drop(labels=indexes, axis=0)
        return self.output

class FormatOutput:
    def __init__(self, cpt, sheetName, elevations, predictions):
        self.cpt = cpt
        self.sheetName = sheetName
        self.elevations = elevations
        self.predictions = predictions

    def findcpt(self):
        for char in self.sheetName:
            if char.isdigit():
                self.cpt = self.cpt + char

        return self.cpt

    def combinecolumns(self):
        currentStation = stations.loc[stations["CPT"] == self.cpt]
        stationArray = np.repeat(a=int(currentStation.iloc[0]["Stations"]), repeats=len(self.elevations))
        outputDataFrame = pd.DataFrame(data=stationArray)
        outputDataFrame = pd.concat([outputDataFrame,
                                                    pd.DataFrame(self.elevations,
                                                                 index=outputDataFrame.index)], axis=1)
        outputDataFrame = pd.concat([outputDataFrame,
                                                    pd.DataFrame(self.predictions,
                                                                 index=outputDataFrame.index)], axis=1)
        outputDataFrame.columns = ['X', 'Y', 'I/O']

        return outputDataFrame


data = pd.read_excel('wslp_f29.xlsx', sheet_name=0)

wslp101 = pd.read_excel('WSLP-101.xlsx', header=8, sheet_name=None, skiprows=[9, 10, 11])

wslp108 = pd.read_excel('B-WSLP-108.xlsx', header=10, sheet_name=None, skiprows=[11, 12])

wslp109 = pd.read_excel('C-WSLP-109.xlsx', header=8, sheet_name=None, skiprows=[9, 10, 11])


wslp101DataFrames = {}
wslp101ElevationDataFrames = {}

for value in wslp101.keys():
    # Get values from every sheet in WSLP-101 using headers
    individualSheetWslp101 = wslp101[value].loc[:, ["Depth ", "qc", 'fs', 'u2', 'qt/pa', 'Rf', "s 'p"]]

    qcSubZeroValues = individualSheetWslp101.index.where(individualSheetWslp101['qc'] < 0)

    # Rename columns to be uniform
    individualSheetWslp101 = individualSheetWslp101.rename(columns={"s 'p": "prec"})

    # Add dataframe to dictionary
    wslp101DataFrames[value] = individualSheetWslp101

    # Get elevation values
    elevationWslp101 = wslp101[value].loc[:, ["Elevation"]]
    wslp101ElevationDataFrames[value] = elevationWslp101

    # Remove any NaN values before formatting
    modifiedDataFrames = DeleteNans(valuesDataFrame=wslp101DataFrames[value], elevationDataFrame=wslp101ElevationDataFrames[value])
    modifiedDataFrames.removenansprec()
    wslp101DataFrames[value] = modifiedDataFrames.valuesDataFrame
    wslp101ElevationDataFrames[value] = modifiedDataFrames.elevationDataFrame

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
    modifiedDataFrames = DeleteNans(valuesDataFrame=wslp101DataFrames[value], elevationDataFrame=wslp101ElevationDataFrames[value])
    modifiedDataFrames.removenansrf()
    wslp101DataFrames[value] = modifiedDataFrames.valuesDataFrame
    wslp101ElevationDataFrames[value] = modifiedDataFrames.elevationDataFrame

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
    modifiedDataFrames = DeleteNans(valuesDataFrame=wslp108DataFrames[value], elevationDataFrame=wslp108ElevationDataFrames[value])
    modifiedDataFrames.removenansprec()
    wslp108DataFrames[value] = modifiedDataFrames.valuesDataFrame
    wslp108ElevationDataFrames[value] = modifiedDataFrames.elevationDataFrame

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
    modifiedDataFrames = DeleteNans(valuesDataFrame=wslp108DataFrames[value], elevationDataFrame=wslp108ElevationDataFrames[value])
    modifiedDataFrames.removenansrf()
    wslp108DataFrames[value] = modifiedDataFrames.valuesDataFrame
    wslp108ElevationDataFrames[value] = modifiedDataFrames.elevationDataFrame

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
    modifiedDataFrames = DeleteNans(valuesDataFrame=wslp109DataFrames[value], elevationDataFrame=wslp109ElevationDataFrames[value])
    modifiedDataFrames.removenansprec()
    wslp109DataFrames[value] = modifiedDataFrames.valuesDataFrame
    wslp109ElevationDataFrames[value] = modifiedDataFrames.elevationDataFrame

    # Format Numbers
    wslp109DataFrames[value].iloc[:, [1]] *= 2000
    wslp109DataFrames[value]['qt/pa'] = np.log10(wslp109DataFrames[value]['qt/pa'])

    # creating and inserting geology column
    wslp109GeoValues = np.zeros((len(wslp109DataFrames[value]), 1))
    wslp109DataFrames[value].insert(6, "geology", wslp109GeoValues, True)

    # Remove any Nan values after calculating Rf
    modifiedDataFrames = DeleteNans(valuesDataFrame=wslp109DataFrames[value], elevationDataFrame=wslp109ElevationDataFrames[value])
    modifiedDataFrames.removenansrf()
    wslp109DataFrames[value] = modifiedDataFrames.valuesDataFrame
    wslp109ElevationDataFrames[value] = modifiedDataFrames.elevationDataFrame


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

yte = Y.loc[idx[round(p*m)+1:len(idx)-1]]


# Create a random forest model
Mdl = sklearn.ensemble.RandomForestClassifier()
Mdl.fit(xtr, np.ravel(ytr))

hte = Mdl.predict(xte)

accuracy = accuracy_score(yte, hte)

confusionMatrix = confusion_matrix(yte, hte)

htr = Mdl.predict(xtr)
accuracy = accuracy_score(ytr, htr)

confusionMatrix = confusion_matrix(ytr, htr)

htAll = Mdl.predict(X)
accuracy = accuracy_score(Y, htAll)

confusionMatrix = confusion_matrix(Y, htAll)


wslp101PredictionDataFrames = {}
wslp101OutputDataFrames = {}

stations = pd.read_excel('WSLP_stations.xlsx', sheet_name=0)
stations = stations.loc[:, ["Stations", "CPT"]]

# Format output by finding CPT and generating 3 columns
for value in wslp101DataFrames:
    wslp101PredictionDataFrames[value] = Mdl.predict(wslp101DataFrames[value])
    formatOutput = FormatOutput("", value, wslp101ElevationDataFrames[value], wslp101PredictionDataFrames[value])
    cpt = formatOutput.findcpt()
    output = formatOutput.combinecolumns()
    wslp101OutputDataFrames[value] = output
    deleteNans = DeleteNans(output=wslp101OutputDataFrames[value])
    wslp101OutputDataFrames[value] = deleteNans.removenansy()


wslp108PredictionDataFrames = {}
wslp108OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
for value in wslp108DataFrames:
    wslp108PredictionDataFrames[value] = Mdl.predict(wslp108DataFrames[value])
    formatOutput = FormatOutput("", value, wslp108ElevationDataFrames[value], wslp108PredictionDataFrames[value])
    cpt = formatOutput.findcpt()
    output = formatOutput.combinecolumns()
    wslp108OutputDataFrames[value] = output
    deleteNans = DeleteNans(output=wslp108OutputDataFrames[value])
    wslp108OutputDataFrames[value] = deleteNans.removenansy()


wslp109PredictionDataFrames = {}
wslp109OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
for value in wslp109DataFrames:
    wslp109PredictionDataFrames[value] = Mdl.predict(wslp109DataFrames[value])
    formatOutput = FormatOutput("", value, wslp109ElevationDataFrames[value], wslp109PredictionDataFrames[value])
    cpt = formatOutput.findcpt()
    output = formatOutput.combinecolumns()
    wslp109OutputDataFrames[value] = output
    deleteNans = DeleteNans(output=wslp109OutputDataFrames[value])
    wslp109OutputDataFrames[value] = deleteNans.removenansy()



# Output to Excel file
# Using current to indicate that writer should start after data already printed
with pd.ExcelWriter('outputData.xlsx') as writer:
    current = 0
    for wslp101Sheet in wslp101OutputDataFrames:
        wslp101OutputDataFrames[wslp101Sheet].to_excel(writer, sheet_name='WSLP-101', index=False, startrow=current, header=False)
        current += len(wslp101OutputDataFrames[wslp101Sheet])
    current = 0
    for wslp108Sheet in wslp108OutputDataFrames:
        wslp108OutputDataFrames[wslp108Sheet].to_excel(writer, sheet_name='WSLP-108', index=False, startrow=current, header=False)
        current += len(wslp108OutputDataFrames[wslp108Sheet])
    current = 0
    for wslp109Sheet in wslp109OutputDataFrames:
        wslp109OutputDataFrames[wslp109Sheet].to_excel(writer, sheet_name='WSLP-109', index=False, startrow=current, header=False)
        current += len(wslp109OutputDataFrames[wslp109Sheet])
    current = 0

workbook = xlsxwriter.Workbook('outputChart.xlsx')

wslp101Worksheet = workbook.add_worksheet()
wslp108Worksheet = workbook.add_worksheet()
wslp109Worksheet = workbook.add_worksheet()

headings = ['CPT', 'Elevation', 'Prediction']

wslp101Worksheet.write_row('A1', headings)
wslp108Worksheet.write_row('A1', headings)
wslp109Worksheet.write_row('A1', headings)

current = 1

for wslp101Sheet in wslp101OutputDataFrames:
    wslp101Worksheet.write_column(row=current, col=0, data=wslp101OutputDataFrames[wslp101Sheet]['X'])
    wslp101Worksheet.write_column(row=current, col=1, data=wslp101OutputDataFrames[wslp101Sheet]['Y'])
    wslp101Worksheet.write_column(row=current, col=2, data=wslp101OutputDataFrames[wslp101Sheet]['I/O'])
    current += len(wslp101OutputDataFrames[wslp101Sheet])

current = 1

for wslp108Sheet in wslp108OutputDataFrames:
    wslp108Worksheet.write_column(row=current, col=0, data=wslp108OutputDataFrames[wslp108Sheet]['X'])
    wslp108Worksheet.write_column(row=current, col=1, data=wslp108OutputDataFrames[wslp108Sheet]['Y'])
    wslp108Worksheet.write_column(row=current, col=2, data=wslp108OutputDataFrames[wslp108Sheet]['I/O'])
    current += len(wslp108OutputDataFrames[wslp108Sheet])

current = 1

for wslp109Sheet in wslp109OutputDataFrames:
    wslp109Worksheet.write_column(row=current, col=0, data=wslp109OutputDataFrames[wslp109Sheet]['X'])
    wslp109Worksheet.write_column(row=current, col=1, data=wslp109OutputDataFrames[wslp109Sheet]['Y'])
    wslp109Worksheet.write_column(row=current, col=2, data=wslp109OutputDataFrames[wslp109Sheet]['I/O'])
    current += len(wslp109OutputDataFrames[wslp109Sheet])


scatterPlot = wslp101Worksheet = workbook.add_chart({'type': 'scatter', })
workbook.close()
