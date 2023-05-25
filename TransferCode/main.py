import numpy as np
import pandas as pd
import sklearn.ensemble
import openpyxl
import xlsxwriter

# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class DeleteNans:
    def __init__(self, valuesDataFrame=None, elevationDataFrame=None, output=None):
        self.valuesDataFrame = valuesDataFrame
        self.elevationDataFrame = elevationDataFrame
        self.output = output

    '''''
    def removenansprec(self):
        nanValueIndexes = self.valuesDataFrame[pd.isna(self.valuesDataFrame['prec'])]
        if not nanValueIndexes.empty:
            for indexes in nanValueIndexes.index:
                self.valuesDataFrame = self.valuesDataFrame.drop(labels=indexes, axis=0)
                self.elevationDataFrame = self.elevationDataFrame.drop(labels=indexes, axis=0)

        return self.valuesDataFrame, self.elevationDataFrame

    '''''

    def removenansrf(self):
        nanValueIndexes = self.valuesDataFrame[pd.isna(self.valuesDataFrame['Rf'])]
        infValueIndexes = self.valuesDataFrame[np.isinf(self.valuesDataFrame['Rf'])]
        if not nanValueIndexes.empty:
            for indexes in nanValueIndexes.index:
                self.valuesDataFrame = self.valuesDataFrame.drop(labels=indexes, axis=0)
                self.elevationDataFrame = self.elevationDataFrame.drop(labels=indexes, axis=0)

        if not infValueIndexes.empty:
            for indexes in infValueIndexes.index:
                self.valuesDataFrame = self.valuesDataFrame.drop(labels=indexes, axis=0)
                self.elevationDataFrame = self.elevationDataFrame.drop(labels=indexes, axis=0)

        return self.valuesDataFrame, self.elevationDataFrame

    def removenansy(self):
        nanValueIndexes = self.output[pd.isna(self.output['Y'])]
        for indexes in nanValueIndexes.index:
            self.output = self.output.drop(labels=indexes, axis=0)
        return self.output

    def removenansqtpa(self):
        nanValueIndexes = self.valuesDataFrame[pd.isna(self.valuesDataFrame['qt/pa'])]
        if not nanValueIndexes.empty:
            for indexes in nanValueIndexes.index:
                self.valuesDataFrame = self.valuesDataFrame.drop(labels=indexes, axis=0)
                self.elevationDataFrame = self.elevationDataFrame.drop(labels=indexes, axis=0)

        return self.valuesDataFrame, self.elevationDataFrame

class FormatOutput:
    def __init__(self, cpt, sheetName, elevations, predictions, stations=None):
        self.cpt = cpt
        self.sheetName = sheetName
        self.elevations = elevations
        self.predictions = predictions
        self.stations = stations

    def findcpt(self):
        for char in self.sheetName:
            if char.isdigit():
                self.cpt = self.cpt + char

        return self.cpt

    def combinecolumns(self):
        currentStation = self.stations.loc[self.stations["CPT"] == self.cpt]
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


class wslpVersionFormatting:
    def __init__(self, wslpDataFrame, wslpSheet, wslpElevationDataFrame, wslpPredictions=None, stations=None):
        self.wslpSheet = wslpSheet
        self.wslpDataFrame = wslpDataFrame
        self.wslpElevationDataFrame = wslpElevationDataFrame
        self.wslpPredictions = wslpPredictions
        self.stations = stations

    def formatValuesA(self):
        for value in self.wslpSheet.keys():
            # Get values from every sheet in WSLP-101 using headers
            individualSheet = self.wslpSheet[value].loc[:, ["Depth ", "qc", 'fs', 'u2', 'qt/pa', 'Rf']]

            # Rename columns to be uniform
            #individualSheet = individualSheet.rename(columns={"s 'p": "prec"})

            individualSheetUnits = individualSheet.head(1)
            individualSheet = individualSheet.iloc[1:]
            individualSheet.reset_index(drop=True, inplace=True)

            # Add dataframe to dictionary
            self.wslpDataFrame[value] = individualSheet

            # Get elevation values
            elevation = self.wslpSheet[value].loc[:, ["Elevation"]]
            elevation.head(1)
            elevation = elevation.iloc[1:]
            elevation.reset_index(drop=True, inplace=True)
            self.wslpElevationDataFrame[value] = elevation

            '''''
            # Remove any NaN values before formatting
            modifiedDataFrames = DeleteNans(valuesDataFrame=self.wslpDataFrame[value],
                                            elevationDataFrame=self.wslpElevationDataFrame[value])
            modifiedDataFrames.removenansprec()
            self.wslpDataFrame[value] = modifiedDataFrames.valuesDataFrame

            self.wslpElevationDataFrame[value] = modifiedDataFrames.elevationDataFrame
            '''''

            # Format Numbers
            self.wslpDataFrame[value] = wslpVersionFormatting.convertUnits(self=self, dataSheet=self.wslpDataFrame[value], version="A", units=individualSheetUnits)

            # creating and inserting geology column
            geoValues = np.zeros((len(self.wslpDataFrame[value]), 1))
            self.wslpDataFrame[value].insert(6, "geology", geoValues, True)

            # Remove any Nan values after calculating Rf
            modifiedDataFrames = DeleteNans(valuesDataFrame=self.wslpDataFrame[value],
                                            elevationDataFrame=self.wslpElevationDataFrame[value])
            modifiedDataFrames.removenansrf()
            self.wslpDataFrame[value] = modifiedDataFrames.valuesDataFrame
            self.wslpElevationDataFrame[value] = modifiedDataFrames.elevationDataFrame

        return self.wslpDataFrame, self.wslpElevationDataFrame

    def formatValuesB(self):
        for value in self.wslpSheet.keys():
            # Get values from every sheet in WSLP-101 using headers
            individualSheet = self.wslpSheet[value].loc[:, ["Depth", "q_c", "fs", "Pw", "q_t", "Rf"]]

            # Rename columns to be uniform
            individualSheet = individualSheet.rename(
                columns={"Depth": "Depth ", "q_c": "qc", "Pw": "u2", "q_t": "qt/pa"})

            # extracting units
            individualSheetUnits = individualSheet.head(1)
            individualSheet = individualSheet.iloc[1:]
            individualSheet.reset_index(drop=True, inplace=True)

            # Add dataframe to dictionary
            self.wslpDataFrame[value] = individualSheet

            # Get elevation values
            elevation = self.wslpSheet[value].loc[:, ["Elevation"]]
            elevation.head(1)
            elevation = elevation.iloc[1:]
            elevation.reset_index(drop=True, inplace=True)
            self.wslpElevationDataFrame[value] = elevation

            '''''
            # Remove any NaN values before formatting
            modifiedDataFrames = DeleteNans(valuesDataFrame=self.wslpDataFrame[value],
                                            elevationDataFrame=self.wslpElevationDataFrame[value])
            modifiedDataFrames.removenansprec()
            self.wslpDataFrame[value] = modifiedDataFrames.valuesDataFrame
            self.wslpElevationDataFrame[value] = modifiedDataFrames.elevationDataFrame
            '''''

            # Format Numbers
            self.wslpDataFrame[value] = wslpVersionFormatting.convertUnits(self=self, dataSheet=self.wslpDataFrame[value], version="B", units=individualSheetUnits)

            # creating and inserting geology column
            geoValues = np.zeros((len(self.wslpDataFrame[value]), 1))
            self.wslpDataFrame[value].insert(6, "geology", geoValues, True)

            # Remove any Nan values after calculating Rf
            modifiedDataFrames = DeleteNans(valuesDataFrame=self.wslpDataFrame[value],
                                            elevationDataFrame=self.wslpElevationDataFrame[value])
            modifiedDataFrames.removenansrf()
            self.wslpDataFrame[value] = modifiedDataFrames.valuesDataFrame
            self.wslpElevationDataFrame[value] = modifiedDataFrames.elevationDataFrame

        return self.wslpDataFrame, self.wslpElevationDataFrame

    def formatValuesC(self):
        for value in self.wslpSheet.keys():
            # Get values from every sheet in WSLP-101 using headers
            individualSheet = self.wslpSheet[value].loc[:, ["Depth ", "qc", 'fs', 'u2', 'qt/pa', 'Rf']]

            # Rename columns to be uniform
            #individualSheet = individualSheet.rename(columns={"sp": "prec"})

            # extracting units
            individualSheetUnits = individualSheet.head(1)
            individualSheet = individualSheet.iloc[1:]
            individualSheet.reset_index(drop=True, inplace=True)

            # Add dataframe to dictionary
            self.wslpDataFrame[value] = individualSheet

            # Get elevation values
            elevation = self.wslpSheet[value].loc[:, ["Elevation"]]
            elevation.head(1)
            elevation = elevation.iloc[1:]
            elevation.reset_index(drop=True, inplace=True)
            self.wslpElevationDataFrame[value] = elevation

            '''''
            # Remove any NaN values before formatting
            modifiedDataFrames = DeleteNans(valuesDataFrame=self.wslpDataFrame[value],
                                            elevationDataFrame=self.wslpElevationDataFrame[value])
            modifiedDataFrames.removenansprec()
            self.wslpDataFrame[value] = modifiedDataFrames.valuesDataFrame
            self.wslpElevationDataFrame[value] = modifiedDataFrames.elevationDataFrame
            '''''

            # Format Numbers
            self.wslpDataFrame[value] = wslpVersionFormatting.convertUnits(self=self, dataSheet=self.wslpDataFrame[value], version="C", units=individualSheetUnits)

            # creating and inserting geology column
            geoValues = np.zeros((len(self.wslpDataFrame[value]), 1))
            self.wslpDataFrame[value].insert(6, "geology", geoValues, True)

            # Remove any Nan values after calculating Rf
            modifiedDataFrames = DeleteNans(valuesDataFrame=self.wslpDataFrame[value],
                                            elevationDataFrame=self.wslpElevationDataFrame[value])
            modifiedDataFrames.removenansrf()
            self.wslpDataFrame[value] = modifiedDataFrames.valuesDataFrame
            self.wslpElevationDataFrame[value] = modifiedDataFrames.elevationDataFrame

        return self.wslpDataFrame, self.wslpElevationDataFrame

    def formatValuesMau3(self):
        for value in self.wslpSheet.keys():
            # Get values from every sheet in WSLP-101 using headers
            individualSheet = self.wslpSheet[value].loc[:, ["Depth .1", "qc.1", 'fs.1', 'u2.1', 'qt/pa', 'Rf']]

            # Rename columns to be uniform
            individualSheet = individualSheet.rename(columns={"Depth .1" : "Depth ", "qc.1": "qc", "fs.1": "fs", "u2.1": "u2"})

            individualSheetUnits = individualSheet.head(1)
            individualSheet = individualSheet.iloc[1:]
            individualSheet.reset_index(drop=True, inplace=True)

            # Add dataframe to dictionary
            self.wslpDataFrame[value] = individualSheet

            # Get elevation values
            elevation = self.wslpSheet[value].loc[:, ["Elevation"]]
            elevation.head(1)
            elevation = elevation.iloc[1:]
            elevation.reset_index(drop=True, inplace=True)
            self.wslpElevationDataFrame[value] = elevation

            '''''
            # Remove any NaN values before formatting
            modifiedDataFrames = DeleteNans(valuesDataFrame=self.wslpDataFrame[value],
                                            elevationDataFrame=self.wslpElevationDataFrame[value])
            modifiedDataFrames.removenansprec()
            

            self.wslpDataFrame[value] = modifiedDataFrames.valuesDataFrame

            self.wslpElevationDataFrame[value] = modifiedDataFrames.elevationDataFrame
            '''''
            # Format Numbers
            self.wslpDataFrame[value] = wslpVersionFormatting.convertUnits(self=self, dataSheet=self.wslpDataFrame[value], version="A", units=individualSheetUnits)

            # creating and inserting geology column
            geoValues = np.zeros((len(self.wslpDataFrame[value]), 1))
            self.wslpDataFrame[value].insert(6, "geology", geoValues, True)

            # Remove any Nan values after calculating Rf
            modifiedDataFrames = DeleteNans(valuesDataFrame=self.wslpDataFrame[value],
                                            elevationDataFrame=self.wslpElevationDataFrame[value])
            modifiedDataFrames.removenansrf()
            modifiedDataFrames.removenansqtpa()
            self.wslpDataFrame[value] = modifiedDataFrames.valuesDataFrame
            self.wslpElevationDataFrame[value] = modifiedDataFrames.elevationDataFrame

        return self.wslpDataFrame, self.wslpElevationDataFrame

    def convertUnits(self, dataSheet, version, units):
        if version == "A":
            dataSheet = dataSheet.astype(float)
            for index, value in enumerate(units.iloc[0].values):
                if value == "tsf":
                    dataSheet.iloc[:, [index]] *= 2000
                elif value == "psi":
                    dataSheet.iloc[:, [index]] *= 144

            dataSheet['qt/pa'] = np.log10(dataSheet['qt/pa'])
            dataSheet['Rf'] = np.log10(dataSheet['Rf'])

        elif version == "B":
            dataSheet = dataSheet.astype(float)
            for index, value in enumerate(units.iloc[0].values):
                if value == "(tsf)" and index != 4:
                    dataSheet.iloc[:, [index]] *= 2000

                elif value == "(psi)":
                    dataSheet.iloc[:, [index]] *= 144
            dataSheet.loc[:, ['qt/pa']] /= 1.0581
            dataSheet['qt/pa'] = np.log10(dataSheet['qt/pa'])
            dataSheet['Rf'] = np.log10(dataSheet['Rf'])

        elif version == "C":
            dataSheet = dataSheet.astype(float)
            for index, value in enumerate(units.iloc[0].values):
                if value == "tsf":
                    dataSheet.iloc[:, [index]] *= 2000

            dataSheet['qt/pa'] = np.log10(dataSheet['qt/pa'])
            dataSheet['Rf'] = np.log10(dataSheet['Rf'])

        else:
            print("wrong version input")
        return dataSheet


    def output(self):
        for value in self.wslpSheet:
            self.wslpPredictions[value] = Mdl.predict(self.wslpSheet[value])
            formatOutput = FormatOutput(cpt="", sheetName=value, elevations=self.wslpElevationDataFrame[value], predictions=self.wslpPredictions[value], stations=self.stations)
            cpt = formatOutput.findcpt()
            output = formatOutput.combinecolumns()
            self.wslpDataFrame[value] = output
            deleteNans = DeleteNans(output=self.wslpDataFrame[value])
            self.wslpDataFrame[value] = deleteNans.removenansy()

data = pd.read_excel('wslp_f29.xlsx', sheet_name=0)


X = data.loc[:, ['Depth ', 'qc', 'fs', 'u2', 'qt/pa', 'Rf', 'geology']]  # 8 feature columns
Y = data.loc[:, ['organic']]  # Output column
m = len(Y)

# Randomizing data
np.set_printoptions(suppress=True, precision=15)
np.random.seed(0)

p = .8
idx = np.random.permutation(m)
xtr = X.loc[idx[1:round(p * m)]]
ytr = Y.loc[idx[1:round(p * m)]]
xte = X.loc[idx[round(p * m) + 1:len(idx) - 1]]
yte = Y.loc[idx[round(p * m) + 1:len(idx) - 1]]

# Create a random forest model
Mdl = sklearn.ensemble.RandomForestClassifier()
Mdl.fit(xtr, np.ravel(ytr))

hte = Mdl.predict(xte)
accuracy_te = accuracy_score(yte, hte)
print("accuracy_te:", accuracy_te)
confusionMatrix1 = confusion_matrix(yte, hte)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix1).plot()

htr = Mdl.predict(xtr)
accuracy_tr = accuracy_score(ytr, htr)
print("accuracy_tr:", accuracy_tr)
confusionMatrix2 = confusion_matrix(ytr, htr)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix2).plot()

htAll = Mdl.predict(X)
accuracy_All = accuracy_score(Y, htAll)
print("accuracy_All:", accuracy_All)
confusionMatrix3 = confusion_matrix(Y, htAll)
ConfusionMatrixDisplay(confusion_matrix=confusionMatrix3).plot()
# plt.show()

# Read file
wslp101 = pd.read_excel('A-WSLP-101.xlsx', header=8, sheet_name=None, skiprows=[10, 11])


# Create dictionaries
wslp101DataFrames = {}
wslp101ElevationDataFrames = {}

# Format file based on type
formattedWslpsA = wslpVersionFormatting(wslpSheet=wslp101, wslpDataFrame=wslp101DataFrames, wslpElevationDataFrame=wslp101ElevationDataFrames)
formattedWslpsA.formatValuesA()


# Read file
wslp102 = pd.read_excel('B-WSLP-102.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslp102DataFrames = {}
wslp102ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslp102, wslpDataFrame=wslp102DataFrames, wslpElevationDataFrame=wslp102ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslp103 = pd.read_excel('B-WSLP-103.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslp103DataFrames = {}
wslp103ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslp103, wslpDataFrame=wslp103DataFrames, wslpElevationDataFrame=wslp103ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslp104 = pd.read_excel('B-WSLP-104.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslp104DataFrames = {}
wslp104ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslp104, wslpDataFrame=wslp104DataFrames, wslpElevationDataFrame=wslp104ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslp105 = pd.read_excel('B-WSLP-105.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslp105DataFrames = {}
wslp105ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslp105, wslpDataFrame=wslp105DataFrames, wslpElevationDataFrame=wslp105ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslp106 = pd.read_excel('B-WSLP-106.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslp106DataFrames = {}
wslp106ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslp106, wslpDataFrame=wslp106DataFrames, wslpElevationDataFrame=wslp106ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslp107 = pd.read_excel('B-WSLP-107.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslp107DataFrames = {}
wslp107ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslp107, wslpDataFrame=wslp107DataFrames, wslpElevationDataFrame=wslp107ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslp108 = pd.read_excel('B-WSLP-108.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslp108DataFrames = {}
wslp108ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslp108, wslpDataFrame=wslp108DataFrames, wslpElevationDataFrame=wslp108ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslp109 = pd.read_excel('C-WSLP-109.xlsx', header=8, sheet_name=None, skiprows=[10, 11])

# Create dictionaries
wslp109DataFrames = {}
wslp109ElevationDataFrames = {}

# Format file based on type
formattedWslpsC = wslpVersionFormatting(wslpSheet=wslp109, wslpDataFrame=wslp109DataFrames, wslpElevationDataFrame=wslp109ElevationDataFrames)
formattedWslpsC.formatValuesC()

# Read file
wslp110 = pd.read_excel('B-WSLP-110.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslp110DataFrames = {}
wslp110ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslp110, wslpDataFrame=wslp110DataFrames, wslpElevationDataFrame=wslp110ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslpMau1 = pd.read_excel('B-WSLP-Mau1.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslpMau1DataFrames = {}
wslpMau1ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslpMau1, wslpDataFrame=wslpMau1DataFrames, wslpElevationDataFrame=wslpMau1ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslpMau2 = pd.read_excel('B-WSLP-Mau2.xlsx', header=10, sheet_name=None, skiprows=[12])

# Create dictionaries
wslpMau2DataFrames = {}
wslpMau2ElevationDataFrames = {}

# Format file based on type
formattedWslpsB = wslpVersionFormatting(wslpSheet=wslpMau2, wslpDataFrame=wslpMau2DataFrames, wslpElevationDataFrame=wslpMau2ElevationDataFrames)
formattedWslpsB.formatValuesB()

# Read file
wslpMau3 = pd.read_excel('D-WSLP-Mau3.xlsx', header=8, sheet_name=None, skiprows=[10, 11])

# Create dictionaries
wslpMau3DataFrames = {}
wslpMau3ElevationDataFrames = {}

# Format file based on type
formattedWslpsMau3 = wslpVersionFormatting(wslpSheet=wslpMau3, wslpDataFrame=wslpMau3DataFrames, wslpElevationDataFrame=wslpMau3ElevationDataFrames)
formattedWslpsMau3.formatValuesMau3()


# Find length of all data sets
dataLengthWslp101 = len(wslp101DataFrames[list(wslp101DataFrames.keys())[0]])
dataLengthWslp102 = len(wslp102DataFrames[list(wslp102DataFrames.keys())[0]])
dataLengthWslp103 = len(wslp103DataFrames[list(wslp103DataFrames.keys())[0]])
dataLengthWslp104 = len(wslp104DataFrames[list(wslp104DataFrames.keys())[0]])
dataLengthWslp105 = len(wslp105DataFrames[list(wslp105DataFrames.keys())[0]])
dataLengthWslp106 = len(wslp106DataFrames[list(wslp106DataFrames.keys())[0]])
dataLengthWslp107 = len(wslp107DataFrames[list(wslp107DataFrames.keys())[0]])
dataLengthWslp108 = len(wslp108DataFrames[list(wslp108DataFrames.keys())[0]])
dataLengthWslp109 = len(wslp109DataFrames[list(wslp109DataFrames.keys())[0]])
dataLengthWslp110 = len(wslp110DataFrames[list(wslp110DataFrames.keys())[0]])
dataLengthWslpMau1 = len(wslpMau1DataFrames[list(wslpMau1DataFrames.keys())[0]])
dataLengthWslpMau2 = len(wslpMau2DataFrames[list(wslpMau2DataFrames.keys())[0]])
dataLengthWslpMau3 = len(wslpMau3DataFrames[list(wslpMau3DataFrames.keys())[0]])

# Read station file
stations = pd.read_excel('WSLP_stations.xlsx', sheet_name=0)
stations = stations.loc[:, ["Stations", "CPT"]]

# Create dictionaries
wslp101PredictionDataFrames = {}
wslp101OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp101 = wslpVersionFormatting(wslpSheet=wslp101DataFrames, wslpDataFrame=wslp101OutputDataFrames, wslpPredictions=wslp101PredictionDataFrames, wslpElevationDataFrame=wslp101ElevationDataFrames, stations=stations)
formatOutputWslp101.output()

# Create dictionaries
wslp102PredictionDataFrames = {}
wslp102OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp102 = wslpVersionFormatting(wslpSheet=wslp102DataFrames, wslpDataFrame=wslp102OutputDataFrames, wslpPredictions=wslp102PredictionDataFrames, wslpElevationDataFrame=wslp102ElevationDataFrames, stations=stations)
formatOutputWslp102.output()

# Create dictionaries
wslp103PredictionDataFrames = {}
wslp103OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp103 = wslpVersionFormatting(wslpSheet=wslp103DataFrames, wslpDataFrame=wslp103OutputDataFrames, wslpPredictions=wslp103PredictionDataFrames, wslpElevationDataFrame=wslp103ElevationDataFrames, stations=stations)
formatOutputWslp103.output()

# Create dictionaries
wslp104PredictionDataFrames = {}
wslp104OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp104 = wslpVersionFormatting(wslpSheet=wslp104DataFrames, wslpDataFrame=wslp104OutputDataFrames, wslpPredictions=wslp104PredictionDataFrames, wslpElevationDataFrame=wslp104ElevationDataFrames, stations=stations)
formatOutputWslp104.output()

# Create dictionaries
wslp105PredictionDataFrames = {}
wslp105OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp105 = wslpVersionFormatting(wslpSheet=wslp105DataFrames, wslpDataFrame=wslp105OutputDataFrames, wslpPredictions=wslp105PredictionDataFrames, wslpElevationDataFrame=wslp105ElevationDataFrames, stations=stations)
formatOutputWslp105.output()

# Create dictionaries
wslp106PredictionDataFrames = {}
wslp106OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp106 = wslpVersionFormatting(wslpSheet=wslp106DataFrames, wslpDataFrame=wslp106OutputDataFrames, wslpPredictions=wslp106PredictionDataFrames, wslpElevationDataFrame=wslp106ElevationDataFrames, stations=stations)
formatOutputWslp106.output()

# Create dictionaries
wslp107PredictionDataFrames = {}
wslp107OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp107 = wslpVersionFormatting(wslpSheet=wslp107DataFrames, wslpDataFrame=wslp107OutputDataFrames, wslpPredictions=wslp107PredictionDataFrames, wslpElevationDataFrame=wslp107ElevationDataFrames, stations=stations)
formatOutputWslp107.output()

# Create dictionaries
wslp108PredictionDataFrames = {}
wslp108OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp108 = wslpVersionFormatting(wslpSheet=wslp108DataFrames, wslpDataFrame=wslp108OutputDataFrames, wslpPredictions=wslp108PredictionDataFrames, wslpElevationDataFrame=wslp108ElevationDataFrames, stations=stations)
formatOutputWslp108.output()

# Create dictionaries
wslp109PredictionDataFrames = {}
wslp109OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp109 = wslpVersionFormatting(wslpSheet=wslp109DataFrames, wslpDataFrame=wslp109OutputDataFrames, wslpPredictions=wslp109PredictionDataFrames, wslpElevationDataFrame=wslp109ElevationDataFrames, stations=stations)
formatOutputWslp109.output()

# Create dictionaries
wslp110PredictionDataFrames = {}
wslp110OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslp110 = wslpVersionFormatting(wslpSheet=wslp110DataFrames, wslpDataFrame=wslp110OutputDataFrames, wslpPredictions=wslp110PredictionDataFrames, wslpElevationDataFrame=wslp110ElevationDataFrames, stations=stations)
formatOutputWslp110.output()

# Create dictionaries
wslpMau1PredictionDataFrames = {}
wslpMau1OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslpMau1 = wslpVersionFormatting(wslpSheet=wslpMau1DataFrames, wslpDataFrame=wslpMau1OutputDataFrames, wslpPredictions=wslpMau1PredictionDataFrames, wslpElevationDataFrame=wslpMau1ElevationDataFrames, stations=stations)
formatOutputWslpMau1.output()

# Create dictionaries
wslpMau2PredictionDataFrames = {}
wslpMau2OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslpMau2 = wslpVersionFormatting(wslpSheet=wslpMau2DataFrames, wslpDataFrame=wslpMau2OutputDataFrames, wslpPredictions=wslpMau2PredictionDataFrames, wslpElevationDataFrame=wslpMau2ElevationDataFrames, stations=stations)
formatOutputWslpMau2.output()

# Create dictionaries
wslpMau3PredictionDataFrames = {}
wslpMau3OutputDataFrames = {}

# Format output by finding CPT and generating 3 columns
formatOutputWslpMau3 = wslpVersionFormatting(wslpSheet=wslpMau3DataFrames, wslpDataFrame=wslpMau3OutputDataFrames, wslpPredictions=wslpMau3PredictionDataFrames, wslpElevationDataFrame=wslpMau3ElevationDataFrames, stations=stations)
formatOutputWslpMau3.output()

# Output to Excel file
# Using current to indicate that writer should start after data already printed
with pd.ExcelWriter('outputData.xlsx') as writer:
    current = 0
    for wslp101Sheet in wslp101OutputDataFrames:
        wslp101OutputDataFrames[wslp101Sheet].to_excel(writer, sheet_name='WSLP-101', index=False, startrow=current,
                                                       header=False)
        current += len(wslp101OutputDataFrames[wslp101Sheet])
    current = 0
    for wslp102Sheet in wslp102OutputDataFrames:
        wslp102OutputDataFrames[wslp102Sheet].to_excel(writer, sheet_name='WSLP-102', index=False, startrow=current,
                                                       header=False)
        current += len(wslp102OutputDataFrames[wslp102Sheet])
    current = 0
    for wslp103Sheet in wslp103OutputDataFrames:
        wslp103OutputDataFrames[wslp103Sheet].to_excel(writer, sheet_name='WSLP-103', index=False, startrow=current,
                                                       header=False)
        current += len(wslp103OutputDataFrames[wslp103Sheet])
    current = 0
    for wslp104Sheet in wslp104OutputDataFrames:
        wslp104OutputDataFrames[wslp104Sheet].to_excel(writer, sheet_name='WSLP-104', index=False, startrow=current,
                                                       header=False)
        current += len(wslp104OutputDataFrames[wslp104Sheet])
    current = 0
    for wslp105Sheet in wslp105OutputDataFrames:
        wslp105OutputDataFrames[wslp105Sheet].to_excel(writer, sheet_name='WSLP-105', index=False, startrow=current,
                                                       header=False)
        current += len(wslp105OutputDataFrames[wslp105Sheet])
    current = 0
    for wslp106Sheet in wslp106OutputDataFrames:
        wslp106OutputDataFrames[wslp106Sheet].to_excel(writer, sheet_name='WSLP-106', index=False, startrow=current,
                                                       header=False)
        current += len(wslp106OutputDataFrames[wslp106Sheet])
    current = 0
    for wslp107Sheet in wslp107OutputDataFrames:
        wslp107OutputDataFrames[wslp107Sheet].to_excel(writer, sheet_name='WSLP-107', index=False, startrow=current,
                                                       header=False)
        current += len(wslp107OutputDataFrames[wslp107Sheet])
    current = 0
    for wslp108Sheet in wslp108OutputDataFrames:
        wslp108OutputDataFrames[wslp108Sheet].to_excel(writer, sheet_name='WSLP-108', index=False, startrow=current,
                                                       header=False)
        current += len(wslp108OutputDataFrames[wslp108Sheet])
    current = 0
    for wslp109Sheet in wslp109OutputDataFrames:
        wslp109OutputDataFrames[wslp109Sheet].to_excel(writer, sheet_name='WSLP-109', index=False, startrow=current,
                                                       header=False)
        current += len(wslp109OutputDataFrames[wslp109Sheet])
    current = 0
    for wslp110Sheet in wslp110OutputDataFrames:
        wslp110OutputDataFrames[wslp110Sheet].to_excel(writer, sheet_name='WSLP-110', index=False, startrow=current,
                                                       header=False)
        current += len(wslp110OutputDataFrames[wslp110Sheet])
    current = 0
    for wslpMau1Sheet in wslpMau1OutputDataFrames:
        wslpMau1OutputDataFrames[wslpMau1Sheet].to_excel(writer, sheet_name='WSLP-Mau1', index=False, startrow=current,
                                                       header=False)
        current += len(wslpMau1OutputDataFrames[wslpMau1Sheet])
    current = 0
    for wslpMau2Sheet in wslpMau2OutputDataFrames:
        wslpMau2OutputDataFrames[wslpMau2Sheet].to_excel(writer, sheet_name='WSLP-Mau2', index=False, startrow=current,
                                                       header=False)
        current += len(wslpMau2OutputDataFrames[wslpMau2Sheet])
    current = 0
    for wslpMau3Sheet in wslpMau3OutputDataFrames:
        wslpMau3OutputDataFrames[wslpMau3Sheet].to_excel(writer, sheet_name='WSLP-Mau3', index=False, startrow=current,
                                                       header=False)
        current += len(wslpMau3OutputDataFrames[wslpMau3Sheet])
    current = 0


