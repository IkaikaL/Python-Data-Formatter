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

data = pd.read_excel('wslp_f29.xlsx', sheet_name = 0)


X = data.loc[: , ['Depth(ft)','qc(psf)','fs(psf)','u2(psf)','log(qt/Pa)','log(Rf)','organic']] # 8 feature columns
Y = data.loc[: , ['organic']] # Output column
m = len(Y)

"""""
print(X)
print(Y)
print(m)
"""""

PredictorNames = ["depth","qc","fs","u2","log(qt/Pa)","log(Rf)","geology","preconsolidation"]


# Randomizing data
p = .8;

idx = np.random.permutation(m)

xtr = X.loc[idx[1:round(p*m)]]

ytr = Y.loc[idx[1:round(p*m)]]

xte = X.loc[idx[round(p*m)+1:len(idx)-1]]

yte = Y.loc[idx[round(p*m)+1:len(idx)-1]]

"""""
print(m)
print(idx)
print(xtr)
print(len(xtr))
print(ytr)
print(len(ytr))
print(xte)
print(len(xte))
print(ytr)
print(len(yte))
"""""

# Create a random forest model
Mdl = sklearn.ensemble.RandomForestClassifier()

Mdl.fit(xtr, np.ravel(ytr))

hte = Mdl.predict(xte)
print(hte)

yPredict = Mdl.predict(xte)
print(yPredict)

accuracy = accuracy_score(yte, yPredict)
print(accuracy)

confusionMatrix = confusion_matrix(yte, yPredict)
ConfusionMatrixDisplay(confusion_matrix = confusionMatrix).plot()

"""""
#hte = float(hte)
accuracy_te = statistics.mean(round(hte) == yte)
confusion_matrix_te = sklearn.metrics.confusion_matrix(yte, round(hte))
cm_display_te = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_te, display_labels = [False, True])
cm_display_te.plot()
plt.show()

htr = Mdl.predict(xtr)
htr = float(htr)
accuracy_tr = statistics.mean(round(htr) == ytr)
confusion_matrix_tr = sklearn.metrics.confusion_matrix(ytr, round(htr))
cm_display_tr = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_tr, display_labels = [False, True])
cm_display_tr.plot()
plt.show()

htall = Mdl.predict(X)
htall = float(htall)
accuracy_all = statistics.mean(round(htall) == Y)
confusion_matrix_all = sklearn.metrics.confusion_matrix(ytr, round(htr))
cm_display_all = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_all, display_labels = [False, True])
cm_display_all.plot()
plt.show()
"""""