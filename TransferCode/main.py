import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble

data = pd.read_excel('wslp_f29.xlsx', sheet_name = 0)


X = data.loc[: , ['Depth(ft)','qc(psf)','fs(psf)','u2(psf)','log(qt/Pa)','log(Rf)','geology']] # 8 feature columns
Y = data.loc[: , ['prec']] # Output column
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
print(xtr)
print(len(xtr))
ytr = Y.loc[idx[1:round(p*m)]]
print(ytr)
print(len(ytr))
xte = X.loc[idx[round(p*m)+1:len(idx)-1]]
print(xte)
print(len(xte))
yte = Y.loc[idx[round(p*m)+1:len(idx)-1]]
print(ytr)
print(len(yte))


# Create a random forest model
Mdl = sklearn.ensemble.RandomForestClassifier(100,xtr,ytr,'Method','classification','minleafsize',7,'OOBPred','On','OOBPredictorImportance','on')

hte = Mdl.predict(xte)
hte = float(hte)
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
"""""