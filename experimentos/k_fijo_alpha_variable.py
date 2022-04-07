import metnum
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import graficar
import pickle

df_train = pd.read_csv("../data/train.csv")


X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)

limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

k = 8
alphas = list(range(1, 41, 1)) + list(range(50, 151, 10))

accurancies = []
y_preds = []

print("Aplico PCA para alpha: %d"% (np.max(alphas)))
pca = metnum.PCA(np.max(alphas))
pca.fit(X_train)
X_train_transf = pca.transform(X_train)
X_val_transf = pca.transform(X_val)
print("Terminé PCA")

for alpha in alphas:
    print("Aplico KNN para alpha: %d"% alpha)
    clf = metnum.KNNClassifier(k)
    clf.fit(X_train_transf[:,:alpha], y_train)
    y_pred = clf.predict(X_val_transf[:,:alpha])

    y_preds.append(y_pred)
    acc = accuracy_score(y_val, y_pred)
    accurancies.append(acc)
    print("Accuracy: {}".format(acc))
    print("------------------------------------------------------------")


graficar.graficarPuntos(accurancies, alphas, 'accuracy', 'alpha', 'Alpha variable con k fijo')
graficar.graficarHeatmap(y_preds[np.argmax(accurancies)].reshape(y_val.shape[0]), y_val.reshape(y_val.shape[0]))
