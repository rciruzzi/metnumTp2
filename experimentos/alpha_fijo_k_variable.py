import metnum
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import graficar
import pickle
import numpy as np

df_train = pd.read_csv("../data/train.csv")

X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)

limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

alpha = 32

print("Aplico PCA")
pca = metnum.PCA(alpha)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_val = pca.transform(X_val)
print("Terminé PCA")

ks = list(range(1, 31, 1)) + list(range(40, 101, 10))

y_preds = []
accurancies = []
times = []

for k in ks:
    print("Aplico KNN para k: %d"% (k))
    
    start_time = time.time()
    clf = metnum.KNNClassifier(k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    times.append(time.time() - start_time)
    y_preds.append(y_pred)
    acc = accuracy_score(y_val, y_pred)
    accurancies.append(acc)
    print("Accuracy: {}".format(acc))
    print("------------------------------------------------------------")


graficar.graficarPuntos(accurancies, ks, 'accuracy', 'k', 'Alpha fijo con k variable')
graficar.graficarPuntos(times, ks, 'Tiempo de ejecución [s]', 'k', 'Alpha fijo con k variable - Complejidad')
graficar.graficarHeatmap(y_preds[np.argmax(accurancies)].reshape(y_val.shape[0]), y_val.reshape(y_val.shape[0]))
