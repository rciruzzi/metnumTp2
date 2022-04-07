import metnum
import pandas as pd
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import graficar
import pickle
import time

df_train = pd.read_csv("../data/train.csv")

df_train = df_train[:8000]

X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)

limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

ks = list(range(1, 16, 1)) + list(range(25, 106, 10))

# accurancies = []
preds = []

times = []

for k in ks:

    print("Aplico KNN para k: %d"% (k))
    start_time = time.time()
    clf = metnum.KNNClassifier(k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_pred = y_pred.reshape(y_pred.shape[0])
    preds.append(y_pred)
    times.append(time.time() - start_time)
    # acc = accuracy_score(y_val, y_pred)
    # accurancies.append(acc)
    # print("Accuracy: {}".format(acc))
    # print("------------------------------------------------------------")

with open("knn_sin_pca.pkl", 'wb') as f:
    pickle.dump([times,preds], f)

with open("knn_sin_pca.pkl", 'rb') as f:
    print(pickle.load(f))

#graficar.graficarHeatmap(preds[0], y_val.reshape(y_val.shape[0]))
