import metnum
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

df_train = pd.read_csv("./data/train.csv")

#df_train = df_train[:500]

X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)

clf = metnum.KNNClassifier(100)

accurancys = []

ks = list(range(2, 11))
for k in ks:
    acc = 0
    blockSize = int(X.shape[0] / k)
    print("Ejecutando el  k-Fold %d"% (k))
    #print("Tamaño del bloque: %d"% (blockSize))
    for i in range(k):

        limite_1 = blockSize*i
        limite_2 = min(X.shape[0], blockSize*(i+1))
        print("Rangos train: 0-%d y %d-%d"% (limite_1, limite_2, X.shape[0]))
        print("Rangos validation: %d-%d"% (limite_1, limite_2))

        X_train, y_train = np.concatenate((X[0:limite_1], X[limite_2:])), np.concatenate((y[0:limite_1], y[limite_2:]))
        X_val, y_val = X[limite_1:limite_2], y[limite_1:limite_2]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        """ print("Tamaños a comparar: %d-%d"% (y_val.shape[0], y_pred.shape[0])) """
        acc += accuracy_score(y_val, y_pred)
        print("Tamaños a comparar: %d-%d"% (y_val.shape[0], y_pred.shape[0]))
    accurancys.append( acc / k)
    print("-----------------------------------------------------------------------------")

plt.xticks(range(len(accurancys)), ks)
plt.xlabel('Ks')
plt.ylabel('Accuracy')
plt.title('K-fold')
plt.ylim((0,1))
plt.bar(range(len(accurancys)), accurancys) 
plt.show()

