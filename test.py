import metnum

import pandas as pd
from sklearn.metrics import accuracy_score

df_train = pd.read_csv("./data/train.csv")

df_train = df_train.sample(frac=.1)


X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)

limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)

print(f"Ahora tengo {len(X_train)} instancias de entrenamiento y {len(X_val)} de validación")

# Acá pueden cambiar el clasificador nuestro por el de sklearn!
print("Aplico PCA")
pca = metnum.PCA(20)
pca.fit(X_train)

X_val = pca.transform(X_val)
print("Termine PCA")

clf = metnum.KNNClassifier(1000)

X_train = pca.transform(X_train)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

acc = accuracy_score(y_val, y_pred)
print("Accuracy: {}".format(acc))


print("Es un: {}".format(y_val[0]))
print("Dije que es un: {}".format(y_pred[0]))

""" y_pred = clf.predict(X_val)
print("X_val len: ", len(X_val))
print("Prediccion len: ", y_pred)

print("todavia no me destrui")

print("No, todavia no..")
 """

