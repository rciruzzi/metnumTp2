import metnum

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

df_train = pd.read_csv("./data/train.csv")

df_train = df_train.sample(frac=.1)


X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)


""" num = 19

print(f"Supuestamente es un {int(y[num])}")

img = X[num].reshape(28, 28)
plt.imshow(img, cmap="Greys")
plt.show() """

limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[limit:], y[:limit]
X_val, y_val = X[limit:], y[limit:]

print("Tamaño original: ")
print(X_val.shape[0])
print(X_val.shape[1])

pca = metnum.PCA(10)
pca.fit(X_train)

print("fit completado")
X_transf = pca.transform(X_val)

# img = X_transf[:,0].reshape(28, 28)
# plt.imshow(img, cmap="Greys")
# plt.show()

# img = X_transf[:,1].reshape(28, 28)
# plt.imshow(img, cmap="Greys")
# plt.show()

# img = X_transf[:,2].reshape(28, 28)
# plt.imshow(img, cmap="Greys")
# plt.show()

""" print("Tamaño: ")
print(X_transf.shape[0])
print(X_transf.shape[1])

cont = 0
for i in range(X_transf.shape[0]):
    for j in range(X_transf.shape[1]):
        if X_transf[i][j] != 0:
            cont += 1
            print(X_transf[i][j])

print(cont * 100/(X_transf.shape[0] * X_transf.shape[1])) """

""" 200*784 x 784*100

200*100 """
""" print(np.allclose(X_transf, X_transf.T)) """

""" img = X_transf[:,0].reshape(28, 28)
print(img)
plt.imshow(img, cmap="Greys") """



