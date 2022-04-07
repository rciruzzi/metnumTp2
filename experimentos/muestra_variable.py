import utils
import graficar
import numpy as np

alpha = 35
k = 7
fs = np.arange(0.1, 1.1, 0.1)

accurancies = []

for f in fs:
    print("Frac: %f" % (f))
    X, y = utils.loadData(frac=f)
    X_train, y_train, X_val, y_val = utils.splitData(X, y)
    acc = utils.accuracyKNNWithPCA(X_train, y_train, X_val, y_val, alpha, k)
    print("Accuracy: %f" % (acc))
    accurancies.append(acc)

graficar.graficarPuntos(accurancies, fs, 'Accurancy', 'Porcentaje de la muestra', 'Calidad de resultados con distintos tamaños de muestra')