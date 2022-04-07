import utils
import kfold
import graficar

X, y = utils.loadData(frac=.1)

alpha = 35
kneighbor = 7

accurancies = []

ks = range(2,11, 1)

for k in ks:
    acc = kfold.split(X, y, k, utils.accuracyKNNWithPCA, alpha, kneighbor)
    print("K: %2d   Accuracy: %f"% (k, acc))
    accurancies.append(acc)

graficar.graficarPuntos(accurancies, ks, 'Accuracy', 'K', 'K-fold con K variable')