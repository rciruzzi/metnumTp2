import utils
import kfold
import pickle

X, y = utils.loadData()

alphas = range(30, 41)
ks = range(7, 16)

#accurancies = []

""" acc = kfold.split(X, y, 5, accuracyKNNWithPCA, 31, 8)
print(acc) """
data = []

maxK = 0
maxAlpha = 0
maxAccuracy = 0

for alpha in alphas:
    for k in ks:
        acc = kfold.split(X, y, 5, utils.accuracyKNNWithPCA, alpha, k)
        print("alpha: %2d   k: %2d   Accuracy: %f"% (alpha, k, acc))
        #accurancies.append(acc)
        data.append([alpha, k, acc])
        if acc > maxAccuracy:
            maxAccuracy = acc
            maxAlpha = alpha
            maxK = k

with open("k_alpha_variables.pkl", 'wb') as f:
    pickle.dump(data, f)

print("-----------------Maximo-------------------")
print("alpha: %2d   k: %2d   Accuracy: %f"% (maxAlpha, maxK, maxAccuracy))