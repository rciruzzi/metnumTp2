import numpy as np

def split(X, y, k, accuracy, alpha, kNeighbor):
    blockSize = int(X.shape[0] / k)
    acc = 0
    for i in range(k):

        limite_1 = blockSize*i
        limite_2 = min(X.shape[0], blockSize*(i+1))
        """ print("Rangos train: 0-%d y %d-%d"% (limite_1, limite_2, X.shape[0]))
        print("Rangos validation: %d-%d"% (limite_1, limite_2)) """

        X_train, y_train = np.concatenate((X[0:limite_1], X[limite_2:])), np.concatenate((y[0:limite_1], y[limite_2:]))
        X_val, y_val = X[limite_1:limite_2], y[limite_1:limite_2]

        """ aux = accuracy(X_train, y_train, X_val, y_val, kNeighbor, alpha)
        print("Accuracy: %f"% (aux))
        acc += aux """
        acc += accuracy(X_train, y_train, X_val, y_val, alpha, kNeighbor)
    return acc / k