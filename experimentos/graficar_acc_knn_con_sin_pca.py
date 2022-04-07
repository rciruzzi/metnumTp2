import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

df_train = pd.read_csv("../data/train.csv")

df_train = df_train[:8000]

X = df_train[df_train.columns[1:]].values
y = df_train["label"].values.reshape(-1, 1)

limit = int(0.8 * X.shape[0]) 

X_train, y_train = X[:limit], y[:limit]
X_val, y_val = X[limit:], y[limit:]

ks = list(range(1, 16, 1)) + list(range(25, 106, 10))

with open("knn_sin_pca.pkl", 'rb') as f:
        knnSinPca = pickle.load(f)
        with open("knn_con_pca.pkl", 'rb') as f2:
            knnConPca = pickle.load(f2)

            accKNNSinPCA = []
            for pred in knnSinPca[1]:
                accKNNSinPCA.append(accuracy_score(y_val, pred))

            accKNNConPCA = []
            for pred in knnConPca[1]:
                accKNNConPCA.append(accuracy_score(y_val, pred))
            
            plt.figure(figsize=(14,16), dpi= 80)
            plt.plot(ks, accKNNSinPCA, label="Sin PCA")
            plt.plot(ks, accKNNConPCA, label="Con PCA")
            plt.title("Accuracy para KNN con k variable", fontdict={'size':20})
            plt.xlabel("k")
            plt.ylabel("Accuracy")
            plt.grid(linestyle='--', alpha=0.5)
            plt.legend()
            plt.show()