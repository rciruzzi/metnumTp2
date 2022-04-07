import pickle
from matplotlib import pyplot as plt

ks = list(range(1, 16, 1)) + list(range(25, 106, 10))

with open("knn_sin_pca.pkl", 'rb') as f:
        knnSinPca = pickle.load(f)
        with open("knn_con_pca.pkl", 'rb') as f2:
            knnConPca = pickle.load(f2)
            
            plt.figure(figsize=(14,16), dpi= 80)
            plt.plot(ks, knnSinPca[0], label="Sin PCA")
            plt.plot(ks, knnConPca[0], label="Con PCA")
            plt.title("Tiempos para KNN con k variable", fontdict={'size':20})
            plt.xlabel("k")
            plt.ylabel("Tiempo en segundos")
            plt.grid(linestyle='--', alpha=0.5)
            plt.legend()
            plt.show()