from matplotlib import pyplot as plt
import pickle

with open("k_alpha_variables.pkl", 'rb') as f:
    data = pickle.load(f)

    ks = range(7, 16)
    accs = {}
    for a in range(30, 41):
        accs[a] = []

    for item in data:
        accs[item[0]].append(item[2])

    plt.figure(figsize=(14,16), dpi= 80)
    for key, value in accs.items():
        plt.scatter(ks, value, s=30, alpha=1, label=key)
    plt.title("Accuracy con k y \u03B1 variables y usando K-Fold", fontdict={'size':20})
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(title="\u03B1")
    plt.show()