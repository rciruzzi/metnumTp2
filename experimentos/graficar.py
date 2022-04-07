from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def graficarPuntos(y_values, x_values, y_name, x_name, title):
    plt.figure(figsize=(14,16), dpi= 80)
    plt.scatter(x_values, y_values, s=30, alpha=1)    
    plt.title(title, fontdict={'size':20})
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid(linestyle='--', alpha=0.5)
    # plt.xlim(-0.01, 1.01)
    plt.legend()
    plt.show()

def graficarHeatmap(predicciones, valores_reales):
    df = pd.DataFrame({"valor": valores_reales, "prediccion": predicciones})
    heatmap = []

    for i in range(10):
        heatmap.append([])
        for j in range(10):
            filtrado = df[(df.valor == i) & (df.prediccion == j)].groupby("valor").count()
            heatmap[i].append(filtrado.values[0][0] if filtrado.values.shape[0] > 0 else 0)
    for i in range(10):
        sumFila = np.sum(heatmap[i])
        for j in range(10):
            heatmap[i][j] /= sumFila 


    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)

    # We want to show all ticks...
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    # ... and label them with the respective list entries
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, str(round(heatmap[i][j], 2)),
                           ha="center", va="center", color="w")

    ax.set_title("Aciertos por clase")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    fig.tight_layout()
    plt.show()

# graficarHeatmap([1,1,1,1,1,1,1,1,2, 0,0 ,0,0,0,0,0,0,0,0], [1,2,3,3,5,6,7,8,2,0,7, 0,0,0,0,0,0,0,0])
