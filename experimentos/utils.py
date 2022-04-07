import metnum
import pandas as pd
from sklearn.metrics import accuracy_score

def loadData(frac=1):
    df_train = pd.read_csv("../data/train.csv")

    df_train = df_train.sample(frac=frac)

    X = df_train[df_train.columns[1:]].values
    y = df_train["label"].values.reshape(-1, 1)

    return X, y

def splitData(X, y, fracTrain=0.8):
    limit = int(fracTrain * X.shape[0]) 

    X_train, y_train = X[:limit], y[:limit]
    X_val, y_val = X[limit:], y[limit:]

    return X_train, y_train, X_val, y_val


def accuracyKNNWithPCA(X_train, y_train, X_val, y_val, alpha, k):
    pca = metnum.PCA(alpha)
    pca.fit(X_train)

    X_train_transform = pca.transform(X_train)
    X_val_transform = pca.transform(X_val)

    clf = metnum.KNNClassifier(k)
    clf.fit(X_train_transform, y_train)
    y_pred = clf.predict(X_val_transform)

    return accuracy_score(y_val, y_pred)