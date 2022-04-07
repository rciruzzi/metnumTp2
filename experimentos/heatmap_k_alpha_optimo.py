import metnum
import utils
import graficar

X, y = utils.loadData()

X_train, y_train, X_val, y_val = utils.splitData(X, y)

k = 7
alpha = 35

pca = metnum.PCA(alpha)
pca.fit(X_train)

X_train_transform = pca.transform(X_train)
X_val_transform = pca.transform(X_val)

clf = metnum.KNNClassifier(k)
clf.fit(X_train_transform, y_train)
y_pred = clf.predict(X_val_transform)

y_pred = y_pred.reshape(y_pred.shape[0])
y_val = y_val.reshape(y_val.shape[0])

graficar.graficarHeatmap(y_pred, y_val)
