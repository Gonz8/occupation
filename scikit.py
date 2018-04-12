import numpy
import sys

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

dt='i3,i3,i3,i3,i3,i3,i3'
dty='i3'
X = numpy.loadtxt("occupationInputs.csv", delimiter=",")
y = numpy.loadtxt("occupationOutputs.csv", delimiter=",")

size = X.shape[0]
trainSize = int(round(size*0.8))

X_train = X[:trainSize,:]
X_test = X[trainSize:,:]
X_train = X_train[:-14,:]
y_train = y[:trainSize,]
y_test = y[trainSize:,]
y_train = y_train[:-14,]

print(X_train.shape)

pca = PCA(n_components=7)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(30,30,30), max_iter=500)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
