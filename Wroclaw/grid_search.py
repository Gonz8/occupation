import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from scipy.stats import pearsonr


hotel_capacity = 178
dta0 = 3
dta4 = 2
dta7 = 1

predictDTA = dta0

# set_nr_path_prefix = "9/9"
#
# X_train = np.loadtxt(set_nr_path_prefix+"_X_train.csv", delimiter=",")
# X_test = np.loadtxt(set_nr_path_prefix+"_X_test.csv", delimiter=",")
# y_train = np.loadtxt(set_nr_path_prefix+"_y_train.csv", delimiter=",")
# y_test = np.loadtxt(set_nr_path_prefix+"_y_test.csv", delimiter=",")



# dt='a10,i4,i4,i4,i4,i4,i4,i4'
# dty='a10,i4,i4,i4'
X = np.loadtxt("occWroInputs.csv", delimiter=",", skiprows=1, usecols=[1,2,3,4,5,6,7])
y = np.loadtxt("occWroOutputs.csv", delimiter=",", skiprows=1, usecols=[predictDTA])
#
size = X.shape[0]
trainSize = int(round(size*0.8))
#
X_train = X[:trainSize,:]
X_test = X[trainSize:,:]
X_train = X_train[:-14,:]
y_train = y[:trainSize,]
y_test = y[trainSize:,]
y_train = y_train[:-14,]


pca = PCA(n_components=7)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

layers = np.arange(1,4)
neurons = np.arange(1,11,2)
hls_array = []
train_losses = []
test_losses = []

mlp = MLPRegressor(hidden_layer_sizes=(5,15), max_iter=45000, verbose=False, alpha=0.4)

for l in layers:
    for n in neurons:
        if (l==1):
            hls = (n)
        if (l==2):
            hls = (n,n)
        if (l==3):
            hls = (n,n,n)
        if (l==4):
            hls = (n,n,n,n)
        if (l == 5):
            hls = (n, n, n, n, n)
        hls_array.append(hls)

for hls in hls_array:
    mlp.set_params(hidden_layer_sizes=hls)
    mlp.fit(X_train, y_train)
    train_losses.append(mlp.loss_)
    predictions = mlp.predict(X_test)
    test_loss = np.mean((predictions - y_test)**2)
    test_losses.append(test_loss)
    print("%s: train_loss: %s, test_loss: %s" % (hls, mlp.loss_, test_loss))

i_hls_optim = np.argmin(test_losses)
hls_optim = hls_array[i_hls_optim]
print("Optimal hidden_layer_sizes parameter:")
print(hls_optim)



#mlp = MLPRegressor(hidden_layer_sizes=(5,15), max_iter=5500, verbose=False, alpha=0.0055)
#mlp.fit(X_train, y_train)

# predictions = mlp.predict(X_test)
# print("MSE : %s" % (mean_squared_error(y_test,predictions)))
