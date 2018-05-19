import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

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

X = np.loadtxt("occWroInputs.csv", delimiter=",", skiprows=1, usecols=[1,2,3,4,5,6,7])
y = np.loadtxt("occWroOutputs.csv", delimiter=",", skiprows=1, usecols=[predictDTA])
#
size = X.shape[0]
trainSize = int(round(size*0.8))

X_train = X[:trainSize,:]
X_test = X[trainSize:,:]
X_train = X_train[:-14,:]
y_train = y[:trainSize,]
y_test = y[trainSize:,]
y_train = y_train[:-14,]

pca = PCA(n_components=7)
pca.fit(X_train)

print(pca.explained_variance_ratio_)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(5,5), max_iter=95000)

scores = []
errors = []
correlations = []

alphas = np.logspace(-6,1,25)

#Train the model with different L2 penalty (regularization term) - alpha
for a in alphas:
    mlp.set_params(alpha=a)
    mlp.fit(X_train, y_train)
    scores.append(mlp.score(X_test, y_test))
    predictions = mlp.predict(X_test)
    errors.append(mean_squared_error(y_test,predictions))
    correlations.append(pearsonr(y_test,predictions))

#Display results
plt.figure()
gs = gridspec.GridSpec(1, 2)

plt.subplot(gs[0,0])
ax = plt.gca()
ax.plot(alphas, scores)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('score R^2')
plt.title('MLPRegressor model scores as a function of the regularization')
plt.axis('tight')

plt.subplot(gs[0,1])
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error MSE')
plt.title('MLPRegressor model MPE as a function of the regularization')
plt.axis('tight')

plt.show()

