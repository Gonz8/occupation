import numpy
import matplotlib.pyplot as plt
import sys

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from scipy.stats import pearsonr

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=numpy.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

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

print("Correlation przed PCA feature/target")
print(pearsonr(X_train[:,0],y_train))
print(pearsonr(X_train[:,1],y_train))
print(pearsonr(X_train[:,2],y_train))
print(pearsonr(X_train[:,3],y_train))
print(pearsonr(X_train[:,4],y_train))
print(pearsonr(X_train[:,5],y_train))
print(pearsonr(X_train[:,6],y_train))

print("MSE feature as prediction:")
print(mean_squared_error(y_train,X_train[:,6]))

pca = PCA(n_components=7)
pca.fit(X_train)

print(pca.explained_variance_ratio_)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(5,15), max_iter=5500, verbose=False, alpha=0.0055)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
print("MSE : %s" % (mean_squared_error(y_test,predictions)))

mean_val = numpy.mean(y_train)
y_test_mean = numpy.zeros(y_test.shape[0])
y_test_mean[:] = mean_val

print("MSE mean : %s" % (mean_squared_error(y_test_mean,predictions)))

print("MSE output as 14 days before : %s" % (mean_squared_error(y_test,X_test[:,6])))

corr = pearsonr(y_test,predictions)
print("Correlation yTest/predictions")
print(corr)

# print("Correlation po PCA feature/target")
# print(pearsonr(X_train[:,0],y_train))
# print(pearsonr(X_train[:,1],y_train))
# print(pearsonr(X_train[:,2],y_train))
# print(pearsonr(X_train[:,3],y_train))
# print(pearsonr(X_train[:,4],y_train))
# print(pearsonr(X_train[:,5],y_train))
# print(pearsonr(X_train[:,6],y_train))
0.0001

#train_sizes, train_scores, valid_scores = learning_curve(mlp, X, y, train_sizes=[50, 80, 110], cv=5)

title = "Learning Curves"
plot_learning_curve(mlp, title, X, y, cv=5, n_jobs=4)
plt.show()