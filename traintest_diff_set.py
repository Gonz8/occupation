import numpy
import matplotlib.pyplot as plt
import sys
import time
from math import sqrt

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from scipy.stats import pearsonr

def doPCA(X_train, X_test, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(X_train)

    print("PCA explained variance ratio:")
    print(pca.explained_variance_ratio_)

    x_train_new = pca.transform(X_train)
    x_test_new = pca.transform(X_test)
    return x_train_new, x_test_new

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

hotel_capacity = 86

set_nr_path_prefix = "10/10"

X_train = numpy.loadtxt(set_nr_path_prefix+"_X_train.csv", delimiter=",")
X_test = numpy.loadtxt(set_nr_path_prefix+"_X_test.csv", delimiter=",")
y_train = numpy.loadtxt(set_nr_path_prefix+"_y_train.csv", delimiter=",")
y_test = numpy.loadtxt(set_nr_path_prefix+"_y_test.csv", delimiter=",")

print("train and test array shape:")
print(X_train.shape)
print(X_test.shape)


# print("Correlation przed PCA feature/target")
# print(pearsonr(X_train[:,0],y_train))
# print(pearsonr(X_train[:,1],y_train))
# print(pearsonr(X_train[:,2],y_train))
# print(pearsonr(X_train[:,3],y_train))
# print(pearsonr(X_train[:,4],y_train))
# print(pearsonr(X_train[:,5],y_train))
# print(pearsonr(X_train[:,6],y_train))

print("MSE 14days feature as prediction (train):")
print(mean_squared_error(y_train,X_train[:,6]))
print("MSE 14days feature as prediction (test) : %s" % (mean_squared_error(y_test,X_test[:,6])))

m_y_train = numpy.mean(y_train)
y_train_onlymean = numpy.zeros(y_train.shape[0])
y_train_onlymean[:] = m_y_train
print("MSE (mean from y_train) as prediction  : %s" % (mean_squared_error(y_train,y_train_onlymean)))

X_train, X_test = doPCA(X_train, X_test, 7)

start_time = time.time()
mlp = MLPRegressor(hidden_layer_sizes=(3,3), max_iter=80000, verbose=False, alpha=0.00009, learning_rate_init=0.0001, tol=0.00001)
mlp.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

print("train loss: %s" % (mlp.loss_))

predictions = mlp.predict(X_test)

mse = mean_squared_error(y_test,predictions)
rmse = sqrt(mse)
map = rmse/hotel_capacity

print("MSE : %s" % (mse))
print("RMSE : %s" % (rmse))
print("Main average prediction:")
print(map)

mean_val = numpy.mean(y_train)
y_test_mean = numpy.zeros(y_test.shape[0])
y_test_mean[:] = mean_val

print("MSE mean : %s" % (mean_squared_error(y_test_mean,predictions)))



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
#0.0001

#train_sizes, train_scores, valid_scores = learning_curve(mlp, X, y, train_sizes=[50, 80, 110], cv=5)

title = "Learning Curves"
# plot_learning_curve(mlp, title, X, y, cv=5, n_jobs=4)
# plt.show()