import numpy
import matplotlib.pyplot as plt
import sys
import time
from math import sqrt

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr

def save_train_test_dataset(set_nr, X_train, X_test, y_train, y_test):
    numpy.savetxt(set_nr+"/"+set_nr+"_X_train.csv", X_train, fmt='%s', delimiter=",")
    numpy.savetxt(set_nr+"/"+set_nr+"_X_test.csv", X_test, fmt='%s', delimiter=",")
    numpy.savetxt(set_nr+"/"+set_nr+"_y_train.csv", y_train, fmt='%s', delimiter=",")
    numpy.savetxt(set_nr+"/"+set_nr+"_y_test.csv", y_test, fmt='%s', delimiter=",")

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


hotel_capacity = 178
dta0 = 3
dta4 = 2
dta7 = 1

predictDTA = dta0

# dt='a10,i4,i4,i4,i4,i4,i4,i4'
# dty='a10,i4,i4,i4'
X = numpy.loadtxt("occWroInputs.csv", delimiter=",", skiprows=1, usecols=[1,2,3,4,5,6,7])
y = numpy.loadtxt("occWroOutputs.csv", delimiter=",", skiprows=1, usecols=[predictDTA])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# print("MSE 14days feature as prediction (train):")
# print(mean_squared_error(y_train,X_train[:,6]))
# print("MSE 14days feature as prediction (test) : %s" % (mean_squared_error(y_test,X_test[:,6])))
#
# #size = X.shape[0]
# print(X_train.shape)
# print(X_test.shape)
# save_train_test_dataset("10", X_train, X_test, y_train, y_test)
# sys.exit()
#
#

# seaborn.regplot(X[:,6], y)
# plt.show()

size = X.shape[0]
trainSize = int(round(size*0.8))

X_train = X[:trainSize,:]
X_test = X[trainSize:,:]
X_train = X_train[:-14,:]
y_train = y[:trainSize,]
y_test = y[trainSize:,]
y_train = y_train[:-14,]

# set_nr_path_prefix = "4/4"
# X_test = numpy.loadtxt(set_nr_path_prefix+"_X_test.csv", delimiter=",")
# y_test = numpy.loadtxt(set_nr_path_prefix+"_y_test.csv", delimiter=",")

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape)
print(X_test.shape)
# sys.exit()

print("Correlation przed PCA feature/target")
print(pearsonr(X_train[:,0],y_train))
print(pearsonr(X_train[:,1],y_train))
print(pearsonr(X_train[:,2],y_train))
print(pearsonr(X_train[:,3],y_train))
print(pearsonr(X_train[:,4],y_train))
print(pearsonr(X_train[:,5],y_train))
print(pearsonr(X_train[:,6],y_train))

print("MSE 14days feature as prediction (train):")
print(mean_squared_error(y_train,X_train[:,6]))
print("MSE 14days feature as prediction (test) : %s" % (mean_squared_error(y_test,X_test[:,6])))

m_y_train = numpy.mean(y_train)
y_train_onlymean = numpy.zeros(y_train.shape[0])
y_train_onlymean[:] = m_y_train
print("MSE (mean from y_train) as prediction  : %s" % (mean_squared_error(y_train,y_train_onlymean)))

pca = PCA(n_components=7)
pca.fit(X_train)

print(pca.explained_variance_ratio_)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

start_time = time.time()
mlp = MLPRegressor(hidden_layer_sizes=(5,5), max_iter=80000, verbose=False, alpha=0.38, learning_rate_init=0.0001, tol=0.00001)
mlp.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))


predictions = mlp.predict(X_test)

mse = mean_squared_error(y_test,predictions)
rmse = sqrt(mse)
map = rmse/hotel_capacity

print("MSE : %s" % (mse))
print("RMSE : %s" % (rmse))
print("Main avarage prediction:")
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
0.0001

#train_sizes, train_scores, valid_scores = learning_curve(mlp, X, y, train_sizes=[50, 80, 110], cv=5)

title = "Learning Curves"
# plot_learning_curve(mlp, title, X, y, cv=5, n_jobs=4)
# plt.show()