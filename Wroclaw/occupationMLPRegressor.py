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

dta0 = 3
dta4 = 2
dta7 = 1

class occupationMLPRegressor:
    hotelCapacity = 178
    predictDTA = dta0

    def __init__(self, inputFile, outputFile, predictDta = dta0, layers = (3,3), alpha = 0.0009, randomSamples = False, hotelCapacity = None):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.predictDTA = predictDta
        # MLP Regressor
        self.mlp = MLPRegressor(hidden_layer_sizes=layers, max_iter=80000, verbose=False, alpha=alpha, learning_rate_init=0.0001, tol=0.00001)
        self.randomSamples = randomSamples
        if hotelCapacity is not None:
            self.hotelCapacity = hotelCapacity

    def loadData(self):
        dtm = 'a10, float, float, float, float, float, float, float'
        X = numpy.loadtxt(self.inputFile, delimiter=",", skiprows=1, usecols=[1, 2, 3, 4, 5, 6, 7])
        y = numpy.loadtxt(self.outputFile, delimiter=",", skiprows=1, usecols=[self.predictDTA])
        print("X Shape:")
        print(X.shape)
        print(X[0])

        dt ='a10'
        Xdates = numpy.loadtxt(self.inputFile, delimiter=",", skiprows=1, dtype=dt, usecols=[0])
        print(Xdates.shape)
        print(Xdates[0])
        return X, y, Xdates

    def trainTestSplit(self, X, y, dates):
        if self.randomSamples:
            return train_test_split(X, y, test_size=0.2)
        size = X.shape[0]
        trainSize = int(round(size * 0.8))

        X_train = X[:trainSize, :]
        X_test = X[trainSize:, :]
        X_train = X_train[:-14, :]
        y_train = y[:trainSize, ]
        y_test = y[trainSize:, ]
        y_train = y_train[:-14, ]

        trainDates = dates[:trainSize]
        trainDates = trainDates[:-14]
        testDates = dates[trainSize:]
        return X_train,X_test,y_train,y_test, trainDates, testDates

    def trainTestSplitWithPrint(self, X, y, dates):
        X_train, X_test, y_train, y_test, trainDates, testDates = self.trainTestSplit(X,y,dates)
        print("X train SIZE : %d" % (X_train.shape[0]))
        print("X test SIZE : %d" % (X_test.shape[0]))
        print("trainDates SIZE : %d" % (trainDates.shape[0]))
        print("testDates SIZE : %d" % (testDates.shape[0]))
        return X_train, X_test, y_train, y_test, trainDates, testDates

    def correlationPrint(self, X_train, y_train):
        print("Correlation feature/target:")
        print(pearsonr(X_train[:, 0], y_train))
        print(pearsonr(X_train[:, 1], y_train))
        print(pearsonr(X_train[:, 2], y_train))
        print(pearsonr(X_train[:, 3], y_train))
        print(pearsonr(X_train[:, 4], y_train))
        print(pearsonr(X_train[:, 5], y_train))
        print(pearsonr(X_train[:, 6], y_train))

    def MSEbeforePrint(self, X_train, X_test, y_train, y_test):
        print("MSE 14days feature as prediction (train):")
        print(mean_squared_error(y_train, X_train[:, 6]))
        print("MSE 14days feature as prediction (test) : %s" % (mean_squared_error(y_test, X_test[:, 6])))

        m_y_train = numpy.mean(y_train)
        y_train_onlymean = numpy.zeros(y_train.shape[0])
        y_train_onlymean[:] = m_y_train
        print("MSE (mean from y_train) as prediction  : %s" % (mean_squared_error(y_train, y_train_onlymean)))

    def doPCA(self, X_train, X_test):
        PCAComp = X_train.shape[1]
        pca = PCA(n_components=PCAComp)
        pca.fit(X_train)

        print(pca.explained_variance_ratio_)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        return X_train, X_test

    def trainPredict(self, X_train, X_test, y_train):
        start_time = time.time()
        self.mlp.fit(X_train, y_train)
        print("--- %s seconds ---" % (time.time() - start_time))
        predictions = self.mlp.predict(X_test)
        return predictions

    def predictErrorsPrint(self, predictions, y_train, y_test):
        mse = mean_squared_error(y_test, predictions)
        rmse = sqrt(mse)
        map = rmse / self.hotelCapacity

        print("MSE : %s" % (mse))
        print("RMSE : %s" % (rmse))
        print("Main avarage prediction:")
        print(map)

        mean_val = numpy.mean(y_train)
        y_test_mean = numpy.zeros(y_test.shape[0])
        y_test_mean[:] = mean_val

        print("MSE mean : %s" % (mean_squared_error(y_test_mean, predictions)))

    def execute(self):
        X, y, Xdates = self.loadData()
        X_train, X_test, y_train, y_test, trainDates, testDates  = self.trainTestSplitWithPrint(X, y, Xdates)
        self.correlationPrint(X_train, y_train)
        self.MSEbeforePrint(X_train, X_test, y_train, y_test)

        X_train, X_test = self.doPCA(X_train, X_test)
        predictions = self.trainPredict(X_train, X_test, y_train)
        self.predictErrorsPrint(predictions, y_train, y_test)
        return testDates, predictions, y_test

    def executeAndReturnPredictDiff(self):
        dates, predictions, realValues = self.execute()
        dateToDiff = {}
        for idx, date in enumerate(dates):
            pred = predictions[idx]
            real = realValues[idx]
            diff = real - pred
            dateToDiff[date] = diff
            # print(date,diff)
        return dateToDiff


def main():
    occMLP = occupationMLPRegressor("occWroInputs.csv","occWroOutputs.csv", dta0, (5,5), 0.38)
    occMLP.executeAndReturnPredictDiff()

if __name__ == '__main__':
    main()
