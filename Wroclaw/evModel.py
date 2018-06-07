import numpy
import matplotlib.pyplot as plt
import sys
import time
from math import sqrt

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from occupationMLPRegressor import occupationMLPRegressor
from sets import Set

from scipy.stats import pearsonr

dta0 = 3
dta4 = 2
dta7 = 1

def toDate(dateStr):
    date = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
    return date

def isDateBetween(date, start, end):
    if start <= date <= end:
        return True
    return False

class evModel:
    hotelCapacity = 178
    predictDTA = dta0
    maxY=0

    def __init__(self, inputFile, outputFile, predictDta = dta0, layers = (3,3), alpha = 0.0009, layCorr = (5,5), alphCorr = 0.0009, randomSamples = False, hotelCapacity = None):
        self.baseModel = occupationMLPRegressor(inputFile,outputFile,predictDta,layers,alpha,randomSamples,hotelCapacity)
        self.randomSamples = randomSamples
        if hotelCapacity is not None:
            self.hotelCapacity = hotelCapacity
        #MLPRegressor
        self.mlp = MLPRegressor(hidden_layer_sizes=layCorr, max_iter=180000, verbose=False, alpha=alphCorr, learning_rate_init=0.0001, tol=0.00001)

    def loadData(self, tempDataSave=False):
        X = []
        y = []
        dates = []
        dtx = 'a10, a10, a500'
        events = numpy.loadtxt("eventsWroFilteredClarin.csv", delimiter=",", dtype=dtx, usecols=[0,1,3])
        dtf = numpy.string_
        Xfeatures = numpy.loadtxt("newInputs0.csv", delimiter=",", dtype=dtf)
        print("events Shape:")
        print(events.shape)
        # print(events[0])

        if tempDataSave:
            header = []
            for w in Xfeatures.transpose().tolist():
                header.append(w.decode('utf-8'))
            X.append(header)


        print("X features Shape:")
        print(Xfeatures.shape)
        # print(Xfeatures[0])

        dateToDiff, dateToPred, dateToRealVal = self.baseModel.executeAndReturnPredictDiff(False)
        print(len(dateToDiff))
        # we cannot remove rows with correction <= 0 - we don't know the real value while predicting
        # for key in dateToDiff.keys():
        #     if dateToDiff[key] <= 0:
        #         del dateToDiff[key]
        # print(len(dateToDiff))

        for key in dateToDiff:
            corr = dateToDiff[key]
            sample = numpy.zeros((Xfeatures.shape[0],), dtype=int)
            sample = self.prepareFindWordSample(sample, Xfeatures, events, toDate(key))
            X.append(sample.tolist())
            dates.append(key)
            if tempDataSave:
                y.append([int(round(corr)), key])
            else:
                y.append(int(round(corr)))

        print(numpy.array(X).shape)
        # print(numpy.array(X)[0])
        print(numpy.array(y).shape)
        # print(numpy.array(y)[0])

        if tempDataSave:
            numpy.savetxt("eventsTempInp.csv", numpy.asarray(X), fmt='%s', delimiter=",")
            numpy.savetxt("eventsTempOutp.csv", numpy.asarray(y), fmt='%s', delimiter=",")

        y = numpy.array(y)
        self.maxY = numpy.max(y)

        return numpy.array(X), y, numpy.array(dates), dateToPred, dateToRealVal


    def prepareFindWordSample(self, sample, Xfeatures, events, date):
        allWordsForDate = Set()
        for ev in events:
            firstDay = toDate(ev[0])
            lastDay = toDate(ev[1])
            evWords = ev[2].split(".")
            if isDateBetween(date,firstDay,lastDay):
                allWordsForDate.update(evWords)
        # print(allWordsForDate)
        for idx, word in enumerate(Xfeatures):
            if word in allWordsForDate:
                sample[idx] = 1
        return sample



    def trainTestSplit(self, X, y, dates):
        if self.randomSamples:
            return train_test_split(X, y, test_size=0.33)
        size = X.shape[0]
        trainSize = int(round(size * 0.66))

        X_train = X[:trainSize, :]
        X_test = X[trainSize:, :]
        y_train = y[:trainSize, ]
        y_test = y[trainSize:, ]
        dates_train = dates[:trainSize,]
        dates_test = dates[trainSize:]

        return X_train,X_test,y_train,y_test, dates_train, dates_test

    def trainTestSplitWithPrint(self, X, y, dates):
        X_train, X_test, y_train, y_test, dates_train, dates_test = self.trainTestSplit(X,y,dates)
        print("X train SIZE : %d" % (X_train.shape[0]))
        print("X test SIZE : %d" % (X_test.shape[0]))
        return X_train, X_test, y_train, y_test, dates_train, dates_test

    def MSEbeforePrint(self, X_train, X_test, y_train, y_test):
        # print("MSE 14days feature as prediction (train):")
        # print(mean_squared_error(y_train, X_train[:, 6]))
        # print("MSE 14days feature as prediction (test) : %s" % (mean_squared_error(y_test, X_test[:, 6])))

        m_y_train = numpy.mean(y_train)
        y_train_onlymean = numpy.zeros(y_train.shape[0])
        y_train_onlymean[:] = m_y_train
        print("MSE (mean from y_train) as prediction  : %s" % (mean_squared_error(y_train, y_train_onlymean)))

    #For LSA, a value of 100 is recommended.
    def doLSA(self, X_train, X_test, lsaComponents):
        lsa = TruncatedSVD(n_components=lsaComponents)
        lsa.fit(X_train)

        print("LSA explained variance ratio SUM: %s" % (lsa.explained_variance_ratio_.sum()))

        X_train = lsa.transform(X_train)
        X_test = lsa.transform(X_test)
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
        map = rmse / self.maxY

        print("MSE : %s" % (mse))
        print("RMSE : %s" % (rmse))
        print("Main avarage prediction:")
        print(map)

        # mean_val = numpy.mean(y_train)
        # y_test_mean = numpy.zeros(y_test.shape[0])
        # y_test_mean[:] = mean_val
        #
        # print("MSE mean : %s" % (mean_squared_error(y_test_mean, predictions)))

    def predictErrorsPrint2(self, predictions, y_test):
        mse = mean_squared_error(y_test, predictions)
        rmse = sqrt(mse)
        map = rmse / self.hotelCapacity

        print("MSE : %s" % (mse))
        print("RMSE : %s" % (rmse))
        print("Main avarage prediction:")
        print(map)

    def execute(self):
        X, y, dates, dateToPred, dateToRealVal = self.loadData()
        X_train, X_test, y_train, y_test, dates_train, dates_test  = self.trainTestSplitWithPrint(X, y, dates)

        self.MSEbeforePrint(X_train, X_test, y_train, y_test)

        X_train, X_test = self.doLSA(X_train, X_test,200)
        predictions = self.trainPredict(X_train, X_test, y_train)
        self.predictErrorsPrint(predictions, y_train, y_test)
        #self.printPred(dates_test, predictions, y_test)

        final_predictions = []
        realValues = []
        for idx, date in enumerate(dates_test):
            pred = predictions[idx]
            basePred = dateToPred[date]
            realVal = dateToRealVal[date]
            if (pred > 0):
                final_predictions.append(basePred + pred)
            else:
                final_predictions.append(basePred)

            realValues.append(realVal)
        self.predictErrorsPrint2(numpy.asarray(final_predictions), numpy.asarray(realValues))

        return dates_test, predictions, y_test

    def printPred(self, dates, predictions, y):
        for idx, date in enumerate(dates):
            print(date, "prediction:", predictions[idx], "realVal:", y[idx])

    # def executeAndReturnPredictDiff(self):
    #     dates, predictions, realValues = self.execute()
    #     dateToDiff = {}
    #     for idx, date in enumerate(dates):
    #         pred = predictions[idx]
    #         real = realValues[idx]
    #         diff = real - pred
    #         dateToDiff[date] = diff
    #         # print(date,diff)
    #     return dateToDiff


def main():
    evM = evModel("occWroInputs.csv","occWroOutputs.csv", dta0, (5,5), 0.38);
    evM.execute()

if __name__ == '__main__':
    main()
