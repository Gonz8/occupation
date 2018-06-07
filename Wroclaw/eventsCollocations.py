import numpy as np
import collections
import nltk
from nltk.collocations import *

def findBigrams(tokens):
    bigram = nltk.collocations.BigramAssocMeasures
    finder = BigramCollocationFinder.from_words(tokens)
    return finder.nbest(bigram.pmi, 20)

def loadData():
    dtx = 'a10, a10, a500, a500'
    csv = np.loadtxt("eventsWroFilteredClarin.csv", delimiter=",", dtype=dtx, usecols=[0,1,2,3])
    print(csv.shape)
    print(csv[0])
    return csv
    # samples = []
    # sample = []
    # k = 1
    # for i in csv:
    #     firstDay = i[0]
    #     lastDay = i[1]
    #     event = i[2]
    #     wordsArray = lpmnProcess(event)
    #     wordsString = ".".join(wordsArray)
    #     sample.append(firstDay)
    #     sample.append(lastDay)
    #     sample.append(event.decode('utf-8'))
    #     sample.append(wordsString)
    #     if wordsString != "":
    #         samples.append(sample)
    #     sample = []
    #
    #     print(k, event)
    #     k+=1
    #     # if k == 401:
    #     #     break
    #
    # print(len(samples))
    # np.savetxt("eventsWroFilteredClarin.csv", np.asarray(samples), fmt='%s', delimiter=",")
    #
    # np.savetxt("newInputs0.csv", np.asarray(words), fmt='%s', delimiter=",")

data = loadData()
k = 1
collocationsList = []
for i in data:
    k+=1
    evWords = i[3].split('.')
    collocations = findBigrams(evWords)
    for coll in collocations:
        collocationsList.append(coll)
    # if k > 11:
    #     break

print(collocationsList)

c = collections.Counter(collocationsList)
print(c)