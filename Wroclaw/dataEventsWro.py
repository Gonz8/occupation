import datetime
import numpy
import sys
from collections import Counter


def dow(date):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dayNumber = date.weekday()
    return days[dayNumber]


def toDate(dateStr):
    date = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
    return date


def dateDiff(d1, d2):
    return (d2 - d1).days

# def uniArray(array_unicode):
#     items = [x.encode('utf-8') for x in array_unicode]
#     array_unicode = numpy.array([items])
#     return array_unicode


dtn = numpy.string_
names = numpy.loadtxt("eventsWro.csv", delimiter=",", dtype=dtn, usecols=[5])
# print(names.shape)
# print(names[0])
c = Counter(names)

dt = 'a10,a10,a500'
csv = numpy.loadtxt("eventsWro.csv", delimiter=",", dtype=dt, usecols=[0,1,5])
csv_last_stay_day = datetime.datetime.strptime("2017-06-05", '%Y-%m-%d')
csv_frst_stay_day = datetime.datetime.strptime("2014-08-11", '%Y-%m-%d')
csv_bigyear = datetime.datetime.strptime("3000-01-01", '%Y-%m-%d')
print(csv.size)
samples = []
sample = []
k = 1
l = 1
for i in csv:
    firstDay = toDate(i[0])
    lastDay = toDate(i[1])
    event = i[2]
    if(dateDiff(csv_frst_stay_day, firstDay) < 0 & dateDiff(csv_frst_stay_day, lastDay) < 0):
        continue
    if(dateDiff(csv_last_stay_day, firstDay) > 0):
        continue
    if(dateDiff(csv_bigyear,lastDay) > 0):
        continue
    #USUN JEDNO 2014-11-25 00:00	2015-08-30 00:00	Estetyka rytualu. Chinska sztuka ludowa z kolekcji dr Zlaty Cerny
    if (dateDiff(firstDay, datetime.datetime.strptime("2014-11-25", '%Y-%m-%d')) == 0):
        continue
    if (dateDiff(firstDay, datetime.datetime.strptime("2001-02-22", '%Y-%m-%d')) == 0):   #2001-02-22 00:00	2015-09-22 00:00	Memoriada
        continue
    if (c[event] >= 6):
        k+=1
        continue
    if (dateDiff(firstDay, lastDay) > 7):
        l+=1
        continue
    sample.append(firstDay)
    sample.append(lastDay)
    sample.append(event.decode('utf-8'))
    samples.append(sample)
    sample = []

print(len(samples))
print(k, "przez wystapienia")
print(l, "przez cz trwania")

# for (i, sam) in enumerate(training_data):
# 	print i, sam

numpy.savetxt("eventsWroFiltered.csv", numpy.asarray(samples), fmt='%s', delimiter=",")
