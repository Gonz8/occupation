import datetime
import numpy
import sys


def dow(date):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dayNumber = date.weekday()
    return days[dayNumber]


def toDate(dateStr):
    date = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
    return date


def dateDiff(d1, d2):
    return (d2 - d1).days


dt = 'a10,i4,i4'
csv = numpy.loadtxt("occupationWro.csv", delimiter=",", dtype=dt)
csv_last_stay_day = datetime.datetime.strptime("2017-06-05", '%Y-%m-%d')
csv_frst_stay_day = datetime.datetime.strptime("2014-08-11", '%Y-%m-%d')
print(csv.size)
samples = [["stay day", "dta 200", "dta 100", "dta 50", "dta 28", "dta 21", "dta 17", "dta 14"]]
sample = []
outp = []
outps = [["stay day", "dta 7 - bookings", "dta 4 bookings", "dta 0 - bookings"]]
stay_dates = []
#k = 1
for i in csv:
    stay = toDate(i[0])
    dta = i[1]
    bookings = i[2]
    if(dateDiff(stay,csv_last_stay_day) >= 1090 | dateDiff(csv_frst_stay_day, stay) < 0):
        continue
    if not (dow(stay) in ["Friday", "Saturday"]):
        if not (stay in stay_dates):
            stay_dates.append(stay)
            sample = [stay]
            outp = [stay]
        if (dta == 200):
            sample.append(bookings)
        if (dta == 100):
            sample.append(bookings)
        if (dta == 50):
            sample.append(bookings)
        if (dta == 28):
            sample.append(bookings)
        if (dta == 21):
            sample.append(bookings)
        if (dta == 17):
            sample.append(bookings)
        if (dta == 14):
            sample.append(bookings)
        if (dta == 7):
            outp.append(bookings)
        if (dta == 4):
            outp.append(bookings)

        if (dta == 0):
            samples.append(sample)
            sample = []
            outp.append(bookings)  # output - ile osob faktycznie spalo w hotelu
            outps.append(outp)
            outp = []

print(len(stay_dates))
print(len(samples))
print(len(outps))


# for (i, sam) in enumerate(training_data):
# 	print i, sam

numpy.savetxt("occWroInputs.csv", numpy.asarray(samples), fmt='%s', delimiter=",")
numpy.savetxt("occWroOutputs.csv", numpy.asarray(outps), fmt='%s', delimiter=",")
