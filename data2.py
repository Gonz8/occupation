import datetime
import numpy

def dow(date):
    days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dayNumber=date.weekday()
    return days[dayNumber]

def toDate(dateStr):
	date = datetime.datetime.strptime(dateStr , '%Y-%m-%d')
	return date

def dateDiff(d1, d2):
	return (d2 - d1).days

dt='a10,a10,i4'
csv = numpy.loadtxt("occupation.csv", delimiter=",", dtype=dt)
csv_last_stay_day = datetime.datetime.strptime("2017-01-24" , '%Y-%m-%d')
samples = []
sample = []
outp = []
outps = []
stay_dates = []
#k = 1
for i in csv:
	stay = toDate(i[0])
	report = toDate(i[1])
	bookings = i[2]
	if (dateDiff(stay,csv_last_stay_day) <= 700):
		if not (dow(stay) in ["Friday", "Saturday"]):
			if not (stay in stay_dates):
				stay_dates.append(stay)
				sample = []
				outp = []
				#sample.append(stay.strftime('%Y-%m-%d'))
			#if not fridayitp
			if (dateDiff(report,stay) == 200):
				sample.append(bookings)
			if (dateDiff(report,stay) == 100):
				sample.append(bookings)
			if (dateDiff(report,stay) == 50):
				sample.append(bookings)
			if (dateDiff(report,stay) == 28):
				sample.append(bookings)
			if (dateDiff(report,stay) == 21):
				sample.append(bookings)
			if (dateDiff(report,stay) == 17):
				sample.append(bookings)
			if (dateDiff(report,stay) == 14):
				sample.append(bookings)
				
			if (dateDiff(report,stay) == -1):
				samples.append(sample)
				sample = []
				outp.append(bookings) #output - ile osob faktycznie spalo w hotelu
				outps.append(outp)
				outp = []
				# if (dateDiff(stay,csv_last_stay_day) == 80):
				# 	training_data = samples
				# 	samples = []

#test_data = samples
print(len(stay_dates))
print(len(samples))
print(len(outps))

# for (i, sam) in enumerate(training_data):
# 	print i, sam

numpy.savetxt("occupationInputs.csv", numpy.asarray(samples), fmt='%s', delimiter=",")
numpy.savetxt("occupationOutputs.csv", numpy.asarray(outps), fmt='%s', delimiter=",")
