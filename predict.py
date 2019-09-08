import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dates = []
prices = []



def getdata(filename):
	with open(filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(row[0])
			prices.append(float(row[1]))
	#print(dates)
	#print(prices)
	return

def showplot(dates,prices)
	linear_mod = linear_model.LinearRegression()
	dates = np.reshape(dates, (len(dates),1))
	prices = np.reshape(prices, (len(prices),1))
	linear_mod.fit(dates,prices)
	plt.scatter(dates,prices,color='yellow')
	plt.plot(dates,linear_mod.predict(dates),color='blue',linewidth=3)
	plt.show()
	return
