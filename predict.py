import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dates = []
prices = []



def get_data(filename):
	with open(filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(row[0])
			prices.append(float(row[1]))
	#print(dates)
	#print(prices)
	return

def show_plot(dates,prices):
	linear_mod = linear_model.LinearRegression()
	dates = np.reshape(dates, (len(dates),1))
	prices = np.reshape(prices, (len(prices),1))
	linear_mod.fit(dates,prices)
	plt.scatter(dates,prices,color='yellow')
	plt.plot(dates,linear_mod.predict(dates),color='blue',linewidth=3)
	plt.show()
	return

def predict_price(dates,prices,x):
	linear_mod = linear_model.LinearRegression()
	dates = np.reshape(dates, (len(dates),1))
	prices = np.reshape(prices, (len(prices),1))
	linear_mod.fit(dates,prices)
	predicted_price = linear_mod.predict(x)
	return predicted_price[0][0],linear_mod.coef_[0][0], linear_mod.intercept_[0]


get_data('TSLA.csv')
print(prices)
print(dates)

show_plot(dates,prices)