import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

dates = []
prices = []


def get_data_pd(filename):
	pd.read_csv(filename)
	df.set_index('Date', inplace=True)
	df.head()
	return



def get_data(filename):
	with open(filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0]))
			prices.append(float(row[5]))
	#print(dates)
	#print(prices)
	return

def show_plot(dates,prices):
	linear_mod = linear_model.LinearRegression()
	reg = linear_model.Ridge(alpha=.4)

	dates = np.reshape(dates, (len(dates),1))
	prices = np.reshape(prices, (len(prices),1))

	linear_mod.fit(dates,prices)
	reg.fit(dates,prices)
	
	plt.scatter(dates,prices,color='red')
	plt.plot(dates,linear_mod.predict(dates),color='blue',linewidth=1)
	plt.plot(dates,reg.predict(dates),color='green',linewidth=1)
	plt.show()
	return

def predict_price(dates,prices,x):
	linear_mod = linear_model.LinearRegression()
	dates = np.reshape(dates, (len(dates),1))
	prices = np.reshape(prices, (len(prices),1))
	linear_mod.fit(dates,prices)
	predicted_price = linear_mod.predict([[x]])
	return predicted_price[0][0],linear_mod.coef_[0][0], linear_mod.intercept_[0]


get_data('TSLA.csv')
print(prices)
print(dates)

show_plot(dates,prices)

predicted_price, coefficient, constant = predict_price(dates,prices,31)

print(predicted_price)
print(coefficient)
print(constant)