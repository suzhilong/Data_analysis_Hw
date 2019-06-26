#python 2.7
#Hw of Data analysis, part1
#author: Su Zhilong
#2019.6.26
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

def get_diff(data,n):
	diff = []
	for i in range(len(data)-n):
		diff.append(data[i+n]-data[i])
	return diff

def plot_data(data,pre_data,title='',data_lab='',pre_data_lab=''):
	plt.plot(data,color='blue',label=data_lab)
	plt.plot(pre_data, color='red',label=pre_data_lab)
	plt.legend(loc='best')
	plt.title(title)
	plt.show()

def get_fitting_error(data,pre_data):
	return sum(abs(data-pre_data))

def get_best_order(data,max_order):
	errors = []
	for i in range(1,max_order+1):
		print '%d order AR model'%(i)  
		diff = get_diff(data,i)
		model = ARIMA(data, order=(i, 1, 0))
		results_AR = model.fit(disp=-1)
		#print results_AR.params
		pre_diff = pd.Series(results_AR.fittedvalues, copy=True)
		pre_diff_cumsum = pre_diff.cumsum()
		pre_data = data.add(pre_diff_cumsum,fill_value=0)

		error = get_fitting_error(data,pre_data)
		errors.append(error)

	return errors,errors.index(min(errors))+1

def plot_error(errors):
	x = np.linspace(1, len(errors), len(errors))
	plt.plot(x,errors)
	plt.xlabel('order')
	plt.title('fitting errors')
	plt.ylabel('fitting errors')
	plt.show()

def main():
	data = pd.read_csv('IBM.csv')
	data = data['High']

	diff = get_diff(data,1)

	errors,best_order = get_best_order(data,10)
	# best_order = 8

	print 'best order:',best_order
	model = ARIMA(data, order=(best_order, 1, 0))
	results_AR = model.fit(disp=-1)
	print 'parameters of %d order AR model:'%(best_order)
	print results_AR.params
	pre_diff = pd.Series(results_AR.fittedvalues, copy=True)
	# pre_diff_cumsum = pre_diff.cumsum()
	pre_data = data.add(pre_diff,fill_value=0)
#plot
	#plot diff
	plot_data(diff,pre_diff,'diff','pre_diff','first order diff')
	#plot fitting errors
	plot_error(errors)
	#plot data and pre_data
	plot_data(data,pre_data,'%d order AR model'%(best_order),'data','pre_data')
	fitting_errors = data-pre_data
	plt.hist(fitting_errors, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
	plt.xlabel("error value")
	plt.ylabel("numbers")
	plt.title("histogram of error")
	plt.show()

if __name__ == '__main__':
	main()