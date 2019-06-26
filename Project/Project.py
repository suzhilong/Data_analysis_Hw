#reference: https://www.jianshu.com/p/cced6617b423
#copyright @suzhilong
#2019.6.25

# -*- coding:utf-8 -*-

import codecs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import AR,ARIMA

def get_data():
	path = './IBM.csv'
	reader = codecs.open(path, 'r')
	lines = reader.readlines()
	data = []
	for line in lines[1:]:
		data.append(float(line.split(',')[1].strip()))
	return data

def plot_data1(data, figTitle='', xlab='', ylab=''):
	x = np.arange(0,len(data))
	plt.title(figTitle)
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.plot(x, data)
	plt.show()

def get_diff(data,n):
	diff = []
	for i in range(len(data)-n):
		diff.append(data[i+n]-data[i])
	return diff


def get_AR1(diff, flag='MLE'):
	'''
	MLE: A * [a0,a1]T = B
	MAP: 
	'''
	diffArr = np.array(diff)

	if flag == 'MLE':
		A = [[len(diffArr)-1, sum(diffArr[:-1])],
			 [sum(diffArr[:-2]), sum(diffArr[:-1]*diffArr[:-1])]]
		B = [sum(diffArr[1:]), sum(diffArr[:-1]*diffArr[1:])]
		a = np.linalg.solve(A,B)
		print 'MLE: a0,a1=',a
		return a
	if flag == 'MAP':
		pass

def main_self():
	data = get_data()
	#plot_data1(data,"stock prices",'date','price')

	diff1 = get_diff(data,1)
	plot_data1(diff1,figTitle='1 order diff')

	a = get_AR1(diff1,'MLE')

	# for i in range(1,6):
	# 	model = AR(diff1)
	# 	print '%d order AR models:'%(i),model.fit(i).params

	model = AR(diff1)
	result_AR1 = model.fit(1)

	fig_data = plt.plot(np.arange(0,len(diff1)),diff1,color='blue',label='data')
	fig_pre = plt.plot(result_AR1.fittedvalues, color='red',label='predict')
	plt.legend(loc='best')
	plt.title('diff1 of data & predict')

	plt.show()

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
		diff = get_diff(data,1)
		model = ARIMA(data, order=(i, 1, 0))
		results_AR = model.fit(disp=-1)
		#print results_AR.params
		pre_diff = pd.Series(results_AR.fittedvalues, copy=True)
		pre_diff_cumsum = pre_diff.cumsum()
		pre_data = data.add(pre_diff_cumsum,fill_value=0)

		error = get_fitting_error(data,pre_data)
		errors.append(error)

		# plot_data(diff,pre_diff,'diff','pre_diff','%d order diff'%(i))
		# plot_data(data,pre_data,'data','pre_data','%d order AR model'%(i))
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

	errors,best_order = get_best_order(data,10)
	# best_order = 8
	#plot fitting errors
	plot_error(errors)

	print 'best order:',best_order
	model = ARIMA(data, order=(best_order, 1, 0))
	results_AR = model.fit(disp=-1)
	print results_AR.params
	pre_diff = pd.Series(results_AR.fittedvalues, copy=True)
	pre_diff_cumsum = pre_diff.cumsum()
	pre_data = data.add(pre_diff_cumsum,fill_value=0)
	# plot_data(data,pre_data,'%d order AR model'%(best_order),'data','pre_data')
	fitting_errors = data-pre_data
	plt.hist(fitting_errors, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
	plt.xlabel("error value")
	plt.ylabel("numbers")
	plt.title("histogram of error")
	plt.show()

if __name__ == '__main__':
	main()
	# main_self()
