import numpy as np
import pandas as pd



def get_data_classif(request):
	cols_names = ['gender', 'seniorcitizen', 'partner', 'dependents', 'tenure', 'phoneservice', 'multiplelines', 'internetservice', 
	              'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'contract',
	              'paperlessbilling', 'paymentmethod', 'monthlycharges', 'totalcharges']
	values = list()
	for name in cols_names:
		values.append(request.form.get(name))
	dataset = dict()
	for key, value in zip(cols_names, values):
		dataset[key] = value
	dataset = pd.DataFrame(data=dataset, index=range(1))
	
	if dataset['tenure'].values == '':
		dataset['tenure'] = 0
	if dataset['monthlycharges'].values == '':
		dataset['monthlycharges'] = 0
	if dataset['totalcharges'].values == '':
		dataset['totalcharges'] = 0
	return dataset



def get_data_regressor(request):
	cols_names = ['model', 'year', 'transmission', 'mileage', 'fueltype', 'tax', 'mpg', 'enginesize']
	df = form_dataframe(request, cols_names)
	df['year']       = df['year'].astype('int64')
	df['mileage']    = df['mileage'].astype('int64')
	df['tax']        = df['tax'].astype('float64')
	df['mpg']        = df['mpg'].astype('float64')
	df['enginesize'] = df['enginesize'].astype('float64')

	if df['year'].values < 2011:
		df['year'] = 2011
	elif df['year'].values > 2023:
		df['year'] = 2023
	
	if df['mileage'].values < 1:
		df['mileage'] = 1
	elif df['mileage'].values > 80_000:
		df['mileage'] = 80_000

	if df['mpg'].values < 10.0:
		df['mpg'] = 10.0
	elif df['mpg'].values > 100.0:
		df['mpg'] = 100.0

	if df['tax'].values < 0.0:
		df['tax'] = 0.0
	elif df['tax'].values > 300.0:
		df['tax'] = 300.0
	return df

	

def form_dataframe(request, cols_names):
	values = list()
	for name in cols_names:
		values.append(request.form.get(name))
	dataset = dict()
	for key, value in zip(cols_names, values):
		dataset[key] = value
	dataset = pd.DataFrame(data=dataset, index=range(1))
	return dataset