from os import listdir
from os.path import isfile, join

import joblib
import os
import pandas as pd
import sys

modules_path = os.path.abspath(os.path.join('models/classifier'))
if modules_path not in sys.path:
    sys.path.append(modules_path)



def load_classifier():
	full_pipeline = joblib.load('models/classifier/classif_pipeline.pkl.z')
	model         = joblib.load('models/classifier/classif_model.pkl.z')
	return full_pipeline, model



def load_regressor():
	pipeline_full = joblib.load('models/regressor/regressor_pipeline.pkl.z')
	model         = joblib.load('models/regressor/regressor_model.pkl.z')
	return pipeline_full, model



def load_time_series():
	files_names      = listdir('models/time-series')
	models_names     = list()
	to_predict_names = list()
	with open('files.txt', 'w') as file:
		for fn in files_names:
			if fn.endswith('csv'):
				file.write('models/time-series' + '/' + fn + '\n')
				to_predict_names.append(fn)
			else:
				file.write('models/time-series' + '/' + fn + '\n')
				models_names.append(fn)
	models     = [joblib.load('models/time-series' + '/' + fn) for fn in models_names]
	to_predict = [pd.read_csv('models/time-series' + '/' + fn, index_col='Unnamed: 0', parse_dates=True) for fn in to_predict_names]
	return to_predict, models