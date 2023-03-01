from flask import Flask, render_template, request

import formdata
import joblib
import numpy as np
import pandas as pd
import pipemodel



app   = Flask(__name__, template_folder='template', static_folder='template/assets')



classifier_pipeline, classifier_model = pipemodel.load_classifier()
regressor_pipeline, regressor_model   = pipemodel.load_regressor()
to_predict, time_series_models        = pipemodel.load_time_series()



@app.route('/')
def home():
	return render_template('index.html')



@app.route('/classifier')
def classifier():
	return render_template('htmlmodels/classifier.html')



@app.route('/cluster')
def cluster():
	return render_template('htmlmodels/cluster.html')



@app.route('/formclassifier', methods=['POST'])
def send():
	dataset = formdata.get_data_classif(request)
	ds = classifier_pipeline.transform(dataset)
	churn = classifier_model.predict(ds)
	churn = churn[0] # churn é um numpy array. quero somente o número
	return render_template('htmlmodels/classifier.html', churn=churn, submitted=True)



@app.route('/formregressor', methods=['POST'])
def formregressor():
	dataframe = formdata.get_data_regressor(request)
	df_transformed = regressor_pipeline.transform(dataframe)
	prediction = regressor_model.predict(df_transformed)
	prediction = prediction[0] # prediction é um numpy array. quero somente o número
	return render_template('htmlmodels/regressor.html', number=float(prediction))



@app.route('/formtimeseries', methods=['POST'])
def formtimeseries():
	select_value = request.form.get('my-select')
	days  = int(select_value[0])
	hours = [str(h) + ':00:00' for h in range(0, 24)] # 0:00:00, 1:00:00, ... 23:00:00
	hours.insert(0, 'horas')
	week_day = ['segunda-feira', 'terca-feira', 'quarta-feita', 'quinta-feira', 'sexta-feira', 'sabado', 'domingo']
	predictions = list()
	predictions.append(hours)
	for df, model, day in zip(to_predict[:days], time_series_models[:days], week_day[:days]):
		pdr = [day] # adiciona o dia da semana às previsões. útil para quando retornar a matrix transposta. o nome do dia sera o cabeçalho
		pdr.extend(list(model.predict(df)))
		predictions.append(pdr)
	matrix = list(map(list, zip(*predictions))) # faz a matriz transposta
	return render_template('htmlmodels/timeseries.html', table_data=matrix)



@app.route('/regressor')
def regressor():
	return render_template('htmlmodels/regressor.html')



@app.route('/timeseries')
def time_series():	
	return render_template('htmlmodels/timeseries.html')



if __name__ == '__main__':
	app.run(debug=True)



#bootstrapmade.com/multi-responsive-bootstrap-template
#source .venv/bin/activate
#flask --app webapp.py run
#http://127.0.0.1:5000/

#sudo systemctl restart webportfolio.service