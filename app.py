import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn import metrics
from flask import Flask, request, render_template
import pickle
import joblib

app = Flask("__name__")


q = ""


@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/predict", methods=['POST'])
def predict():

    inputQuery1 = str(request.form['query1'])
    inputQuery2 = str(request.form['query2'])
    inputQuery3 = str(request.form['query3'])
    inputQuery4 = int(request.form['query4'])
    inputQuery5 = str(request.form['query5'])
    inputQuery6 = str(request.form['query6'])
    inputQuery7 = int(request.form['query7'])
    inputQuery8 = int(request.form['query8'])
    inputQuery9 = str(request.form['query9'])
    inputQuery10 = str(request.form['query10'])
    
    od_pair = str(inputQuery5) + " " + str(inputQuery6)
    
    if (int(inputQuery4) > 250):
        greater_than_250 = True
    else:
        greater_than_250 = False
        
    if (int(inputQuery4) > 0) & (int(inputQuery4) <= 99):
          distance_band =  'CTRI'
    elif (int(inputQuery4) >= 100) & (int(inputQuery4) <= 250):
      distance_band =  'STRI'
    elif (int(inputQuery4) >= 251) & (int(inputQuery4) <= 450):
      distance_band = 'MTRI'
    elif (int(inputQuery4) >= 451) & (int(inputQuery4) <= 850):
      distance_band = 'TTRI'
    else:
      distance_band = 'LTRI'


    date = pd.to_datetime(inputQuery10)
    day_of_week = date.dayofweek
    week = date.week
    day_of_month = date.day
    month_of_year = date.month
    date_quarter = date.quarter
    feature_weight = 1
     
     
    #model = pickle.load(open("freight_chg_prediction_June.sav", "rb"))
    model = joblib.load('reduced_trained_model.pkl') 
    
    #to deploy
    file = open("reduced_enc_freight_prediction.obj", 'rb')
    enc_loaded = pickle.load(file)
    dummy = 0

    

    data = [[inputQuery1, inputQuery2, inputQuery3,
             inputQuery4, inputQuery5, inputQuery6, inputQuery7, inputQuery8, dummy, od_pair, inputQuery9, 
             greater_than_250, distance_band, day_of_week, week, day_of_month, month_of_year, date_quarter, feature_weight]]
    
    new_df = pd.DataFrame(data, columns=['commodity', 'exchange_type', 'scac', 'pay_distance', 'dest_market_id', 'origin_market_id', 'pieces',
                                         'weight', 'freight_chg',	'OD_Pair',	'customer_type', 'greater_than_250', 'distance_band', 'day_of_week',
                                         'week',	'day_of_month',	'month_of_year', 'date_quarter', 'feature_weight'])
    


    encoded_df = enc_loaded.transform(new_df)

    
    #encoded_df = encoded_df[encoded_df >= 0]
    
    final_df = encoded_df.drop(['freight_chg'], axis=1)
    
    single = model.predict(final_df)
    prediction = np.expm1(single)
    

    return render_template('home.html', output1=prediction,  query1=request.form['query1'], query2=request.form['query2'], query3=request.form['query3'], query4=request.form['query4'], query5=request.form['query5'],
                          query6=request.form['query6'], query7=request.form[
                               'query7'], query8=request.form['query8'], query9=request.form['query9'], query10=request.form['query10'])
if __name__ == "__main__":
    app.run()
