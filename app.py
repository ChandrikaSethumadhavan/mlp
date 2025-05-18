import pickle  #same code from app.py, put here so that it matches the "application" name in python config file under ebextensions
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__) 
app = application

#route to render the home page
@app.route('/')
def index():
    return render_template('index.html') #seasrch for index.html in templates folder in the project directory

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    #get the data from the form
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'), race_ethnicity= request.form.get('ethnicity'), parental_level_of_education = request.form.get('parental_level_of_education'), lunch = request.form.get('lunch'), test_preparation_course = request.form.get('test_preparation_course'), reading_score = float(request.form.get('reading_score')), writing_score = float(request.form.get('writing_score'))

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0]) #render the home.html file and pass the results to it
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000) #run the app 
 #debug =True has been removed to avoid the error of "werkzeug" not being able to be imported and it is not needed in the production environment
