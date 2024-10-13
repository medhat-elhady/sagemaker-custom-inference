from flask import Flask
import flask
from prophet import Prophet
import pandas
import os
import json
import logging

#Load in model

#If you plan to use a your own model artifacts,
#your model artifacts should be stored in /opt/ml/model/
# Step 4: Load the saved model
with open('/opt/ml/model/prophet_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# The flask app for serving predictions
app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    health = loaded_model is not None
    status = 200 if health else 404
    return flask.Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
   
    #Process input
    input_json = flask.request.get_json()
    dates = input_json['input']

    df = pd.DataFrame({'ds': dates})
   
    forecast = loaded_model.predit(df)

    # Transform predictions to JSON
    result = {
        'output': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].values
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')
