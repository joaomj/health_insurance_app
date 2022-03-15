import os
import pickle
import pandas as pd
import sys
import boto3
from pathlib import Path
from flask import Flask, request, Response
from decouple import config
from healthinsurance.HealthInsurance import HealthInsurance

# loading model
""" Its much better to preload model on memory to reduce API response time"""
# path = str(Path.cwd().parents[0])
# path = path + '/models'
# model = pickle.load(open(path + '/model_knn.pkl', 'rb')) # where the model is stored
# model = pickle.load(open('models/model_knn.pkl', 'rb')) # where the model is stored


# open model stored on a S3 Bucket using CONFIG VARS from Heroku:
# s3client = boto3.client('s3',
#                         aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'], 
#                         aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
#                         )


# response = s3client.get_object(Bucket = os.environ['AWS_BUCKET'], Key = os.environ['AWS_MODEL_FILEPATH'])

# body = response['Body'].read()
# model = pickle.loads(body)

# =====================================================
# solution from
# https://towardsdatascience.com/how-to-load-data-from-a-pickle-file-in-s3-using-python-ffe2866b7eba

# import boto3.session

# cred = boto3.Session().get_credentials()
# ACCESS_KEY = cred.access_key
# SECRET_KEY = cred.secret_key
# s3client = boto3.client('s3',
#                         aws_access_key_id = ACCESS_KEY, 
#                         aws_secret_access_key = SECRET_KEY
#                         )

# bucket = 'my_bucket_name'
# key = 'obj_key'

# response = s3client.get_object(Bucket = bucket, Key = key)
# body = response['Body'].read()
# model = pickle.loads(body)
# =====================================================

# =====================================================
# my adaptation
ACCESS_KEY = config('ACCESS_KEY')
SECRET_KEY = config('SECRET_KEY')
BUCKET = config('BUCKET')
OBJ_KEY = config('OBJ_KEY')

s3client = boto3.client('s3',
                        aws_access_key_id = ACCESS_KEY, 
                        aws_secret_access_key = SECRET_KEY
                        )

response = s3client.get_object(Bucket = BUCKET, Key = OBJ_KEY)
body = response['Body'].read()
model = pickle.loads(body)
# =====================================================



# initialize API
app = Flask(__name__)

# creating endpoint
@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()

    if test_json: # if there is data
        if isinstance(test_json, dict): # if json is dictionary, then is unique data
            test_raw = pd.DataFrame(test_json, index=[0])
        
        else: # multiple example
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys()) # if json has multiple lines, then each json key will be dataframe's columns names


        # instantiate HealthInsurance class
        pipeline = HealthInsurance()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response
    
    else:
        return Response('No data was sent to the API', status = 200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port = port, debug = True)
    # app.run(host = '0.0.0.0', port = port)