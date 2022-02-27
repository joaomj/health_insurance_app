import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

class HealthInsurance:
    def __init__(self):
        
        path = str(Path.cwd().parents[0])
        scalers_path = path + '/features'
        
        self.annual_premium_scaler =                    pickle.load(open(scalers_path + '/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler =                               pickle.load(open(scalers_path + '/age_scaler.pkl', 'rb'))
        self.vintage_scaler =                           pickle.load(open(scalers_path + '/vintage_scaler.pkl', 'rb'))
        self.target_encoder_gender_scaler =             pickle.load(open(scalers_path + '/target_encoder_gender_scaler.pkl', 'rb'))
        self.target_encoder_region_code_scaler =        pickle.load(open(scalers_path + '/target_encoder_region_code_scaler.pkl', 'rb'))
        self.fe_policy_sales_channel_scaler =           pickle.load(open(scalers_path + '/frequency_encoder_policy_sales_scaler.pkl', 'rb'))

    # =================================    
    # one class for each CRISP-DM step

    def data_cleaning(self, data):
        pass
        return data
    
    def feature_engineering(self, data):
        # ONLY RUN ONCE

        """vehicle_age:
           <1 : 'New'
           1-2 : 'Average'
           >2 : 'Used' """
           
        data['vehicle_age'] = data['vehicle_age'].apply(lambda x: 'New' if x == '< 1 Year' else
                                                    'Average' if x == '1-2 Year' else
                                                    'Used')


        return data
    
    def data_preparation(self, data):
        
        # annual_premium - StandardScaler
        data['annual_premium'] = self.annual_premium_scaler.transform(data[['annual_premium']].values) # encoder receives an array

        # age - MinMaxScaler
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # vintage - MinMaxScaler
        data['vintage'] = self.vintage_scaler.transform(data[['vintage']].values)

        # gender - Target Encoder
        data.loc[:,'gender'] = data['gender'].map(self.target_encoder_gender_scaler)

        # region_code - Target Encoder
        data.loc[:,'region_code'] = data['region_code'].map(self.target_encoder_region_code_scaler)

        # vehicle_age - One Hot Encoder
        data = pd.get_dummies(data, prefix = 'vehicle_age', columns = ['vehicle_age'])

        # policy_sales_channel - Frequency Encoder
        # Frequency Encoding gives more weight to more frequent values of the categorical attribute
        data.loc[:,'policy_sales_channel'] = data['policy_sales_channel'].map(self.fe_policy_sales_channel_scaler) 

        # Features Selection (arbitrary)
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage', 'policy_sales_channel', 'previously_insured']

        return data[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # model prediction
        pred = model.predict_proba(test_data)

        # join prediction with original data
        original_data.rename(columns = {'response':'score'}, inplace = True)
        original_data['score'] = pred

        # return json to be used by API
        return original_data.to_json(orient = 'records', date_format = 'iso')