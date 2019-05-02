'''
Functions for cleaning the dataset.

Si Young Byun
'''

import pandas as pd
from pipeline import read_config

config = read_config('config.yaml')
datatypes = config['datatypes']
time_format = config['time_format']
features = config['features']

def convert_dtypes(df):

    for dtype, features in datatypes.items():
        if dtype == "time":
            for feature in features:
                df[feature] = pd.to_datetime(df[feature], format=time_format)
        else:
            for feature in features:
                df[feature] = df[feature].astype(dtype)
    

def select_features(df):

    df = df[features]

    return df


def generate_time_label(df, start, end, label):
	
    time_int = df[end] - df[start]
    df[label] = (time_int <= pd.to_timedelta(60, unit='days')).astype('int')


def generate_dummy(df, features):
    '''
    Given the dataset and the variable, generate dummy variables for it.
    - df: (pandas dataframe) dataframe of the dataset
    - variable: (str) the name of the variable
    Output:
    - Nothing
    '''

    for feature in features:
        dummy = pd.get_dummies(df[feature])
        df = pd.concat([df, dummy], axis=1)
        df = df.drop(columns=[feature])

    return df