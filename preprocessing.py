'''
Functions for cleaning the dataset.

Si Young Byun
'''

import pandas as pd

from config import *
from extract_data import *

CONFIG = read_config('config.yaml')
DATA_DIR = CONFIG['data_dir']
DATAFILE = CONFIG['datafile']
DATATYPES = CONFIG['datatypes']
TIME_FORMAT = CONFIG['time_format']
FEATURES = CONFIG['features']
START = CONFIG['time_features']['start']
END = CONFIG['time_features']['end']
OUTCOME = CONFIG['outcome_var']
CAT = CONFIG['feature_types']['categorical']

# Convert to the correct dtypes
def convert_dtypes(df):

    for dtype, features in DATATYPES.items():
        if dtype == "time":
            for feature in features:
                df[feature] = pd.to_datetime(df[feature], format=TIME_FORMAT)
        else:
            for feature in features:
                df[feature] = df[feature].astype(dtype)
    

def select_features(df, features):

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

if __name__ == "__main__":

    try:
        df = read_data(DATA_DIR + DATAFILE)
        convert_dtypes(df)
        print("\nThe data is successfully loaded!\n")
        summarize_data(df)
        print("ATTEMPTING TO PREPROCESS THE DATA")

        try:
            clean_name = "clean_" + DATAFILE
            directory = DATA_DIR + clean_name
            df = select_features(df, FEATURES)
            generate_time_label(df, START, END, OUTCOME)
            print("The label is appended as {}.".format(OUTCOME))
            df = generate_dummy(df, CAT)
            print("Dummy variables are created based on {}.".format(CAT))
            df.to_csv(DATA_DIR + clean_name)
            print("The cleaned data is saved as {}.\n".format(directory))

        except:
            print("Failed to clean and save the dataset.\n")

    except:
        print("Failed to read the data. Please check the filename.")