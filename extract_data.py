'''
Functions for reading the data.

Si Young Byun
'''
import os
import pandas as pd

from config import *

CONFIG = read_config('config.yaml')
DATATYPES = CONFIG['datatypes']

# Reading the data
def convert_dtypes(df):

    for dtype, features in DATATYPES.items():
        if dtype == "time":
            for feature in features:
                df[feature] = pd.to_datetime(df[feature])
        else:
            for feature in features:
                df[feature] = df[feature].astype(dtype)


def read_data(filename, convert=False):
    '''
    Read a dataset and print a short summary of the data.
    Return a dataframe of the dataset.
    Input:
    - filename: (string) the directory of the dataset
    Output:
    - df: (pandas dataframe) dataframe of the dataset
    '''
    _, ext = os.path.splitext(filename)

    if ext == '.csv':
        df = pd.read_csv(filename, index_col=0)
    elif ext == '.xls':
        df = pd.read_excel(filename, header=1)
    
    if convert is True:
        convert_dtypes(df)

    return df


# Summarize the loaded data
def summarize_data(df):

    print("################################################################\n")
    print("Summary for the loaded dataset\n")
    print("Data Shape: {}\n".format(df.shape))
    print("Descritive Statistics:\n\n{}\n".format(df.describe()))
    print("################################################################\n")
