'''
Functions for reading the data.

Si Young Byun
'''
import os
import pandas as pd

from config import log_msg, read_config

CONFIG = read_config('config.yaml')
DATATYPES = CONFIG['datatypes']

# Reading the data
def convert_dtypes(df, logger=None):

    log_msg(logger, "\n# Converting the data into preconfigured dtypes...")
    
    try:
        for dtype, features in DATATYPES.items():
            if dtype == "time":
                for feature in features:
                    df[feature] = pd.to_datetime(df[feature])
            else:
                for feature in features:
                    df[feature] = df[feature].astype(dtype)
        
        log_msg(logger, "SUCCESS")
    
    except:
        log_msg(logger, "FAILED")


def read_data(filename, convert=False, logger=None):
    '''
    Read a dataset and print a short summary of the data.
    Return a dataframe of the dataset.
    Input:
    - filename: (string) the directory of the dataset
    Output:
    - df: (pandas dataframe) dataframe of the dataset
    '''
    log_msg(logger, "\n# Loading the data...")
    
    try:
        _, ext = os.path.splitext(filename)

        if ext == '.csv':
            df = pd.read_csv(filename, index_col=0)
            log_msg(logger, "SUCCESS")
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filename, index_col=None)
            log_msg(logger, "SUCCESS")

        if convert is True:
            convert_dtypes(df, logger)
        
        return df

    except:
        log_msg(logger, "FAILED")


# Summarize the loaded data
def summarize_data(df, logger=None):

    row = df.shape[0]
    col = df.shape[1]

    partition = "\n#########################################################\n"
    log_msg(logger, partition)
    log_msg(logger, "Summary for the loaded dataset\n")
    log_msg(logger, "The total number of rows is {}\n".format(row))
    log_msg(logger, "The total number of rows is {}\n".format(col))
    log_msg(logger, "Descriptive Statistics:\n\n{}\n".format(df.describe()))
    log_msg(logger, partition)
