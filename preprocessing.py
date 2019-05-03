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


def analyze_missing_data(df):
    '''
    Find variables with missing data and return those in a list.
    Input:
    - df: (pandas dataframe) dataframe of the dataset
    Output:
    - nan_vars: (list) list of the name of the variables with missing values
    '''
    nan = df.isna().sum()
    nan_perc = round(100 * nan / len(df.index), 2)
    nan_df = pd.concat([nan, nan_perc], axis=1)
    nan_df = nan_df.rename(columns={0: 'NaN', 1: 'Percent of NaN'})
    nan_df = nan_df.sort_values(by=['Percent of NaN'], ascending=False)
    only_nan_df = nan_df[nan_df['NaN'] > 0]
    nan_vars = only_nan_df.index.tolist()

    print(only_nan_df)
    print("\n################################################################")
    print("\nThe following features have missing values:\n{}".format(nan_vars))

    message = "\n- {} has {} missing values, which are {}% of the entire data"

    for var in nan_vars:
        num = only_nan_df.loc[var][0]
        perc = only_nan_df.loc[var][1]
        print(message.format(var, int(num), perc))

    return nan_vars


def impute_missing_data(df, columns):
    '''
    Given the dataset, impute missing data for the given variables.
    - df: (pandas dataframe) dataframe of the dataset
    - columns: (list) the name of columns
    Output:
    - Nothing
    '''
    for column in columns:
        if df[column].dtype == 'object':
            estimate = df[column].value_counts().idxmax()
            print("For {}, the most common value is selected.".format(column))
        else:
            if abs(df[column].kurt()) > 3:
                cond = df[column].median()
                print("For {}, median is selected.".format(column))
            else:
                cond = df[column].mean()
                print("For {}, mean is selected.".format(column))
            estimate = round(cond)
        df[column] = df[column].fillna(estimate)

    print("\nImputation completed!")

if __name__ == "__main__":

    try:
        df = read_data(DATA_DIR + DATAFILE)
        print("\nThe data is successfully loaded!\n")
        summarize_data(df)
        print("ATTEMPTING TO PREPROCESS THE DATA\n")

        try:
            clean_name = "clean_" + DATAFILE
            directory = DATA_DIR + clean_name
            df = select_features(df, FEATURES)
            print("The following features have been selected:\n{}".format(FEATURES))
            generate_time_label(df, START, END, OUTCOME)
            print("\nThe label is appended as {}.".format(OUTCOME))
            df = generate_dummy(df, CAT)
            print("Dummy variables are created based on {}.".format(CAT))
            nan_vars = analyze_missing_data(df)
            impute_missing_data(df, nan_vars)
            df = df.reset_index()
            df = df.drop(['projectid', 'datefullyfunded'], axis=1)
            df.to_csv(DATA_DIR + clean_name)
            print("The cleaned data is saved as {}.\n".format(directory))

        except:
            print("Failed to clean and save the dataset.\n")

    except:
        print("Failed to read the data. Please check the filename.")
    