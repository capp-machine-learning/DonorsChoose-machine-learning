'''
Functions for cleaning the dataset.

Si Young Byun
'''

import pandas as pd
#import sys

from config import load_log_config, log_msg, read_config
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


def select_features(df, features, logger=None):

    msg = "\n# Attempting to SELECT the following features:\n    {}..."
    log_msg(logger, msg.format(features))
    try:
        df = df[features]
        log_msg(logger, "SUCCESS")

    except:
        log_msg(logger, "FAILED")   

    return df


def generate_time_label(df, time_cols, dep_var, logger=None):
	
    msg = "\n# Attempting to GENERATE a label called '{}' based on {}..."
    log_msg(logger, msg.format(dep_var, time_cols))

    try:
        (start, end) = time_cols
        time_int = df[end] - df[start]
        df[dep_var] = (time_int <= pd.to_timedelta(60, unit='days')).astype('int')
        log_msg(logger, "SUCCESS")

    except:
        log_msg(logger, "FAILED")


def generate_dummy(df, features, logger=None):
    '''
    Given the dataset and the variable, generate dummy variables for it.
    - df: (pandas dataframe) dataframe of the dataset
    - variable: (str) the name of the variable
    Output:
    - Nothing
    '''
    msg = "\n# For the following variables, dummy variables have been created."
    log_msg(logger, msg)
    log_msg(logger, "    >>>{}".format(features))

    try:
        for feature in features:
            dummy = pd.get_dummies(df[feature], prefix=feature)
            df = pd.concat([df, dummy], axis=1)
            df = df.drop(columns=[feature])
        log_msg(logger, "SUCCESS")
        return df
    
    except:
        log_msg(logger, "FAILED")
        return None


def analyze_missing_data(df, logger=None):
    '''
    Find variables with missing data and return those in a list.
    Input:
    - df: (pandas dataframe) dataframe of the dataset
    Output:
    - nan_vars: (list) list of the name of the variables with missing values
    '''

    msg = "\n# Attempting to analyze the missing data..."
    log_msg(logger, msg)
    
    try:
        log_msg(logger, "SUCCESS")
        nan = df.isna().sum()
        nan_perc = round(100 * nan / len(df.index), 2)
        nan_df = pd.concat([nan, nan_perc], axis=1)
        nan_df = nan_df.rename(columns={0: 'NaN', 1: 'Percent of NaN'})
        nan_df = nan_df.sort_values(by=['Percent of NaN'], ascending=False)
        only_nan_df = nan_df[nan_df['NaN'] > 0]
        nan_vars = only_nan_df.index.tolist()
        #logging
        msg1 = "\nThe following features have missing values:\n    >>>{}\n"
        log_msg(logger, msg1.format(nan_vars))
        msg2 = "- {} has {} missing values, which are {}% of the data"

        for var in nan_vars:
            num = only_nan_df.loc[var][0]
            perc = only_nan_df.loc[var][1]
            log_msg(logger, msg2.format(var, int(num), perc))

        return nan_vars

    except:
        log_msg(logger, "FAILED")


def impute_missing_data(df, columns, logger=None):
    '''
    Given the dataset, impute missing data for the given variables.
    - df: (pandas dataframe) dataframe of the dataset
    - columns: (list) the name of columns
    Output:
    - Nothing
    '''
    msg = "\n# Attempting to impute the missing data...\n"
    log_msg(logger, msg)

    try:
        for col in columns:
            if df[col].dtype == 'object':
                estimate = df[col].value_counts().idxmax()
                msg = "- For {}, the most common value is selected."
            else:
                if abs(df[col].kurt()) > 3:
                    cond = df[col].median()
                    msg = "- For {}, median is selected."
                else:
                    cond = df[col].mean()
                    msg = "- For {}, mean is selected."
                estimate = round(cond)
            log_msg(logger, msg.format(col))
            df[col] = df[col].fillna(estimate)
        
        log_msg(logger, "SUCCESS")
    
    except:
        log_msg(logger, "FAILED")


def find_region(string):
    
    if isinstance(string, str):
        if string in ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']:
            return 'Northeast'

        elif string in ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO',
                        'NE', 'SD', 'ND']:
            return 'Midwest'

        elif string in ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV',
                        'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX']:
            return 'South'

        elif string in ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK',
                        'CA', 'HI', 'OR', 'WA']:
            return 'West'

        else:
            return 'others'


def find_gender(string):
    
    if isinstance(string, str):
        if string in ['Mr.']:
            return 'male'

        elif string in ['Mrs.', 'Ms.']:
            return 'female'

        else:
            return 'others'


def apply_using_func(df, variable, function, logger=None):

    log_msg(logger, "\n# Discretizing {}...".format(variable))

    try:
        df[variable] = df[variable].apply(function)
        log_msg(logger, "SUCCESS")
    
    except:
        log_msg(logger, "FAILED")


#-----------------------------------------------------------------------------#
if __name__ == "__main__":

    if os.path.isfile("./results.log"):
        os.remove("./results.log")
    
    LOGGER = load_log_config()
    log_msg(LOGGER, "# Deleted the existing log file.")

    df = read_data(DATA_DIR + DATAFILE, convert=True, logger=LOGGER)
    summarize_data(df, logger=LOGGER)

    clean_name = "clean_" + DATAFILE
    directory = DATA_DIR + clean_name

    #Preprocessing the data
    df = select_features(df, FEATURES, LOGGER)
    generate_time_label(df, [START, END], OUTCOME, LOGGER)
    apply_using_func(df, 'teacher_prefix', find_gender, LOGGER)
    apply_using_func(df, 'school_state', find_region, LOGGER)
    df = generate_dummy(df, CAT, LOGGER)

    #Missing Data and Imputation
    nan_vars = analyze_missing_data(df, LOGGER)
    impute_missing_data(df, nan_vars, LOGGER)

    df = df.reset_index()
    df = df.drop(['projectid', 'datefullyfunded'], axis=1)
    df.to_csv(DATA_DIR + clean_name, index=False)
    log_msg(LOGGER, "\nThe cleaned data is saved as {}.\n".format(directory))
