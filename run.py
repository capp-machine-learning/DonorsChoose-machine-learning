#!/usr/bin/env python3
'''
Python main file for the application.
'''
import pandas as pd
import os

from config import load_log_config, log_msg, read_config
from extract_data import read_data, summarize_data
from preprocessing import select_features, generate_time_label
from model import temporal_loop, find_best_model


CONFIG = read_config('config.yaml')
DATA_DIR = CONFIG['data_dir']
DATAFILE = CONFIG['datafile']
DATATYPES = CONFIG['datatypes']
FEATURES = CONFIG['features']
START = CONFIG['time_features']['start']
END = CONFIG['time_features']['end']
OUTCOME = CONFIG['outcome_var']


def main(log=False):
    
    if log:
        if os.path.isfile("./results.log"):
            os.remove("./results.log")
        LOGGER = load_log_config()
        log_msg(LOGGER, "# Deleted the existing log file.")
    
    else:
        LOGGER = None
    
    # Read the full data
    df = read_data(DATA_DIR + DATAFILE, convert=True, logger=LOGGER)
    summarize_data(df, logger=LOGGER)

    # Selecting features and generate the dependent variable
    df = select_features(df, FEATURES, logger=LOGGER)
    generate_time_label(df, [START, END], OUTCOME, LOGGER)

    # Splitting the dataframe (Debugging purpose)
    temporal_sets = temporal_loop(df, clean=True, logger=LOGGER)
    best_models = []

    for date, datasets in temporal_sets:
        
        eval_csv = './evaluations/{}_evaluations.csv'.format(date)
        models = find_best_model(datasets, grid=True, scale=True, save='./evaluations/{}_evaluations.csv'.format(date))
        best_models.append(models)
        log_msg(LOGGER, "\n# The evaluation has been saved as {}.".format(eval_csv))
        log_msg(LOGGER, "\n# The best model for for {} set has been selected.".format(date))
    
    #models = find_best_model(datasets, grid=True, scale=True, save='./evaluations/evaluations.csv')
    
    log_msg(LOGGER, best_models)
    log_msg(LOGGER, "\nJob completed")
    
    return best_models

    #temporal_df = pd.DataFrame(columns=['Training Data','Testing Data'])


#-----------------------------------------------------------------------------#
if __name__ == "__main__":

    main(log=True)
