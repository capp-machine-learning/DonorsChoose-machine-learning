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
from viz import plot_precision_recall_curve


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
        
        (X_train, X_test, y_train, y_test) = datasets
        dts = (X_train.values.astype(float),
               X_test.values.astype(float),
               y_train,
               y_test)
        count = 0
        eval_csv = './evaluations/{}_evaluations.csv'.format(date)
        models = find_best_model(dts, grid=True, scale=True, save=eval_csv)
        
        for mdl in models:
            save1 = './images/{}_{}_prc.png'.format(date, count)
            plot_precision_recall_curve(datasets[1], datasets[3], mdl, save1)
            msg2 = "\n# A Precision Recall curve has been saved as {}."
            log_msg(LOGGER, msg2.format(save1))
            count += 1

        best_models.append(models)
        msg3 = "\n# The evaluation has been saved as {}."
        msg4 = "\n# The best model for {} set has been selected.\n"
        log_msg(LOGGER, msg3.format(eval_csv))
        log_msg(LOGGER, msg4.format(date))
    
    log_msg(LOGGER, best_models)
    log_msg(LOGGER, "\nJob completed")
    
    return best_models



#-----------------------------------------------------------------------------#

if __name__ == "__main__":

    main(log=True)
