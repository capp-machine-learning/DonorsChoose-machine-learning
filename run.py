'''
Python main file for the application.
'''
import pandas as pd
import os

from config import load_log_config, log_msg, read_config
from extract_data import read_data, summarize_data
from preprocessing import select_features, generate_time_label
from model import temporal_loop


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
    start_dates = ['2012-07-01', '2013-01-01', '2013-07-01']
    temporal_sets = temporal_loop(df, 'date_posted', start_dates, 6)

    for date, datasets in temporal_sets:
        (X_train, X_test, y_train, y_test) = datasets
        
        log_msg(LOGGER, "# The data for {} is saved.\n".format(date))
    
    return temporal_sets

    #temporal_df = pd.DataFrame(columns=['Training Data','Testing Data'])

    

#-----------------------------------------------------------------------------#
if __name__ == "__main__":

    main(log=True)
