'''
Functions for building and evaluating the dataset.

Si Young Byun
'''

from sklearn.model_selection import train_test_split

from config import *

CONFIG = read_config('config.yaml')
OUTCOME = CONFIG['outcome_var']
TEST_SIZE = CONFIG['test_size']
RANDOM_STATE = CONFIG['random_state']

def create_X_y_set(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    
    Given the dataset, split the dataset into the outcome set and feature set.
    Then, split those sets into the test set and the train set.
    - df: (pandas dataframe) dataframe of the dataset
    - test_size: (float) size of the test set
    - rand: (int) number for random state
    Output:
    - X_train, X_test, y_train, y_test: (pandas dataframe) test/train sets
    
    X_df = df.drop(labels=[OUTCOME], axis=1)
    y_df = df[OUTCOME]
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test