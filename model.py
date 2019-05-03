'''
Functions for building and evaluating the dataset.

Si Young Byun
'''

import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve

from config import *

CONFIG = read_config('config.yaml')
OUTCOME = CONFIG['outcome_var']
TEST_SIZE = CONFIG['test_size']
RANDOM_STATE = CONFIG['random_state']


def split_set(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    '''
    Given the dataset, split the dataset into the outcome set and feature set.
    Then, split those sets into the test set and the train set.
    - df: (pandas dataframe) dataframe of the dataset
    - test_size: (float) size of the test set
    - rand: (int) number for random state
    Output:
    - X_train, X_test, y_train, y_test: (pandas dataframe) test/train sets
    '''
    X_df = df.drop(labels=[OUTCOME], axis=1)
    y_df = df[OUTCOME]
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def temporal_train_test_split(df, time_col, period=None):
    '''
    Creates temporal train/test splits.
    Inputs: 
        df
        time_col (str): name of time column
        period (str of the format '1M', '3M', '6M', etc.): how long the 
            testing period should be
    Returns:
        test_dfs: list of test dfs
        train_dfs: list of training dfs
        times: list of start times indexed to test and train lists
    '''
    time_starts = pd.date_range(start = df[time_col].min(), 
                                end = df[time_col].max(), freq = period)
    times = []
    train_sets = []
    test_sets = []

    for i, time in enumerate(time_starts[:-1]):
        time_split = time + pd.DateOffset(1)
        train_mask = (df[time_col] <= time)
        test_mask = (df[time_col] > time_split) & (df[time_col] < time_starts[i+1])
        
        train = df.loc[train_mask]
        train = train.drop(time_col, axis=1)
        test = df.loc[test_mask]
        test = test.drop(time_col, axis=1)

        train_sets.append(train)
        test_sets.append(test)
        times.append(time)
        
    return train_sets, test_sets, times


def train_model(X_train, y_train, clf, grid=None):

    clfs = {
        'LR': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'DT': DecisionTreeClassifier(),
        'SVM': LinearSVC(),
        'RF': RandomForestClassifier(),
        'GB': GradientBoostingClassifier()}
    
    model = clfs[clf]

    if grid:
        params = grid[clf]
        model.set_params(**params)

    return model.fit(X_train, y_train)

