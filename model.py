'''
Functions for building and evaluating the dataset.

Si Young Byun
'''

import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

from config import log_msg, read_config
from preprocessing import apply_func, find_gender, find_region, analyze_missing_data, impute_missing_data, generate_dummy, preprocess

CONFIG = read_config('config.yaml')
OUTCOME = CONFIG['outcome_var']
CAT = CONFIG['feature_types']['categorical']
DATA_START_DATE = CONFIG['data_start_date']
START_DATES = CONFIG['start_dates']
UNIT, GAP = CONFIG['test_train_gap']
MONTH = CONFIG['test_month_size']
TIME_COL = CONFIG['time_column']
TEST_SIZE = CONFIG['test_size']
RANDOM_STATE = CONFIG['random_state']
THOLD = CONFIG['threshold']
PR_THOLD = CONFIG['PR_thold']
GRID = CONFIG['parameters']
MODELS = CONFIG['models']
TUNING = CONFIG['tuning']
METRICS = CONFIG['metrics']
FINAL = CONFIG['final_metric']


def joint_sort_descending(l1, l2):

    idx = np.argsort(l1)[::-1]

    return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):

    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]

    return predictions_binary


def precision_at_k(y_true, y_scores, k):

    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision


def recall_at_k(y_true, y_scores, k):

    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


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


def temporal_split(df, start_date, clean=False, logger=None):
    '''
    
    '''
    start_date = pd.to_datetime(start_date)
    end_date = start_date + pd.DateOffset(months=MONTH)
    train_start = start_date - pd.to_timedelta(GAP, unit=UNIT)

    partition = "\n#########################################################\n"
    log_msg(logger, partition)
    msg = """# CREATING train/test sets with:
        - TRAIN SET: {} - {},
        - TEST SET: {} - {}"""
    log_msg(logger, msg.format(DATA_START_DATE,
                               train_start - pd.to_timedelta(1, unit='days'),
                               start_date,
                               end_date - pd.to_timedelta(1, unit='days')))

    train = df[df[TIME_COL] < train_start]
    test = df[(df[TIME_COL] >= start_date) & (df[TIME_COL] < end_date)]

    y_train = train[OUTCOME]
    y_test = test[OUTCOME]
    X_train = train.drop(OUTCOME, axis=1)
    X_test = test.drop(OUTCOME, axis=1)

    #Preprocessing the data
    if clean:
        X_train, X_test = preprocess(df, X_train, X_test, logger)
        X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    log_msg(logger, partition)

    return (X_train, X_test, y_train, y_test)


def temporal_loop(df, clean=False, logger=None):

    msg = "\n# Attempting to SPLIT the data temporally..."
    log_msg(logger, msg)
    msg1 = "# with each test set starting on the following dates:\n{}\n"
    log_msg(logger, msg1.format(START_DATES))

    temporal_sets = []
    for start_date in START_DATES:
        sets = temporal_split(df, start_date, clean, logger)
        pkg = (start_date, sets)
        temporal_sets.append(pkg)
    
    return temporal_sets


def standard_scale_data(X_train, X_test):
    '''
    Given X_train, X_test datasets, scales the features using MinMaxScaler().
    '''
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def train_model(X_train, y_train, clf, tuning, grid=None):
    '''
    Given X_train, X_test datasets, scales the features using MinMaxScaler().
    '''

    clfs = {
        'Baseline': DummyClassifier,
        'LR': LogisticRegression,
        'KNN': KNeighborsClassifier,
        'DT': DecisionTreeClassifier,
        'SVM': svm.LinearSVC,
        'RF': RandomForestClassifier,
        'AB': AdaBoostClassifier,
        'BG': BaggingClassifier}
    
    model = clfs[clf]

    if grid:
        param_grid = grid[clf]
        classifier = GridSearchCV(model(), param_grid, cv=5, scoring=tuning)
        classifier.fit(X_train, y_train)
        best_params = classifier.best_params_
        package = (best_params, model(**best_params).fit(X_train, y_train))
            
    else:
        package = ("Default", model().fit(X_train, y_train))

    return package


def evaluate_model(X_test, y_test, model, metrics):
    '''
    Evaluate a model using a given metrics.
    '''
    calc_thold = lambda x, y: 0 if x < y else 1

    if isinstance(model, svm.LinearSVC):
        pred_scores_test = model.decision_function(X_test)
        pred_test = model.predict(X_test)
    
    else:
        pred_scores_test = model.predict_proba(X_test)[:, 1]
        pred_test = np.array([calc_thold(sc, THOLD) for sc in pred_scores_test])

    if metrics == 'Accuracy':
        score = accuracy_score(y_pred=pred_test, y_true=y_test)

    elif metrics == 'F1':
        score = f1_score(y_pred=pred_test, y_true=y_test)

    elif metrics == 'ROC_AUC':
        score = roc_auc_score(y_score=pred_scores_test, y_true=y_test)

    elif 'Recall' in metrics:
        k = int(re.findall(r'\d+', metrics)[0])
        score = recall_at_k(y_test, pred_scores_test, k)

    elif 'Precision' in metrics:
        k = int(re.findall(r'\d+', metrics)[0])
        score = precision_at_k(y_test, pred_scores_test, k)

    else:
        score = None

    return score


def train_configured_models(X_train, y_train, grid=False):
    '''
    Train the model using preconfigured settings in config.yaml.
    If grid is False, it will use the sklearn default parameters
    for the predetermined models.
    '''

    trained_models = list()

    for m in MODELS:
        if grid is True:
            params, model = train_model(X_train, y_train, m, TUNING, GRID)
        else:
            params, model = train_model(X_train, y_train, m, TUNING)
        trained_models.append((m, params, model))
    
    return trained_models


def find_best_model(split_set, grid=False, scale=False, save=None):
    '''
    Find the best model given the split datasets. If grid is False, default
    parameters will be used. If True, it will use the "best" parameters for each
    classifier model. If scale is False, it will not scale features. If save is
    None, the resulting evaluation table will not be saved.
    '''

    X_train, X_test, y_train, y_test = split_set

    columns = ['model', 'parameters'] + METRICS

    results = pd.DataFrame(columns=columns)

    i = 0
    best_model = list()
    best_score = 0

    if scale is True:
        X_train, X_test = standard_scale_data(X_train, X_test)

    models = train_configured_models(X_train, y_train, grid=grid)

    for name, params, model in models:
        results.loc[i,'model'] = name
        results.loc[i,'parameters'] = str(params)
        for metric in METRICS:
            score = evaluate_model(X_test, y_test, model, metric)
            results.loc[i, metric] = score
            if metric == FINAL:
                if score > best_score:
                    best_score = score
                    best_model = [model]
                elif score == best_score:
                    best_model.append(model)
        i += 1

    results = results.sort_values([FINAL], ascending=False)
    results = results.reset_index(drop=True)
        
    if save:
        results.to_csv(path_or_buf=save, index=False)

    return best_model
