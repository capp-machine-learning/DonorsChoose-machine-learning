'''
Functions for building and evaluating the dataset.

Si Young Byun
'''

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

from config import *

CONFIG = read_config('config.yaml')
OUTCOME = CONFIG['outcome_var']
TEST_SIZE = CONFIG['test_size']
RANDOM_STATE = CONFIG['random_state']
THOLD = CONFIG['threshold']
PR_THOLD = CONFIG['PR_thold']
GRID = CONFIG['parameters']
MODELS = CONFIG['models']


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


def temporal_train_test_split(df, time_col, start_date, period):
    '''
    
    '''
    start_date = pd.to_datetime(start_date)
    end_date = start_date + pd.DateOffset(months=period)

    train = df[df[time_col] < start_date].drop(time_col, axis=1)
    test = df[(df[time_col] >= start_date) & (df[time_col] < end_date)].drop(time_col, axis=1)

    y_train = train[OUTCOME]
    y_test = test[OUTCOME]
    X_train = train.drop(OUTCOME, axis=1)
    X_test = test.drop(OUTCOME, axis=1)

    return (X_train, X_test, y_train, y_test)


def temporal_loop(df, time_col, start_dates, period):

    temporal_sets = []
    for i in start_dates:
        sets = temporal_train_test_split(df, time_col, i, period)
        pkg = (i, sets)
        temporal_sets.append(pkg)
    
    return temporal_sets


def train_model(X_train, y_train, clf, grid=None):

    clfs = {
        'LR': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'DT': DecisionTreeClassifier(),
        'SVM': svm.LinearSVC(),
        'RF': RandomForestClassifier(),
        'GB': GradientBoostingClassifier(),
        'BG': BaggingClassifier()}
    
    model = clfs[clf]
    models = []

    if grid:
        param_grid = grid[clf]
        for params in ParameterGrid(param_grid):
            model.set_params(**params)
            package = (params, model.fit(X_train, y_train))
            models.append(package)
            
    else:
        package = ("Default", model.fit(X_train, y_train))
        models.append(package)

    return models


def evaluate_model(X_test, y_test, model):

    calc_thold = lambda x, y: 0 if x < y else 1

    if isinstance(model, svm.LinearSVC):
        pred_scores_test = model.decision_function(X_test)
        pred_test = model.predict(X_test)
    
    else:
        pred_scores_test = model.predict_proba(X_test)[:, 1]
        pred_test = np.array([calc_thold(sc, THOLD) for sc in pred_scores_test])

    acc = accuracy_score(y_pred=pred_test, y_true=y_test)
    f1 = f1_score(y_pred=pred_test, y_true=y_test)
    auc = roc_auc_score(y_score=pred_scores_test, y_true=y_test)
    eval_metrics = [acc, f1, auc]

    for t in PR_THOLD:
        eval_metrics.append(
            precision_at_k(y_test, pred_scores_test, t))
        eval_metrics.append(
            recall_at_k(y_test, pred_scores_test, t))
    
    return eval_metrics


def evaluation_table(temporal_sets, grid=False):

    full_results = pd.DataFrame(columns=['Date', 'Model','Parameters','Accuracy','F1','AUC_ROC','Precision_at_1%', 'Recall_at_1%','Precision_at_2%', 'Recall_at_2%','Precision_at_5%', 'Recall_at_5%','Precision_at_10%', 'Recall_at_10%','Precision_at_20%', 'Recall_at_20%','Precision_at_30%', 'Recall_at_30%','Precision_at_50%', 'Recall_at_50%'])

    i = 0

    for (date, (X_train, X_test, y_train, y_test)) in temporal_sets:

        for m in MODELS:
            if grid is True:
                mdls = train_model(X_train, y_train, m, GRID)
            else:
                mdls = train_model(X_train, y_train, m)
            for params, ms in mdls:
                (acc, f1, auc, p1, r1, p2, r2, p5, r5, p10, r10, p20, r20, p30, r30, p50, r50) = evaluate_model(X_test, y_test, ms)
                full_results.loc[i,'Date'] = date
                full_results.loc[i,'Model'] = m
                full_results.loc[i,'Parameters'] = str(params)
                full_results.loc[i,'Accuracy'] = acc
                full_results.loc[i,'F1'] = f1
                full_results.loc[i,'AUC_ROC'] = auc
                full_results.loc[i,'Precision_at_1%'] = p1
                full_results.loc[i,'Recall_at_1%'] = r1
                full_results.loc[i,'Precision_at_2%'] = p2
                full_results.loc[i,'Recall_at_2%'] = r2
                full_results.loc[i,'Precision_at_5%'] = p5
                full_results.loc[i,'Recall_at_5%'] = r5
                full_results.loc[i,'Precision_at_10%'] = p10
                full_results.loc[i,'Recall_at_10%'] = r10
                full_results.loc[i,'Precision_at_20%'] = p20
                full_results.loc[i,'Recall_at_20%'] = r20
                full_results.loc[i,'Precision_at_30%'] = p30
                full_results.loc[i,'Recall_at_30%'] = r30
                full_results.loc[i,'Precision_at_50%'] = p50
                full_results.loc[i,'Recall_at_50%'] = r50
                i += 1


    return full_results

