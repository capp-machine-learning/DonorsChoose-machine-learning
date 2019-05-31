'''
Functions for visualizing the dataset.

Si Young Byun
'''
from sklearn import svm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from inspect import signature
import missingno as missingno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_precision_recall_curve(X_test, y_test, model, save='./images/prc.png'):

    if isinstance(model, svm.LinearSVC):
        pred_scores_test = model.decision_function(X_test)
    
    else:
        pred_scores_test = model.predict_proba(X_test)[:, 1]
    
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, pred_scores_test)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(pred_scores_test)
    for value in pr_thresholds:
        num_above_thresh = len(pred_scores_test[pred_scores_test>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model
    plt.title(name)
    plt.savefig(save, bbox_inches="tight")