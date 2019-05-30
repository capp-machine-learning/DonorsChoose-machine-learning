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


def plot_feature_importances(importances, col_names, save='./images/fi.png'):
    """
    Plot the feature importances of the model. The code adapted from:
    The University of Michigan
    """
    indices = np.argsort(importances)[::-1][:5]
    labels = col_names[indices][::-1]

    fig, _ = plt.subplots(figsize=[12, 8])
    plt.barh(range(5), sorted(importances, reverse=True)[:5][::-1], color='g',
             alpha=0.4, edgecolor=['black']*5)

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.yticks(np.arange(5), labels)
    
    plt.savefig(save)


def plot_precision_recall_curve(X_test, y_test, model, save='./images/prc.png'):

    if isinstance(model, svm.LinearSVC):
        pred_scores_test = model.decision_function(X_test)
    
    else:
        pred_scores_test = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test=y_test, y_score=pred_scores_test)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(model)

    plt.savefig(save)