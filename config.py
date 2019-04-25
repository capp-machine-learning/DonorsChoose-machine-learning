# config.py
'''
Config for Machine Learning Analysis.

Si Young Byun (syb234)
'''
#from pathlib import Path

#data_dir = Path()

PIPELINE_CONFIG = {
    'dataset': 'data/credit-data.csv',
    'data_dict': 'data/data-dictionary.xls',
    'outcome_var': 'SeriousDlqin2yrs',
    'test_size': 0.3,
    'threshold': 0.4,
    'random_state': 10
}