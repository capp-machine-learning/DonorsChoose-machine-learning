'''
Python function that allows the user to read the config file.

Si Young Byun
'''

import yaml
import logging

logging.basicConfig(filename='./results.log', filemode='w', level=logging.INFO)


def read_config(filename, verbose=False):

    try:
        with open(filename, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
        return cfg
    
    except:
        