'''
Python file that allows the user to read the config file.

Si Young Byun
'''

import yaml

def read_config(filename):

    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    return cfg