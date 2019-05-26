'''
Python function that allows the user to read the config file.

Si Young Byun
'''

import yaml
import logging
import os
import logging.config
logging.config.fileConfig('log_config.conf')
LOGGER = logging.getLogger("DonorsML")


def read_config(filename, log=False):

    if log is True:
        LOGGER.info("\n# Loading configurations...")

    try:
        with open(filename, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        if log is True:
            LOGGER.info("    >>> SUCCESS!")

        return cfg
    
    except:
        if log is True:
            LOGGER.info("    >>> FAILED!")
        
        return None


#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    # For debugging purpose only
    if os.path.isfile("./results.log"):
        os.remove("./results.log")
        logging.config.fileConfig('log_config.conf')
        LOGGER = logging.getLogger("DonorsML")
        LOGGER.info("# Deleted the existing log file.")

    cfg = read_config("./config.yaml", log=True)