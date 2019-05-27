'''
Python function that allows the user to log and read the config file.

Si Young Byun
'''

import yaml
import logging
import os
import logging.config


def load_log_config(logger_name, config_filename):
    '''
    A function that loads a logger with the given name, using the given config.
    Input:
    - logger_name: a string that contains the name of the logger
    - config_filename: a string that contains the directory to the config file
    Output:
    - logger: the loaded logger
    '''
    
    logging.config.fileConfig(config_filename)
    logger = logging.getLogger(logger_name)

    return logger


def log_msg(logger, message):
    '''
    A function that logs an info-level message to the logger.
    Input:
    - logger: a logger
    - message: a string containing the message
    Output:
    - None
    '''

    if logger:
        logger.info(message)


def read_config(filename, logger=None):
    '''
    A function that reads the config for this machine learning model.
    If a logger is given, it will log the process.
    Input:
    - filename: a string that contains the directory to the config file
    - logger: the logger
    Output:
    - cfg: a dictionary that contains the config for the machine learning model
    '''

    msg1 = "\n# Loading configurations..."
    log_msg(logger, msg1)

    try:
        with open(filename, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        msg2 = "    >>> SUCCESS!"
        log_msg(logger, msg2)

        return cfg
    
    except:
        msg3 = "    >>> FAILED!"
        log_msg(logger, msg3)

        return None


#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    # For debugging purpose only
    if os.path.isfile("./results.log"):
        os.remove("./results.log")
    
    LOGGER = load_log_config('DonorsML', 'log_config.conf')
    log_msg(LOGGER, "# Deleted the existing log file.")

    cfg = read_config("./config.yaml", logger=LOGGER)
