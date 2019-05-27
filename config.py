'''
Python function that allows the user to read the config file.

Si Young Byun
'''

import yaml
import logging
import os
import logging.config
#logging.config.fileConfig('log_config.conf')
#LOGGER = logging.getLogger("DonorsML")

def load_log_config(logger_name, config_filename):
    
    logging.config.fileConfig(config_filename)
    logger = logging.getLogger(logger_name)

    return logger


def log_msg(logger, message):

    if logger:
        logger.info(message)


def read_config(filename, logger=None):

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
