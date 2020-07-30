import data_util as du
import metric_util as mt
import configparser
from os import path
import logging
from datetime import datetime
import file_service as fs
import os

class ResultsMetadata(object):
    
        def __init__(self, pattern, config_filename, n):
            config_loc = path.join('config')
            self.config_filename = config_filename
            config_file = os.path.join(config_loc, config_filename)
            config = configparser.ConfigParser()
            config.read(config_file)
            self.config = config
            self.logger = create_logger(self.config)
            self.pattern = pattern
            self.n = n
            
        def init_results_metadadata(self, root_dir):
            self.root_dir = root_dir
            dim_dir = "d" + str(self.n)
            self.solution_dir = path.join(dim_dir, self.pattern)
        
    
def create_logger(config):
    
    ##########################################
    # logging setup
    ##########################################
    
    start_date = str(datetime.now())
    r_date = "{}-{}-{}".format(start_date[0:4], start_date[5:7], start_date[8:10])
    log_dir = config.get("log", "log.dir")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    current_date = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    # create a file handler
    app_log_name = 'tensor_completion_' +  str(current_date) + '.log'
    app_log = path.join(log_dir, app_log_name)
    handler = logging.FileHandler(app_log)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.info("Tensor Completion...")

    ##########################################
    # end logging stuff:
    ##########################################
    logger.info('Starting @ {}'.format(str(current_date)))
    
    return logger