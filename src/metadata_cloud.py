import configparser
from os import path
import logging
from datetime import datetime
import file_service as fs
import os

class MetadataCloud(object, ):
    
    
    def __init__(self, root_dir):
        config_loc = path.join('config')
        config_filename = 'solution.config'
        config_file = os.path.join(config_loc, config_filename)
        config = configparser.ConfigParser()
        config.read(config_file)
        self.config = config
        self.logger = create_logger(self.config)
        self.root_dir = root_dir
        self.init_meta_data()

    def init_meta_data(self):  
        self.solution_dir = 'google_drive'
        self.solution_folder = fs.create_batch_directory(self.root_dir, self.solution_dir)
    
        self.logger.info("Created solution dir at: [%s]" ,self.solution_folder)
        return self.solution_folder
    
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
    app_log_name = 'google_drive_' +  str(current_date) + '.log'
    app_log = path.join(log_dir, app_log_name)
    handler = logging.FileHandler(app_log)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.info("Google Drive...")

    ##########################################
    # end logging stuff:
    ##########################################
    logger.info('Starting @ {}'.format(str(current_date)))
    
    return logger
