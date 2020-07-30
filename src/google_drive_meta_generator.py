import configparser
import sys
import os
from os import path
import google_drive_client as gcl
import argparse
from os import path
import logging
from datetime import datetime
import traceback
import file_service as fs
import google_drive_client as gcl

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

def main(content_dir, parent_id):
    logger = create_logger()
    logger.info('Starting Content Uploader for content folder: %s ; Parent Id: %s', str(content_dir), str(parent_id))
    metadata_folder = init_metadata(content_dir, logger)
    client = gcl.GoogleClient(logger, metadata_folder)

    try:
        content = client.generate_folder_content(parent_id)
        generate_metadata(content, client, parent_id, logger)
    except Exception as e:
        errorMsg = traceback.format_exc()
        print errorMsg
        logger.error(errorMsg) 

def generate_metadata(data, client, parent_id, logger):
    
    if not client.check_folder_exists(parent_id, "google_drive"):
        folder_id = client.create_folder(parent_id, "google_drive")
    else:
        folder_id = client.get_folder_with_parent_id(parent_id, "google_drive")
        
    logger.info("goodle_drive folder id: " + str(folder_id))
    client.generate_metadata_file(data,  file_path = 'random')
    client.upload_file_to_parent_folder(client.google_meta_path, folder_id)
    
def init_metadata(content_dir, logger):
    metadata_folder = os.path.join(content_dir, "google_drive")
    logger.info("Medata Folder: " + str(metadata_folder))
    fs.ensure_dir(metadata_folder)
    return metadata_folder
    
def create_logger():
    
    
    ##########################################
    # logging setup
    ##########################################
    
    start_date = str(datetime.now())
    r_date = "{}-{}-{}".format(start_date[0:4], start_date[5:7], start_date[8:10])
    log_dir = config.get("content_uploader", "content.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    current_date = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    # create a file handler
    app_log_name = 'content_uploader_' +  str(current_date) + '.log'
    app_log = path.join(log_dir, app_log_name)
    handler = logging.FileHandler(app_log)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    ##########################################
    # end logging stuff:
    ##########################################
    logger.info('Starting @ {}'.format(str(r_date)))
    
    return logger
        

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("\n\n\tUSAGE: [--content_dir --parent_id)\n\n")
        sys.exit(0)
        
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--content_dir')
    arg_parser.add_argument('--parent_id')
    
    args = arg_parser.parse_args()
    content_dir = args.content_dir
    parent_id = args.parent_id
    
    main(content_dir, parent_id) 
    