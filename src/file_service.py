import logging
import sys
import collections
from datetime import datetime
import time
import traceback
import os
from os import path
import csv
from datetime import datetime

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
def create_batch_directory(root_dir, target_dir, create_run=True):
    
    current_date = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    # create a file handler
    dir_name = 'run_' + str(current_date)
    
    if create_run:
        batch_dir_path = path.join(root_dir, dir_name,target_dir)
    else:
        batch_dir_path = path.join(root_dir,target_dir)
        
    ensure_dir(batch_dir_path)
    
    return batch_dir_path