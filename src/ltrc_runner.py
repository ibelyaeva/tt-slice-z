import numpy as np
import nilearn
import configparser
import os
from os import path
import logging
import metadata as mdt
import data_util as du
import low_rank as lr
import time

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

def complete_random_4D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 3)
    root_dir = config.get('log', 'scratch.dir4Dnonconvex')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    #observed_ratio_list = [0.9]
    
    x0, y0, z0 = (2,32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    max_iter = 50
    for item in observed_ratio_list:
        current_runner = lr.LowRank(subject_scan_path, item, 3, 1, meta.logger, meta, max_iter)
        current_runner.complete()

if __name__ == "__main__":
    pass
    #complete_random_2D()
    complete_random_4D()
    #complete_random_3D()