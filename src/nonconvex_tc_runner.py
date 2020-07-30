import os
import numpy as np
from math import ceil
import pyten
from pyten.tenclass import Tensor  # Use it to construct Tensor object
from pyten.tools import tenerror
from pyten.method import *
from datetime import datetime
import file_service as fs
import csv
import mri_draw_utils as mrd
import metric_util as mt

import data_util as du

import configparser
from os import path
import logging
import metadata as mdt


config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

from pyten.method import rimrltc2 as rim, rimrltc

import mri_draw_utils as mri_d
import data_util as dtu
import metric_util as mt
from scipy import ndimage
from nilearn.masking import compute_epi_mask
import nc_tc as nc
np.random.seed(0)

def complete_random_4D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 4)
    root_dir = config.get('log', 'scratch.dir4Dnonconvex')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    observed_ratio_list = [0.50]
    ranks = {}
    
    ranks[0.95] = 250
    ranks[0.95] = 120
    ranks[0.9] = 120
    ranks[0.85] = 120
    ranks[0.8] = 120
    ranks[0.75] = 120
    ranks[0.7] = 120
    ranks[0.6] = 120
    ranks[0.5] = 120
    ranks[0.4] = 120
    ranks[0.3] = 115
    ranks[0.25] = 110
    ranks[0.2] = 105
    ranks[0.15] = 90
    ranks[0.1] = 81
    
    for item in observed_ratio_list:
        current_runner = nc.NonconvexTC(subject_scan_path, item, meta.logger, meta, 4)
        current_runner.complete()
        
if __name__ == "__main__":
    pass
    complete_random_4D()