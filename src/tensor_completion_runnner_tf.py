import texfig

import nilearn

from medpy.io import load
from medpy.features.intensity import intensities
from nilearn import image
import nibabel as nib
from medpy.io import header
from medpy.io import load, save
import copy
from nilearn import plotting
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from math import ceil
from nilearn.datasets import MNI152_FILE_PATH
from sklearn.model_selection import train_test_split
from nibabel.affines import apply_affine
from nilearn.image.resampling import coord_transform, get_bounds, get_mask_bounds
from skimage.draw import ellipsoid
from nilearn.image import resample_img
from nilearn.masking import compute_background_mask
import pyten
from pyten.tenclass import Tensor  # Use it to construct Tensor object
from pyten.tools import tenerror
from pyten.method import *
from datetime import datetime
import file_service as fs
import csv

import metric_util as mt
import completion_riemannian as rm
import data_util as du
import mri_draw_utils as mrd
import solution as sl
import solution_writer as sw

import configparser
from os import path
import logging
import metadata as mdt
import complete_tensor as ct
import tensor_util as tu


config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

def complete_random_4D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 4)
    root_dir = config.get('log', 'scratch.dir4D')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.9]
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletion(subject_scan_path, item, 4, 1, meta.logger, meta)
        current_runner.complete()

def complete_random_3D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 3)
    root_dir = config.get('log', 'scratch.dir3D')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.9]
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletion(subject_scan_path, item, 3, 0, meta.logger, meta)
        current_runner.complete()

        
def complete_random_2D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 2)
    root_dir = config.get('log', 'scratch.dir2D')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.25]
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletion(subject_scan_path, item, 2, 1, meta.logger, meta)
        current_runner.complete()

if __name__ == "__main__":
    pass
    #complete_random_2D()
    #complete_random_4D()
    complete_random_3D()

    
    