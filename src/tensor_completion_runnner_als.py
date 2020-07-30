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
import completion_als as als
import data_util as du
import mri_draw_utils as mrd
import solution as sl
import solution_writer as sw

import configparser
from os import path
import logging


def complete_random4d():
    subject_scan_path = du.get_full_path_subject1()
    print "Subject Path: " + str(subject_scan_path)
    
    title = 'fMRI 4D Scan Completion. Pattern - Randomly Missing Values.'
    pattern = 'random'
    solution_path = "/work/scratch/tensor_completion/als/4D/random/"
    
    folder = "results"
    img_folder = "d4/random"
    all_solutions = sw.SolutionWriter(folder)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.9]
       
    for item in observed_ratio_list:
        
        print("Observation Ratio: " + str(item))
        #observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors = rm.complete_tensor_random_pattern(subject_scan_path, item, n=-1)
        observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img = als.complete_tensor_random_pattern(subject_scan_path, item, n=-1)
        
        x_true = image.index_img(x_true_img,1)
        x_hat = image.index_img(x_hat_img,1)
        x_miss = image.index_img(x_miss_img,1)
        
        mrd.draw_original_vs_reconstructed_rim(x_true, x_hat, x_miss, title, rel_err, observed_ratio, coord=None, folder=img_folder)
        #compl_sol = save_solution(observed_ratio, 4, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors,  pattern, solution_path)
    
        #all_solutions.solutions.append(compl_sol)
    
    print ("Saving All Solutions ...")
    #all_solutions.write_all()
    print ("Saving All Solutions ... Done.")
    
    print ("4D Random Missing Value Pattern Simulations Completed... Done.")

def save_solution(observed_ratio, d, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, test_errors, pattern, solution_path, tsc, nrmse,scan_folder=None):
    
    print ("Saving Solution ...")
    print("tsc = " + str(tsc))
    print("nrmse = " + str(nrmse))
    
    solution = sl.Solution(pattern, d, observed_ratio, x_true_img, x_hat_img, x_miss_img, rel_err, solution_errors, test_errors, solution_path, tsc=tsc, nrmse=nrmse, scan_folder=scan_folder)
    solution.save_solution_scans()
    solution.write_summary()
    solution.write_cost()
    
    print ("Solution scans saved @ " + str(scan_folder))
    print ("Solution saved @ " + str(solution_path))
    
    
    return solution

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
    app_log_name = 'tensor_completion' +  str(current_date) + '.log'
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

def init_meta_data(n, pattern, config, logger):
    
        root_dir = config.get('log', 'scratch.dir')
        dim_dir = "d" + str(n)
        solution_dir = path.join(dim_dir, pattern)
        
        solution_folder = fs.create_batch_directory(root_dir, solution_dir)
       
        images_folder = fs.create_batch_directory(solution_folder, "images", False)
        results_folder = fs.create_batch_directory(solution_folder, "results", False)
        reports_folder = fs.create_batch_directory(solution_folder, "reports", False)
        scans_folder = fs.create_batch_directory(solution_folder, "scans", False)
            
        logger.info("Created solution dir at: [%s]" ,solution_folder)
        logger.info("Created images dir at: [%s]" ,images_folder)
        logger.info("Created results dir at: [%s]" ,results_folder)
        logger.info("Created reports dir at: [%s]" ,reports_folder)
        logger.info("Created scans dir at: [%s]" ,scans_folder)
        
        return solution_dir, images_folder, results_folder, reports_folder, scans_folder
    
def init_meta_data2D(n, pattern, config, logger):
    
        root_dir = config.get('log', 'scratch.dir2D')
        dim_dir = "d" + str(n)
        solution_dir = path.join(dim_dir, pattern)
        
        solution_folder = fs.create_batch_directory(root_dir, solution_dir)
       
        images_folder = fs.create_batch_directory(solution_folder, "images", False)
        results_folder = fs.create_batch_directory(solution_folder, "results", False)
        reports_folder = fs.create_batch_directory(solution_folder, "reports", False)
        scans_folder = fs.create_batch_directory(solution_folder, "scans", False)
            
        logger.info("Created solution dir at: [%s]" ,solution_folder)
        logger.info("Created images dir at: [%s]" ,images_folder)
        logger.info("Created results dir at: [%s]" ,results_folder)
        logger.info("Created reports dir at: [%s]" ,reports_folder)
        logger.info("Created scans dir at: [%s]" ,scans_folder)
        
        return solution_dir, images_folder, results_folder, reports_folder, scans_folder
    
if __name__ == "__main__":
    pass
    #complete_random3d()
    complete_random4d();
    #complete_random2d()
    config_loc = path.join('config')
    config_filename = 'solution.config'
    config_file = os.path.join(config_loc, config_filename)
    config = configparser.ConfigParser()
    config.read(config_file)
    
    logger = create_logger(config)
    #complete_random3d1(config, logger)
    #complete_random2d()
    
    