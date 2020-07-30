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
from nilearn.image import resample_img
from nilearn.masking import compute_background_mask
import pyten
from pyten.tenclass import Tensor  # Use it to construct Tensor object
from pyten.tools import tenerror
from pyten.method import *
from datetime import datetime
import file_service as fs
import csv
from collections import OrderedDict

import mri_draw_utils as mrd

import configparser
import tensor_util as tu
import pandas as pd
import mri_draw_utils as mrd
import metric_util as mt

from pyten.method import rimrltc, rimrltc2, halrtc

class NonconvexTC(object):
    
    def __init__(self, path, observed_ratio, logger,meta, d):
        self.x_hat = path
        self.logger = logger
        self.meta = meta
        self.alpha = None
        self.epsilon = 1e-5
        self.d = d
        self.observed_ratio = observed_ratio
        self.missing_ratio = 1.0 - observed_ratio
        self.init()
        self.init_dataset(path)
    
    def complete(self): 
        
        # Construct Image Tensor to be Completed
        x_true = Tensor(self.ground_truth)
        x_train = Tensor(self.sparse_observation)
        mask_indices = self.mask_indices
        print ("mask_indices=" + str(mask_indices))
    
        self.x_hat1, self.rse_cost_history, self.cost_history, self.tcs_cost_history = rimrltc(x_train, x_true,mask_indices, max_iter=100, epsilon=1e-16, alpha=None)
        self.x_hat = self.x_hat1.data
        self.x_hat = mt.reconstruct2(self.x_hat, self.ground_truth, self.mask_indices)
        self.x_hat_img = mt.reconstruct_image_affine(self.ground_truth_img, self.x_hat)
        self.x_miss_img = mt.reconstruct_image_affine(self.ground_truth_img, self.sparse_observation)
        self.save_solution_scans_and_history(0, self.suffix, self.scan_mr_folder)
    
    def init(self):
        
        self.rse_cost_history = []
        self.train_cost_history = []
        self.tcs_cost_history = []
        self.tcs_z_scored_history = []
        self.cost_history = []
        self.scan_mr_folder = self.meta.create_scan_mr_folder(self.missing_ratio)
        self.scan_mr_iteration_folder = self.meta.create_scan_mr_folder_iteration(self.missing_ratio)
        self.images_mr_folder_iteration = self.meta.create_images_mr_folder_iteration(self.missing_ratio)
        self.suffix = self.meta.get_suffix(self.missing_ratio)
        
        self.logger.info(self.scan_mr_iteration_folder)
        self.logger.info(self.suffix)
    
    def init_dataset(self, path):
        self.x_true_img = mt.read_image_abs_path(path)
        
        self.x_true_data = np.array(self.x_true_img.get_data())
        self.x_true_reshaped = copy.deepcopy(self.x_true_data)
        self.x_true_copy = copy.deepcopy(self.x_true_data)
              
        self.target_shape = mt.get_target_shape(self.x_true_data, self.d)
        self.logger.info("Target Shape: " + str(self.target_shape))
        self.x_true_reshaped_rank = mt.reshape_as_nD(self.x_true_copy, self.d,self.target_shape)
        self.logger.info("D = " + str(self.d) + "; Original Shape: " + str(self.x_true_data.shape) + "; Target Shape: " + str(self.target_shape))
        
        self.tensor_shape = tu.get_tensor_shape(self.x_true_data)
        self.max_tt_rank = tu.get_max_rank(self.x_true_reshaped_rank)
        
        self.logger.info("Tensor Shape: " + str(self.tensor_shape) + "; Max Rank: " + str(self.max_tt_rank))
        
        # mask_indices after reshape
        self.mask_indices = self.init_mask()
        
        #init after reshape
        self.ten_ones = tu.get_ten_ones(self.x_true_reshaped)
       
        # ground truth to be initialized later after reshape
        self.ground_truth, self.norm_ground_truth = tu.normalize_data(self.x_true_reshaped)
        self.logger.info("Norm Ground Truth: " + str(self.norm_ground_truth))
        
        self.ground_truth_img = mt.reconstruct_image_affine(self.x_true_img, self.ground_truth)
        
        if len(self.x_true_reshaped.shape) > 2:
            self.ground_truth_img = mt.reconstruct_image_affine(self.x_true_img, self.ground_truth)
         
        #initial approximation to be initialized later after reshape
        self.x_init = tu.init_random(self.x_true_reshaped) 
        self.x_init_tcs = self.ground_truth * (1./np.linalg.norm(self.x_init))
        
        self.x_init, self.norm_x_init = tu.normalize_data(self.x_init)
        self.logger.info("Norm X Init: " + str(self.norm_x_init))
                     
        # sparse_observation to be initialized later after reshape
        self.sparse_observation = tu.create_sparse_observation(self.ground_truth, self.mask_indices)
        self.norm_sparse_observation = np.linalg.norm(self.sparse_observation)
       
        
        # create x_miss
        self.x_miss = np.array(self.sparse_observation)
        
    def init_mask(self):
        self.mask_indices = tu.get_mask_with_epi(self.x_true_data, self.x_true_img, self.observed_ratio, self.d)
        return self.mask_indices
           
    def save_solution_scans_and_history(self, iteration, suffix, folder): 
        
        self.logger.info("Missing Ratio: " + str(self.missing_ratio))
        x_true_path = os.path.join(folder, "x_true_img_" + str(suffix))
        x_hat_path = os.path.join(folder, "x_hat_img_" + str(suffix))
        x_miss_path = os.path.join(folder, "x_miss_img_" + str(suffix))
        
        self.logger.info("x_hat_path: " + str(x_hat_path))
        nib.save(self.x_hat_img, x_hat_path)
            
        self.logger.info("x_miss_path: " + str(x_miss_path))
        nib.save(self.x_miss_img, x_miss_path)
            
        self.logger.info("x_true_path: " + str(x_true_path))
        nib.save(self.ground_truth_img, x_true_path)
        
        self.logger.info("Missing Ratio: " + str(self.missing_ratio))
        
        self.logger.info("Missing Ratio: " + str(self.missing_ratio))
        x_true_path = os.path.join(folder, "x_true_img_" + str(suffix))
        x_hat_path = os.path.join(folder, "x_hat_img_" + str(suffix))
        x_miss_path = os.path.join(folder, "x_miss_img_" + str(suffix))
        
        self.logger.info("x_hat_path: " + str(x_hat_path))
        nib.save(self.x_hat_img, x_hat_path)
            
        self.logger.info("x_miss_path: " + str(x_miss_path))
        nib.save(self.x_miss_img, x_miss_path)
            
        self.logger.info("x_true_path: " + str(x_true_path))
        nib.save(self.ground_truth_img, x_true_path)
        
        output_cost = OrderedDict()
        indices = []

        cost_arr = []
        tsc_arr = []
        
        rse_arr = []

        counter = 0
        for item in  self.cost_history:
            self.logger.info(item)
            cost_arr.append(item)
            indices.append(counter)
            counter = counter + 1
    
        output_cost['k'] = indices
        output_cost['cost'] = cost_arr
    
        output_df = pd.DataFrame(output_cost, index=indices)

        results_folder = self.meta.results_folder
        
        fig_id = 'solution_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_df, results_folder, fig_id)  

        tsc_score_output = OrderedDict()
        tsc_score_indices = []

        counter = 0
        for item in self.tcs_cost_history:
            tsc_arr.append(item)
            tsc_score_indices.append(counter)
            counter = counter + 1

        tsc_score_output['k'] = tsc_score_indices
        tsc_score_output['tsc_cost'] = tsc_arr
    
        output_tsc_df = pd.DataFrame(tsc_score_output, index=tsc_score_indices)
        fig_id = 'tsc_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_tsc_df, results_folder, fig_id) 
        
        # output rse history
        
        rse_output = OrderedDict()
        rse_indices = []
        counter = 0
        
        for item in self.rse_cost_history:
            rse_arr.append(item)
            rse_indices.append(counter)
            counter = counter + 1

        rse_output['k'] = rse_indices
        rse_output['rse_cost'] = rse_arr
        
        output_rse_df = pd.DataFrame(rse_output, index=rse_indices)
        fig_id = 'rse_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_rse_df, results_folder, fig_id)    
        
        self.title = "Test"
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.x_hat_img,0), image.index_img(self.x_miss_img, 0), self.title + " Iteration: " + str(iteration),
                        self.tcs_cost_history[iteration], self.observed_ratio, self.tcs_cost_history[iteration], self.tcs_cost_history[iteration], 2, coord=None, folder=self.scan_mr_folder, iteration=iteration)
              
        
        
