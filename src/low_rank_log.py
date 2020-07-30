import numpy as np
import nilearn
import configparser
import os
from os import path
import logging
import metadata as mdt
import data_util as du

import nilearn
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
import metric_util as mt
import tensor_util as tu
import time
import lrtc_log as lr

import mri_draw_utils as mrd

class LowRankLog(object):
    
    def __init__(self, data_path, observed_ratio, d, n, logger, meta, max_iter, z_score = 2):
        self.observed_ratio = observed_ratio
        self.missing_ratio = 1.0 - self.observed_ratio
        self.d = d
        self.n = n
        self.logger = logger
        self.meta = meta
        self.z_scored_mask = None
        self.z_score = z_score
        self.init_cost_history()
        self.init_dataset(data_path)
        self.max_iter = max_iter
        
    def init_dataset(self, path):
        self.x_true_img = mt.read_image_abs_path(path)
        
        self.x_true_data = np.array(self.x_true_img.get_data())
        self.x_true_reshaped = copy.deepcopy(self.x_true_data)
        self.x_true_copy = copy.deepcopy(self.x_true_data)
        
        self.target_shape = mt.get_target_shape(self.x_true_data, self.d)
        self.logger.info("Target Shape: " + str(self.target_shape))
        
        self.target_shape = mt.get_target_shape(self.x_true_data, self.d)
        self.logger.info("Target Shape: " + str(self.target_shape))
        self.x_true_reshaped_rank = mt.reshape_as_nD(self.x_true_copy, self.d,self.target_shape)
        self.logger.info("D = " + str(self.d) + "; Original Shape: " + str(self.x_true_data.shape) + "; Target Shape: " + str(self.target_shape))
        
        self.tensor_shape = tu.get_tensor_shape(self.x_true_data)
        self.max_tt_rank = tu.get_max_rank(self.x_true_reshaped_rank)
        self.max_tt_rank = 120
        
        self.logger.info("Tensor Shape: " + str(self.tensor_shape) + "; Max Rank: " + str(self.max_tt_rank))
             
        # mask_indices after reshape
        self.mask_indices = self.init_mask()
        
        #init after reshape
        self.ten_ones = tu.get_ten_ones(self.x_true_reshaped)
       
        # ground truth to be initialized later after reshape
        self.ground_truth, self.norm_ground_truth = tu.normalize_data(self.x_true_reshaped)
        self.logger.info("Norm Ground Truth: " + str(self.norm_ground_truth))
        
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
        
        self.epsilon = 1e-3
        self.train_epsilon = 1e-5
        self.backtrack_const = 1e-4
        
        # related z_score structures
        self.std_img = None
        self.mean_img = None
        self.z_scored_image = None
        self.ground_truth_z_score = None
        self.mask_z_score_indices = None
        self.mask_z_indices_count = None
        
        
    def init_mask(self):
        self.mask_indices = tu.get_mask_with_epi(self.x_true_data, self.x_true_img, self.observed_ratio, self.d)
        return self.mask_indices
    
        
    def init_cost_history(self):
        self.rse_cost_history = []
        self.train_cost_history = []
        self.tcs_cost_history = []
        self.tcs_z_scored_history = []
        self.summary_history = []
        
    def complete(self):
        
        self.logger.info("Starting Tensor Completion. Tensor Dimension:" + str(self.d))
        t1 = time.time()
        
        self.original_tensor_shape = tu.get_tensor_shape(self.x_true_data)
        self.original_max_rank = tu.get_max_rank(self.x_true_data)
        self.logger.info("Original Tensor Shape: " + str(self.original_tensor_shape) + "; Original Tensor Max Rank: " + str(self.original_max_rank)) 
        
        t2 = time.time()
        total_time = str(t2 - t1)
        self.logger.info("Finished Tensor Completion. Tensor Dimension:" + str(self.d)  + "... Done.")  
        self.logger.info("Execution time, seconds: " + str(total_time))
        
        self.rtc_runner = lr.LowRankTensorCompletionLog(self.ground_truth_img,
                                                    self.ground_truth, self.tensor_shape, 
                                                    self.x_init,
                                                    self.mask_indices, 
                                                    self.z_scored_mask,
                                                    self.sparse_observation,
                                                    self.norm_sparse_observation, 
                                                    self.x_init_tcs,
                                                    self.ten_ones, 
                                                    self.max_tt_rank, 
                                                    self.observed_ratio,
                                                    self.epsilon, self.train_epsilon,
                                                    self.backtrack_const, self.logger, self.meta, self.d, self.max_iter, self.z_score,
                                                    )
        
        self.rtc_runner.complete()
        
        #mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.rtc_runner.x_hat_img,0), image.index_img(self.rtc_runner.x_miss_img, 0), "4D fMRI Tensor Completion",
        #                                    self.rtc_runner.tsc_score, self.observed_ratio, self.rtc_runner.tsc_score, self.rtc_runner.tcs_z_score, 2, coord=None, folder=self.rtc_runner.meta.images_folder,  iteration = -1)
    
        
        