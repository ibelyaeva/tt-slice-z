import texfig

import tensorflow as tf
import numpy as np
import t3f
tf.set_random_seed(0)
np.random.seed(0)
import matplotlib.pyplot as plt
import metric_util as mt
import data_util as du
from t3f import shapes
from nilearn import image
import copy
from nilearn import plotting
from t3f import ops
import mri_draw_utils as mrd
from t3f import initializers
from t3f import approximate
from scipy import optimize 
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from collections import OrderedDict
import pandas as pd
from scipy import stats
from nilearn.image import math_img
import cost_computation as cst
import tensor_util as tu
import nibabel as nib
import os
import metadata as mdt
import metric_util as mt
import low_rank as lr
import time
from numpy.linalg import inv as inv


class LowRankTensorCompletion(object):
    
    def __init__(self, ground_truth_img, ground_truth, tensor_shape, x_init, mask_indices, z_scored_mask, 
                 sparse_observation_org,
                 norm_sparse_observation,
                 x_init_tcs,
                 ten_ones, max_tt_rank, observed_ratio, epsilon, train_epsilon, backtrack_const, logger, meta, d, max_iter, z_score = 2):
        
        self.ground_truth_img = ground_truth_img
        self.ground_truth = ground_truth
        self.tensor_shape = tensor_shape
        self.x_init = x_init
        self.mask_indices = mask_indices
        self.z_scored_mask = z_scored_mask
        self.sparse_observation_org = sparse_observation_org
        self.norm_sparse_observation = norm_sparse_observation
        self.x_init_tcs = x_init_tcs
        self.ten_ones = ten_ones
        self.max_tt_rank = max_tt_rank
        self.observed_ratio = observed_ratio
        self.missing_ratio = 1.0 - observed_ratio
        self.logger = logger
        self.meta = meta
        self.d = d
        self.z_score = z_score
                 
        self.epsilon = epsilon
        self.train_epsilon = train_epsilon
        
        self.theta = 2
        self.alpha = 1000
        self.rho = 0.01
        self.beta = 0.1 * self.rho
        self.title = ""
        self.max_iter = max_iter
        self.backtrack_const = backtrack_const
        self.init()
       
    def init(self):      
        self.rse_cost_history = []
        self.train_cost_history = []
        self.tcs_cost_history = []
        self.tcs_z_scored_history = []
        self.cost_history = []
        self.elapsed_time = []
        self.penalty = []
        self.scan_mr_folder = self.meta.create_scan_mr_folder(self.missing_ratio)
        self.scan_mr_iteration_folder = self.meta.create_scan_mr_folder_iteration(self.missing_ratio)
        self.images_mr_folder_iteration = self.meta.create_images_mr_folder_iteration(self.missing_ratio)
        self.suffix = self.meta.get_suffix(self.missing_ratio)
        
        self.original_shape = self.ground_truth.shape
        self.target_shape = mt.get_target_shape(self.ground_truth, self.d)
        self.logger.info("D = " + str(self.d) + "; Original Shape: " + str(self.original_shape) + "; Target Shape: " + str(self.target_shape))
        
        self.sparse_observation_org = copy.deepcopy(self.ground_truth)
        self.sparse_observation_org[self.mask_indices == 0] = 0.0
        
        self.x_miss_img = mt.reconstruct_image_affine(self.ground_truth_img, self.sparse_observation_org)
        
        if self.d == 3 or self.d == 2:
            self.mask_indices = mt.reshape_as_nD(self.mask_indices, self.d,self.target_shape)
            
        #create x_miss
        self.x_miss = np.array(self.sparse_observation_org)
        
        if self.d == 3 or self.d == 2:
            self.x_miss =  mt.reshape_as_nD(self.x_miss, self.d,self.target_shape)
        
        #update ground_truth if needed
        if self.d == 3 or self.d == 2:
            self.ground_truth = mt.reshape_as_nD(self.ground_truth, self.d,self.target_shape)
            
        self.logger.info("Ground Truth Shape: " + str(self.ground_truth.shape))
        
        #reshape ten_ones
        if self.d == 3 or self.d == 2:
            self.ten_ones = mt.reshape_as_nD(self.ten_ones, self.d,self.target_shape)
            
        self.logger.info("Ten Ones Truth Shape: " + str(self.ten_ones.shape))
        
        # reshape self.x_init_tcs
        if self.d == 3 or self.d == 2:
            self.x_init_tcs = mt.reshape_as_nD(self.x_init_tcs, self.d,self.target_shape)
            
        self.logger.info("Init TCS Shape: " + str(self.x_init_tcs.shape))
        
        # reshape self.x_init
        if self.d == 3 or self.d == 2:
            self.x_init = mt.reshape_as_nD(self.x_init, self.d,self.target_shape)
            
        self.logger.info("X Init Shape: " + str(self.x_init.shape))
        
        self.logger.info(self.scan_mr_iteration_folder)
        self.logger.info(self.suffix)
    
    def gemanp(self, img, alpha, theta):
        G = 0
        for k in range(img.shape[2]):
            u, s, v = np.linalg.svd(img[:,:,k], full_matrices = 0)
            for sigma in s:
                G = G + alpha * sigma / (sigma + theta)
        return G
    
    def gsvt_geman(self, X, alpha, theta, rho):
        u, s, v = np.linalg.svd(X, full_matrices = 0)
        th = rho * alpha * theta / (alpha + theta) ** 2
        s = s - th
        s[s < 0] = 0
        X_hat = np.matmul(np.matmul(u, np.diag(s)), v)
        return X_hat
    
    def supergradient(self, s_hat, lambda0, theta):
        """Supergradient of the Geman function."""
        return (lambda0 * theta / (s_hat + theta) ** 2)
    
    def ten2mat(self,tensor, mode):
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
    
    def mat2ten(self, mat, tensor_size, mode):
        index = list()
        index.append(mode)
        for i in range(tensor_size.shape[0]):
            if i != mode:
                index.append(i)
        return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)
    
    def GLTC_Geman(self,dense_tensor, sparse_tensor, alpha, beta, rho, theta, maxiter, mask):
        """Main function of the GLTC-Geman."""
        
        t_start = time.time()
        dim0 = sparse_tensor.ndim
        dim1, dim2, dim3 = sparse_tensor.shape
        dim = np.array([dim1, dim2, dim3])
        binary_tensor = mask.copy()
        #binary_tensor[np.where(sparse_tensor != 0)] = 1
        tensor_hat = sparse_tensor.copy()
    
        X = np.zeros((dim1, dim2, dim3, dim0)) # \boldsymbol{\mathcal{X}} (n1*n2*3*d)
        Z = np.zeros((dim1, dim2, dim3, dim0)) # \boldsymbol{\mathcal{Z}} (n1*n2*3*d)
        T = np.zeros((dim1, dim2, dim3, dim0)) # \boldsymbol{\mathcal{T}} (n1*n2*3*d)
        for k in range(dim0):
            X[:, :, :, k] = tensor_hat
            Z[:, :, :, k] = tensor_hat
    
        D1 = np.zeros((dim1 - 1, dim1)) # (n1-1)-by-n1 adjacent smoothness matrix
        for i in range(dim1 - 1):
            D1[i, i] = -1
            D1[i, i + 1] = 1
        D2 = np.zeros((dim2 - 1, dim2)) # (n2-1)-by-n2 adjacent smoothness matrix
        for i in range(dim2 - 1):
            D2[i, i] = -1
            D2[i, i + 1] = 1
        pos = np.where((dense_tensor != 0) & (sparse_tensor == 0))    
        rankp = np.zeros(maxiter)
        rse = np.zeros(maxiter)
        rse1 = mt.relative_error(tensor_hat,self.ground_truth)
        count = 0
        for iters in range(self.max_iter):
            for k in range(dim0):
                Z_hat = self.ten2mat(X[:, :, :, k] + T[:, :, :, k] / rho, k)
                Z_hat = self.gsvt_geman(Z_hat, alpha, theta, rho)
                Z[:, :, :, k] = self.mat2ten(Z_hat, dim, k)
                var = self.ten2mat(rho * Z[:, :, :, k] - T[:, :, :, k], k)
                if k == 0:
                    var0 = self.mat2ten(np.matmul(inv(beta * np.matmul(D1.T, D1) + rho * np.eye(dim1)), var), dim, k)
                elif k == 1:
                    var0 = self.mat2ten(np.matmul(inv(beta * np.matmul(D2.T, D2) + rho * np.eye(dim2)), var), dim, k)
                else:
                    var0 = Z[:, :, :, k] - T[:, :, :, k] / rho
                X[:, :, :, k] = np.multiply(1 - binary_tensor, var0) + sparse_tensor
            tensor_hat = np.mean(X, axis = 3)
        
            rse1 = mt.relative_error(tensor_hat,self.ground_truth)
            rse[iters] = rse1
            rankp[iters] = self.gemanp(tensor_hat, alpha, theta)
        
            self.x_hat = tensor_hat.copy()
            tsc = cst.tsc(self.x_hat, self.ground_truth, self.ten_ones, self.mask_indices)
            cost = tu.loss_func(self.x_hat, dense_tensor)
        
            self.rse_cost_history.append(rse1)
            self.cost_history.append(cost)
            self.tcs_cost_history.append(tsc)
            self.penalty.append(rankp[iters])
        
            t2 = time.time()
            elapsed_time = t2 - t_start
            self.elapsed_time.append(elapsed_time)
            self.x_hat_img = mt.reconstruct_image_affine_d(self.ground_truth_img, self.x_hat, self.d, self.tensor_shape)
            
            count = count + 1
            print ("iters=" + str(iters) + "; " + str(rse[iters]) + "rank penalty= " + str(rankp[iters]))   
            for k in range(dim0):
                T[:, :, :, k] = T[:, :, :, k] + rho * (X[:, :, :, k] - Z[:, :, :, k])
                X[:, :, :, k] = tensor_hat.copy()
            
            
                
            self.save_solution_scans_and_history(count, self.suffix, self.scan_mr_folder)
            
        #self.save_solution_scans(self, self.suffix, self.scan_mr_folder)
        return self.x_hat, rankp, rse
    
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
        
        # output elapsed time history
        elapsed_time_arr = []
        elapsed_time_output = OrderedDict()
        elapsed_time_indices = []
        counter = 0
        
        for item in self.elapsed_time:
            elapsed_time_arr.append(item)
            elapsed_time_indices.append(counter)
            counter = counter + 1

        elapsed_time_output['k'] = elapsed_time_indices
        elapsed_time_output['elapsed_time'] = elapsed_time_arr
        
        output_train_df = pd.DataFrame(elapsed_time_output, index=elapsed_time_indices)
        fig_id = 'elapsed_time' + '_' + self.suffix
        mrd.save_csv_by_path(output_train_df, results_folder, fig_id) 
        
        penalty_arr = []
        penalty_output = OrderedDict()
        penalty_indices = []
        counter = 0
        
        for item in self.penalty:
            penalty_arr.append(item)
            penalty_indices.append(counter)
            counter = counter + 1

        penalty_output['k'] = penalty_indices
        penalty_output['penalty'] = penalty_arr
        
        penalty_df = pd.DataFrame(penalty_output, index=elapsed_time_indices)
        fig_id = 'penalty' + '_' + self.suffix
        mrd.save_csv_by_path(penalty_df, results_folder, fig_id) 
        
        #mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.x_hat_img,0), image.index_img(self.x_miss_img, 0), self.title + " Iteration: " + str(iteration),
        #                0, self.observed_ratio, 0, 0, 2, coord=None, folder=self.images_mr_folder_iteration, iteration=-1)
    
    def save_solution_scans(self, iteration, suffix, folder): 
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.x_hat_img,0), image.index_img(self.x_miss_img, 0), self.title + " Iteration: " + str(iteration),
                        0, self.observed_ratio, 0, 0, 2, coord=None, folder=self.images_mr_folder_iteration, iteration=-1)
    
               
    def complete(self):
        self.logger.info("Opt Started")
        self.GLTC_Geman(self.ground_truth, self.x_miss, self.alpha, self.beta, self.rho, self.theta, self.max_iter, self.mask_indices)
        self.logger.info("Opt Completed")