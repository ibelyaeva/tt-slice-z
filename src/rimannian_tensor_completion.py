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
from skimage.measure import compare_ssim as ssim
from tensorflow.python.util import nest
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


class RiemannianTensorCompletion(object):
    
    def __init__(self, ground_truth_img, ground_truth, tensor_shape, x_init, mask_indices, z_scored_mask, 
                 sparse_observation_org,
                 norm_sparse_observation,
                 x_init_tcs,
                 ten_ones, max_tt_rank, observed_ratio, epsilon, train_epsilon, backtrack_const, logger, meta, d, z_score = 2):
        
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
        
        self.backtrack_const = backtrack_const
        
        self.init()
       
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
                  
    def init_variables(self):
        
        self.title = str(self.d) + "D fMRI Tensor Completion"
        
        self.original_shape = self.ground_truth.shape
        self.target_shape = mt.get_target_shape(self.ground_truth, self.d)
        
        self.logger.info("D = " + str(self.d) + "; Original Shape: " + str(self.original_shape) + "; Target Shape: " + str(self.target_shape))
        
        #save sparse_observation_org before reshaping
        self.sparse_observation_org = copy.deepcopy(self.ground_truth)
        self.sparse_observation_org[self.mask_indices == 0] = 0.0
        
        # save x_miss_img
        self.x_miss_img = mt.reconstruct_image_affine(self.ground_truth_img, self.sparse_observation_org)
        
        # dont' change ground truth image
        
        # reshape mask_indices
        if self.d == 3 or self.d == 2:
            self.mask_indices = mt.reshape_as_nD(self.mask_indices, self.d,self.target_shape)
            
        self.logger.info("Mask Indices Shape: " + str(self.mask_indices.shape))
        
        #create x_miss
        self.x_miss = np.array(self.sparse_observation_org)
        
        if self.d == 3 or self.d == 2:
            self.x_miss =  mt.reshape_as_nD(self.x_miss, self.d,self.target_shape)
            
        self.logger.info("Miss X Shape: " + str(self.x_miss.shape))
        
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
        
        
        # reshape self.z_scored_mask
        if self.d == 3 or self.d == 2:
            self.z_scored_mask = mt.reshape_as_nD(self.z_scored_mask, self.d,self.target_shape)
            
        self.logger.info("Z Score Mask Shape: " + str(self.z_scored_mask.shape))
        
        
        # init tensor flow variables
        
        self.ground_truth_tf_var = t3f.to_tt_tensor(self.ground_truth, self.max_tt_rank)
        self.A_var = t3f.get_variable('A', initializer=self.ground_truth_tf_var, trainable=False)
        self.ground_truth_var = tf.get_variable('ground_truth', initializer=self.ground_truth, trainable=False)
        
        
        self.sparsity_mask = tf.get_variable('sparsity_mask', initializer=self.mask_indices, trainable=False)
        self.sparsity_mask = tf.cast(self.sparsity_mask, tf.float32)
        
        self.logger.info("TF Sparsity Mask Shape: " + str(self.sparsity_mask))
        
        self.sparse_observation = self.ground_truth_var * self.sparsity_mask  
        self.logger.info("TF Sparse Observation Shape: " + str(self.sparse_observation))   
        
        self.x_reconstr_init = mt.reconstruct2(self.x_init_tcs, self.ground_truth, self.mask_indices)
        self.tsc_score_init = cst.tsc(self.x_reconstr_init, self.ground_truth, self.ten_ones, self.mask_indices).astype('float32')
        self.logger.info("TCS Score Initial Value: " + str(self.tsc_score_init))
        
        self.denom_tsc = np.linalg.norm((1.0 - self.mask_indices) * self.ground_truth)
        self.denom_tsc_tf = tf.get_variable('denom_tsc_tf', initializer=self.denom_tsc, trainable=False)
        self.denom_tsc_tf = tf.cast(self.denom_tsc_tf, tf.float32)
        
        self.normAOmegavar = tf.get_variable('normAOmega', initializer=self.norm_sparse_observation, trainable=False)
        
        # init initial tensor as Tensor Flow Variable
        self.x_train_tf = t3f.to_tt_tensor(self.x_init, max_tt_rank=self.max_tt_rank)
        self.X = t3f.get_variable('X', initializer=self.x_train_tf)
        
        # initialize estimate of X
        self.X_new = t3f.get_variable('X_new', initializer=self.x_train_tf)
        
        # tf_ones
        self.tf_ones = tf.get_variable('tf_ones', initializer=self.ten_ones, trainable=False)
        self.tf_ones = tf.cast(self.tf_ones, tf.float32)
        
        # tt_zeros
        if self.d == 2 or self.d == 3:
            self.tt_zeros = initializers.tensor_zeros(list(self.target_shape))
        else:
            self.tt_zeros = initializers.tensor_zeros(list(self.tensor_shape))
        
        # algorithm constants
        self.counter = tf.get_variable('counter', initializer=0)
        self.one_tf = tf.constant(1)
        self.zero_point1_tf = tf.constant(0.1)
        
        self.backtrack_const_tf = tf.constant(self.backtrack_const)
        
        
    def init_algorithm(self):
        self.init_variables()
        
    def init_gradient_computation(self):
        
        if self.d == 2 or self.d == 3:
            self.eta_old = tf.get_variable('eta_old', shape=self.target_shape, validate_shape=False)
            self.grad_old = tf.get_variable('grad_old', shape=self.target_shape, validate_shape=False)
        else:
            self.eta_old = tf.get_variable('eta_old', shape=self.tensor_shape, validate_shape=False)
            self.grad_old = tf.get_variable('grad_old', shape=self.tensor_shape, validate_shape=False)
        
        self.logger.info("X Shape: " + str(self.X.get_shape()))
        self.initial_cost = cst.compute_loss(self.X, self.sparsity_mask, self.sparse_observation)
        
        self.cost = tf.get_variable('cost', initializer=self.initial_cost)
        self.cost_new = tf.get_variable('cost_new', initializer=0.0)
        
        self.completion_score = tf.get_variable('completion_score', initializer=self.tsc_score_init, dtype=tf.float32)

        self.grad_full_0 = self.sparsity_mask * t3f.full(self.X) - self.sparse_observation
        self.grad_t3f_0 = t3f.to_tt_tensor(self.grad_full_0, max_tt_rank=self.max_tt_rank)

        self.gradnorm_omega_0 = t3f.frobenius_norm(self.grad_t3f_0) / (self.normAOmegavar)
        self.riemannian_grad_0 = t3f.riemannian.project(self.grad_t3f_0, self.X)
        self.riemannian_grad_0 = t3f.round(self.riemannian_grad_0, max_tt_rank=self.max_tt_rank, epsilon=1e-15)

        self.grad_t3f_old = t3f.get_variable('grad_t3f_old', initializer=self.riemannian_grad_0)
        
        self.riemannian_grad_full_0 = t3f.full(self.riemannian_grad_0)
        self.riemannian_grad_init_op = tf.assign(self.grad_old, self.riemannian_grad_full_0)

        self.eta_t3f_0 = -self.riemannian_grad_0
        self.eta_t3f_old = t3f.get_variable('eta_t3f_old', initializer=self.eta_t3f_0)
        
        self.eta_0 = -self.riemannian_grad_init_op
        self.eta_omega_0 = self.sparsity_mask * t3f.full(-self.riemannian_grad_0)

        self.alpha_0 = cst.compute_step_size(self.eta_omega_0, self.riemannian_grad_init_op)
        self.eta_op_0 = tf.assign(self.eta_old, self.eta_0, validate_shape=False)

        self.eta_norm = tf.get_variable('eta_norm', initializer=0.0)
        self.eta_norm_init_op = tf.assign(self.eta_norm, cst.frobenius_norm_tf(self.eta_op_0))

        self.alpha_old = tf.get_variable('alpha_old', initializer=0.0)

        self.train_step_0 = t3f.assign(self.X, t3f.round(self.X - self.alpha_0 * self.eta_t3f_0, max_tt_rank=self.max_tt_rank))
        
        self.logger.info("Rim 0:" + str(self.riemannian_grad_0))
        self.logger.info("Eta 0:" + str(self.eta_t3f_0))
        
# section of riemannian computations     
        
    def get_theta(self):
        return t3f.full(self.ip_xitrans_xi / self.inner_product_rim_grad)

    def get_value_zero_point1(self):
        return self.zero_point1_tf
    
    def get_conj_dir(self):
        self.logger.info("conjugate gradient")
        tf.Print(self.zero_point1_tf, [self.zero_point1_tf], message="conjugate gradient")
        return t3f.full(approximate.add_n([-self.riemannian_grad, self.beta * self.eta_trans], max_tt_rank=self.max_tt_rank))

    def get_rim_grad(self):
        self.logger.info("steepest descent")
        tf.Print(self.zero_point1_tf, [self.zero_point1_tf], message="steepest descent")
        return t3f.full(-self.riemannian_grad)  
    
    
    def define_train_operations(self):
        self.counter_step = tf.assign(self.counter, self.counter + 1)

        self.grad_trans = t3f.riemannian.project(self.grad_t3f_old, self.X)
        self.eta_trans = t3f.riemannian.project(self.eta_t3f_old, self.X)
    
        self.grad_full = self.sparsity_mask * t3f.full(self.X) - self.sparse_observation
        self.grad_t3f = t3f.to_tt_tensor(self.grad_full, max_tt_rank=self.max_tt_rank)

        self.loss = 0.5 * t3f.frobenius_norm_squared(self.grad_t3f)
        self.gradnorm_omega = t3f.frobenius_norm(self.grad_t3f) / (self.normAOmegavar)

        self.riemannian_grad = t3f.round(t3f.riemannian.project(self.grad_t3f, self.X), max_tt_rank=self.max_tt_rank)

        self.riemannian_grad_norm = t3f.frobenius_norm(self.riemannian_grad)

        self.inner_product_rim_grad = t3f.flat_inner(self.riemannian_grad, self.riemannian_grad)
        self.ip_xitrans_xi = t3f.flat_inner(self.grad_trans, self.riemannian_grad)
        self.theta = self.ip_xitrans_xi / self.inner_product_rim_grad
        
        # compute beta
        self.inner_product_rim_grad_old = t3f.flat_inner(self.grad_t3f_old, self.grad_t3f_old)
        self.beta = tf.maximum(0.0, (self.inner_product_rim_grad - self.ip_xitrans_xi) / self.inner_product_rim_grad_old)
        # eta = -riemannian_grad + beta*eta_trans
        
        self.logger.info("Rim Grad shape: " + str(self.riemannian_grad))
        self.logger.info("eta_trans: " + str(self.eta_trans))

        self.eta_cond = tf.cond((tf.abs(self.theta) >= self.zero_point1_tf), self.get_rim_grad, self.get_conj_dir)
        self.eta_norm_update_op = tf.assign(self.eta_norm, cst.frobenius_norm_tf(self.eta_cond))

        self.eta = t3f.to_tt_tensor(self.eta_cond, max_tt_rank=self.max_tt_rank)

        self.inprod_grad_eta = t3f.flat_inner(self.riemannian_grad, self.eta)
        self.logger.info("eta: " + str(self.eta))
        
        self.eta_omega = self.sparsity_mask * t3f.full(self.eta)
        self.alpha = cst.compute_step_size(self.eta_omega, self.grad_full)
        
        self.train_new_step = t3f.assign(self.X_new, t3f.round(self.X + self.alpha * self.eta, max_tt_rank=self.max_tt_rank))
        
        self.updated_cost_new = cst.compute_loss(self.train_new_step, self.sparsity_mask, self.sparse_observation)
        self.cost_new_op = tf.assign(self.cost_new, self.updated_cost_new)
        
        self.tsc_score_new = cst.tsc_tf(t3f.full(self.train_new_step), self.tf_ones, self.sparsity_mask, self.sparse_observation, self.denom_tsc_tf)
        self.tsc_score_op = tf.assign(self.completion_score, self.tsc_score_new)
        
        self.train_step = t3f.assign(self.X, t3f.round(self.X + self.alpha * self.eta, max_tt_rank=self.max_tt_rank))
        
        self.tsc_score_update = cst.tsc_tf(t3f.full(self.X_new), self.tf_ones, self.sparsity_mask, self.sparse_observation, self.denom_tsc_tf)
        self.tsc_score_update_op = tf.assign(self.completion_score, self.tsc_score_update)

        self.updated_cost = cst.compute_loss(self.train_step, self.sparsity_mask, self.sparse_observation)
        self.cost_op = tf.assign(self.cost, self.updated_cost)
        
        self.eta_update_op = t3f.assign(self.eta_t3f_old, self.eta, use_locking=True)
        self.grad_update_op = t3f.assign(self.grad_t3f_old, self.riemannian_grad)
        
        
    def complete(self):
            
        self.logger.info("Starting Tensor Completion. Tensor Dimension:" + str(len(self.ground_truth)) + "; Tensor Shape: " + str(self.tensor_shape) + "; Max Rank: " + str(self.max_tt_rank))
        self.init_algorithm()
        self.init_gradient_computation()
        self.define_train_operations()
            
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
            
        gradnorm_val, alpha_val, eta_op_0_val, cost_val, cost_new_val, eta_norm_val, tsc_score_val, _, _ = self.sess.run([self.gradnorm_omega_0, self.alpha_0, self.eta_op_0, self.cost,
                                                                                                                      self.cost_new, self.eta_norm, self.completion_score, self.train_step_0.op, self.eta_norm_init_op.op])
        print gradnorm_val, alpha_val, cost_val, cost_new_val, eta_norm_val, tsc_score_val
        self.logger.info("tsc_score_0:" + str(tsc_score_val))
        
        #tcs history
        self.tcs_cost_history.append(tsc_score_val)
        
        #train history
        self.train_cost_history.append(gradnorm_val)
        
        #Compute x_hat at iteration 0
        self.x_hat = mt.reconstruct2(self.sess.run(t3f.full(self.X)), self.ground_truth, self.mask_indices)
        self.x_hat_img = mt.reconstruct_image_affine_d(self.ground_truth_img, self.x_hat, self.d, self.tensor_shape)
        
        #tcs z score history        
        tcs_z_score = tu.tsc_z_score(self.x_hat,self.ground_truth, self.ten_ones, self.mask_indices, self.z_scored_mask)
        self.tcs_z_scored_history.append(tcs_z_score)
        
        self.logger.info("tcs_z_score 0:" + str(tcs_z_score))
        
        #rse cost history
        rse_cost = mt.relative_error(self.x_hat,self.ground_truth)
        self.rse_cost_history.append(rse_cost)
        self.logger.info("rse cost 0:" + str( rse_cost))
        
        self.save_solution_scans_iteration(self.suffix, self.scan_mr_iteration_folder, 0)
        
        i = 0
        cost_nan = False
        self.logger.info("Epsilon: " + str(self.epsilon))
        while gradnorm_val > self.epsilon: 
        #for k in range(2):
            i = i + 1
            F_v, gradnorm_val, alpha_val, theta_val, beta_val, cost_new_val, cost_val, tsc_score_val, eta_norm_val, inprod_grad_eta_val, riemannian_grad_norm_val, _, _, _, _, _, _ = self.sess.run([self.loss, self.gradnorm_omega, self.alpha,
                           self.theta, self.beta, self.cost_new, self.cost,
                           self.completion_score, self.eta_norm, self.inprod_grad_eta, self.riemannian_grad_norm,
                           self.eta_norm_update_op.op, self.cost_new_op.op, self.train_new_step.op,
                           self.tsc_score_update_op.op, self.eta_update_op.op, self.grad_update_op.op])
    
            print "alpha_val: " + str(alpha_val)
            print "theta_val: " + str(theta_val)
            print "inprod_grad_eta_val: " + str(inprod_grad_eta_val)
            print "rim_grad_norm: " + str(riemannian_grad_norm_val)
            print "eta_norm_val: " + str(eta_norm_val) 
            
            lr = alpha_val
            cost_prev_value = cost_val
            cost_new_value = cost_new_val
            cost_0 = cost_val
    
            self.logger.info("Cost New: " + str(cost_new_value))
            self.logger.info("Cost Old: " + str(cost_prev_value))
        
            tsc_score_old = self.tcs_cost_history[i - 1]
            tcs_z_score_old = self.tcs_z_scored_history[i-1]
            rse_cost_old = self.rse_cost_history[i-1]
            
            self.x_hat = mt.reconstruct2(self.sess.run(t3f.full(self.X_new)), self.ground_truth, self.mask_indices)
            self.x_hat_img = mt.reconstruct_image_affine_d(self.ground_truth_img, self.x_hat, self.d, self.tensor_shape)
            
            tsc_score_new = cst.tsc(self.x_hat, self.ground_truth, self.ten_ones, self.mask_indices).astype('float32')
            tcs_z_score = tu.tsc_z_score(self.x_hat,self.ground_truth, self.ten_ones, self.mask_indices, self.z_scored_mask)

            self.logger.info("TSC Score New: " + str(tsc_score_new))
            self.logger.info("TSC Score Old: " + str(tsc_score_old))
            
            
            self.logger.info("TSC Z Score New: " + str(tcs_z_score))
            self.logger.info("TSC Z Score Old: " + str(tcs_z_score_old))
                  
            backtrackiter = 0

            max_iter_count = 10
            armijo = (cost_new_val <= cost_prev_value + self.backtrack_const*(lr)*inprod_grad_eta_val)
            #armijo = ((tsc_score_new < tsc_score_old) or (cost_new_val <= cost_prev_value + self.backtrack_const * (lr) * inprod_grad_eta_val))
            self.logger.info("armijo: " + str(armijo))
    
            proposed_initial_alpha = 2 * (cost_new_value - cost_prev_value) / inprod_grad_eta_val
            self.logger.info("proposed_initial_alpha: " + str(proposed_initial_alpha))
            
            while ((cost_new_value > cost_prev_value + self.backtrack_const * (lr) * inprod_grad_eta_val)):
        
                self.logger.info("Backtracking")
                lr = 0.5 * lr
                self.logger.info("Learning Rate: " + str(lr))
                self.sess.run(t3f.assign(self.X_new, t3f.round(self.X + lr * self.eta_t3f_old, max_tt_rank=self.max_tt_rank)).op)
                x_new_val = self.sess.run(t3f.full(self.X_new))
                x_new_val[self.mask_indices == 0] = 0.0
                      
                
                self.logger.info("proposed_initial_alpha: " + str(proposed_initial_alpha))
                proposed_step_size = proposed_initial_alpha / inprod_grad_eta_val
                self.logger.info("proposed_step_size: " + str(proposed_step_size))
                proposed_initial_alpha = 0.5 * proposed_step_size
        
                cost_new_value = tu.loss_func(x_new_val, self.mask_indices)
                self.sess.run(tf.assign(self.cost_new, cost_new_value).op)
        
                tsc_score_new, _ = self.sess.run([self.completion_score, self.tsc_score_update_op.op])
                self.logger.info("TSC Score New: " + str(tsc_score_new))
                self.logger.info("TSC Score Old: " + str(tsc_score_old))
        
                if tu.is_nan(cost_new_value):
                    cost_nan = True
                    break
        
                self.logger.info("Cost New: " + str(cost_new_value))
                self.logger.info("Cost Old: " + str(cost_prev_value))
                backtrackiter = backtrackiter + 1
                self.logger.info("backtrackiter_count: " + str(backtrackiter))
    
                if cost_nan:
                    self.logger.info("Cost is Nan. Breaking after " + str(i) + "; iterations")
                    break
        
                if backtrackiter >= max_iter_count:
                    self.logger.info("Breaking BackTracking after " + str(i) + "; iterations")
                    break
            
            if (cost_new_value < cost_prev_value):
                # update x
                self.logger.info("updating X: initial cost: " + str(cost_prev_value) + "; New Cost: " + str(cost_new_value))
                self.sess.run([t3f.assign(self.X, self.X_new).op, tf.assign(self.cost, cost_new_value).op])
                              
                self.x_hat = mt.reconstruct2(self.sess.run(t3f.full(self.X)), self.ground_truth, self.mask_indices)
                self.x_hat_img = mt.reconstruct_image_affine_d(self.ground_truth_img, self.x_hat, self.d, self.tensor_shape)
                
                # recompute tcs_score for accepted x_hat
                tsc_score_new = cst.tsc(self.x_hat, self.ground_truth, self.ten_ones, self.mask_indices).astype('float32')
                self.tcs_cost_history.append(tsc_score_new)
                self.cost_history.append(cost_new_value)
                
                # recompute tcs_z_score for accepted x_hat
                tcs_z_score = tu.tsc_z_score(self.x_hat,self.ground_truth, self.ten_ones, self.mask_indices, self.z_scored_mask)
                self.tcs_z_scored_history.append(tcs_z_score)
                
                rse_cost = mt.relative_error(self.x_hat,self.ground_truth)
                self.rse_cost_history.append(rse_cost)
                
                self.logger.info("Accepted TCS Score: " + str(tsc_score_new) + "; Iteration #: " + str(i))
                self.logger.info("Accepted TCS Z Score: " + str(tcs_z_score) + "; Iteration #: " + str(i))
                self.logger.info("Accepted Training Cost: " + str(cost_new_value) + "; Iteration #: " + str(i))
                self.logger.info("Accepted RSE Cost: " + str(rse_cost) + "; Iteration #: " + str(i))
            
            else:
                self.logger.info("Reject Step: not updating X: initial cost: " + str(cost_prev_value) + "; New Cost: " + str(cost_new_value))
                self.tcs_cost_history.append(tsc_score_old)
                self.cost_history.append(cost_prev_value)
                self.tcs_z_scored_history.append(tcs_z_score_old)
                self.rse_cost_history.append(rse_cost_old)
                
                self.logger.info("Saving previous TCS Score: " + str(tsc_score_old) + "; Iteration #: " + str(i))
                self.logger.info("Saving previous TCS Z Score: " + str(tcs_z_score_old) + "; Iteration #: " + str(i))
                self.logger.info("Saving previous Cost: " + str(cost_prev_value) + "; Iteration #: " + str(i))  
                self.logger.info("Saving previous RSE Cost: " + str(rse_cost_old) + "; Iteration #: " + str(i))   
        
            # train cost history
            self.train_cost_history.append(gradnorm_val)
            
            self.logger.info("Train Cost : " + str(self.train_cost_history[i]) + "; Iteration #: " + str(i))
            
            self.save_solution_scans_iteration(self.suffix, self.scan_mr_iteration_folder, i)
            
            
            if i > 1:
                diff_train = np.abs(self.train_cost_history[i] - self.train_cost_history[i - 1]) / np.abs(self.train_cost_history[i])
                print (F_v, i, gradnorm_val, diff_train, alpha_val, theta_val, beta_val, cost_val, cost_new_val)
                
                self.logger.info("gradnorm : " + str(gradnorm_val) +
                                    "; diff train: " + str(diff_train) +
                                     "; alpha" + str(alpha_val) + 
                                     "; theta = " + str(theta_val) +
                                     "; cost : " + str(cost_val) + 
                                     "; cost_new: " + str(cost_new_val) +
                                     "; beta : " + str(beta_val))
                   
                if diff_train <= self.train_epsilon:
                    self.logger.info("Optimization Completed. Breaking after " + str(i) + " iterations" + "; Reason Relative Tolerance of Training Iterations Exceeded Trheshold: " + str(self.train_epsilon))
                    break
            else:
                print (F_v, i, gradnorm_val, alpha_val, theta_val, beta_val, cost_val, cost_new_val)
                
                self.logger.info("gradnorm : " + str(gradnorm_val) +
                                     "; alpha" + str(alpha_val) + 
                                     "; theta = " + str(theta_val) +
                                     "; cost : " + str(cost_val) + 
                                     "; cost_new: " + str(cost_new_val) +
                                     "; beta : " + str(beta_val))
            
            self.logger.info("Current Iteration #:" + str(i))
                
        self.logger.info("Optimization Completed After Iterations #:" + str(i))
        
        # compute final results after algorithm completion    
        self.x_hat = self.sess.run(t3f.full(self.X))
        self.x_hat_img = mt.reconstruct_image_affine_d(self.ground_truth_img, self.x_hat, self.d, self.tensor_shape)
             
        self.images_folder = self.meta.images_folder
        
        
        self.tcs_z_score = tu.tsc_z_score(self.x_hat,self.ground_truth, self.ten_ones, self.mask_indices, self.z_scored_mask)
        self.tsc_score = cst.tsc(self.x_hat, self.ground_truth, self.ten_ones, self.mask_indices)
        
        self.logger.info("Observed Ratio: " + str(self.observed_ratio))
        self.logger.info("Final TCS Z-Score: " + str(self.tcs_z_score))
        self.logger.info("Final TCS Score: " + str(self.tsc_score))

        
        output_cost = OrderedDict()
        indices = []

        cost_arr = []
        tsc_arr = []
        tsc_z_score_arr = []
        
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
        
        # output z-score
        tsc_z_score_output = OrderedDict()
        tsc_z_score_indices = []
        
        counter = 0
        for item in self.tcs_z_scored_history:
            tsc_z_score_arr.append(item)
            tsc_z_score_indices.append(counter)
            counter = counter + 1

        tsc_z_score_output['k'] = tsc_z_score_indices
        tsc_z_score_output['tsc_z_cost'] = tsc_z_score_arr
        
        output_tsc_z_df = pd.DataFrame( tsc_z_score_output, index=tsc_z_score_indices)
        fig_id = 'tsc_z_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_tsc_z_df, results_folder, fig_id) 
        
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
        
        output_rse_df = pd.DataFrame( rse_output, index=rse_indices)
        fig_id = 'rse_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_rse_df, results_folder, fig_id) 
        
        # output train history
        train_arr = []
        train_output = OrderedDict()
        train_indices = []
        counter = 0
        
        for item in self.train_cost_history:
            train_arr.append(item)
            train_indices.append(counter)
            counter = counter + 1

        train_output['k'] = train_indices
        train_output['train_cost'] = train_arr
        
        output_train_df = pd.DataFrame(train_output, index=train_indices)
        fig_id = 'train_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_train_df, results_folder, fig_id) 
        
        
        # save final solution scans
        self.save_solution_scans(self.suffix, self.scan_mr_folder)
        
        
    def save_solution_scans(self, suffix, folder): 
        
        self.logger.info("Missing Ratio: " + str(self.missing_ratio))
        x_true_path = os.path.join(folder,"x_true_img_" + str(suffix))
        x_hat_path = os.path.join(folder,"x_hat_img_" + str(suffix))
        x_miss_path = os.path.join(folder,"x_miss_img_" + str(suffix))
        
        self.logger.info("x_hat_path: " + str(x_hat_path))
        nib.save(self.x_hat_img, x_hat_path)
            
        self.logger.info("x_miss_path: " + str(x_miss_path))
        nib.save(self.x_miss_img, x_miss_path)
            
        self.logger.info("x_true_path: " + str(x_true_path))
        nib.save(self.ground_truth_img, x_true_path)
        
        #if self.d==3:
        #        mrd.draw_original_vs_reconstructed_rim_z_score(self.ground_truth_img, self.x_hat_img, self.x_miss_img, self.title,
        #           self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=0)
        
         #       mrd.draw_original_vs_reconstructed_rim_z_score(self.ground_truth_img, self.x_hat_img, self.x_miss_img, self.title,
         #           self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=69)
        
         #       mrd.draw_original_vs_reconstructed_rim_z_score(self.ground_truth_img, self.x_hat_img, self.x_miss_img, self.title,
         #           self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time = 119)
        
         #       mrd.draw_original_vs_reconstructed_rim_z_score(self.ground_truth_img, self.x_hat_img, self.x_miss_img, self.title,
         #           self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=143)
        #else:
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.x_hat_img,0), image.index_img(self.x_miss_img, 0), self.title,
                    self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=0)
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 69), image.index_img(self.x_hat_img,69), image.index_img(self.x_miss_img, 69), self.title,
                    self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=69)
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 119), image.index_img(self.x_hat_img,119), image.index_img(self.x_miss_img, 119), self.title,
                    self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time = 119)
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 143), image.index_img(self.x_hat_img,143), image.index_img(self.x_miss_img, 143), self.title,
                    self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=143)
        
    def save_solution_scans_iteration(self, suffix, folder, iteration): 
        
        self.logger.info("Missing Ratio: " + str(self.missing_ratio))
        self.logger.info("Suffix: " + str(suffix))
        self.logger.info("Folder: " + str(folder))
        self.logger.info("Iteration: " + str(iteration))
        
        x_true_path = os.path.join(folder,"x_true_img_" + str(suffix) + '_' + str(iteration))
        x_hat_path = os.path.join(folder,"x_hat_img_" + str(suffix) + '_' + str(iteration))
        x_miss_path = os.path.join(folder,"x_miss_img_" + str(suffix) + '_' + str(iteration))
        
        self.logger.info("x_hat_path: " + str(x_hat_path))
        nib.save(self.x_hat_img, x_hat_path)
            
        self.logger.info("x_miss_path: " + str(x_miss_path))
        nib.save(self.x_miss_img, x_miss_path)
            
        self.logger.info("x_true_path: " + str(x_true_path))
        nib.save(self.ground_truth_img, x_true_path)
        
        
        #if self.d==3:
        #        mrd.draw_original_vs_reconstructed_rim_z_score(self.ground_truth_img, self.x_hat_img, self.x_miss_img, self.title + " Iteration: " + str(iteration),
        #                self.tcs_cost_history[iteration], self.observed_ratio, self.tcs_cost_history[iteration], self.tcs_z_scored_history[iteration], 2, coord=None, folder=self.images_mr_folder_iteration, iteration=iteration)
        #else:
            
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.x_hat_img,0), image.index_img(self.x_miss_img, 0), self.title + " Iteration: " + str(iteration),
                        self.tcs_cost_history[iteration], self.observed_ratio, self.tcs_cost_history[iteration], self.tcs_z_scored_history[iteration], 2, coord=None, folder=self.images_mr_folder_iteration, iteration=iteration)
            
        
    
        

        
        
        
        
