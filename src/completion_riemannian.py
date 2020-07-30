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
from pyten.tools.tenerror import  tenerror_omega

from pyten.method import rimrltc2 as rim, rimrltc

import mri_draw_utils as mri_d
import data_util as dtu
import metric_util as mt
from scipy import ndimage
from nilearn.masking import compute_epi_mask
np.random.seed(0)


def complete_tensor_random_pattern(data_path, observed_ratio=0.9, n=-1):
    x_true_org = mt.read_image_abs_path(data_path)
    
    if n >=0:
        x_true_img = image.index_img(x_true_org,n)
    else:
        x_true_img = x_true_org
        
    print ("Subject Data Path Location: " + str(data_path) + "\n" + "Subject Scan: " + str(x_true_img))
    
    # prepare multidimensional arrays of ground truth and training data
    x_true = np.array(x_true_img.get_data())
    #mask = (np.random.rand(x_true.shape[0],x_true.shape[1],x_true.shape[2]) < observed_ratio).astype('int') 
    mask = get_mask(x_true, observed_ratio)
    x_train = copy.deepcopy(x_true)
    x_train[mask==0] = 0.0
    
    A = Tensor(x_train)  # Construct Image Tensor to be Completed
    X0 = Tensor(np.array(x_true))  # Save the Ground Truth 
    
    x_hat_tensor, solution_errors = rimrltc(A, mask, max_iter=1000, epsilon=1e-12, alpha=None)
    x_hat = x_hat_tensor.data

    orig_X = X0
    [abs_error, rel_error] = tenerror(x_hat_tensor, orig_X, mask)
    print ("The Relative Error is:" + str(rel_error) + "; Observed Ratio: " + str(observed_ratio))
    
    #reconstruct orginal scan
    ten_ones = np.ones_like(mask)
    x_reconstr = mt.reconstruct(x_hat,x_true, ten_ones, mask)

    rel_err = mt.relative_error(x_reconstr,x_true)
    print "My Relative Error: " + str(rel_err)  
    
    x_hat_img = mt.reconstruct_image_affine(x_true_img, x_reconstr)
    x_miss_img = mt.reconstruct_image_affine(x_true_img,x_train)
    
    return observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors

def complete_tensor_random_pattern_with_init(data_path, x_init, mask, observed_ratio=0.9, n=-1):
    x_true_org = mt.read_image_abs_path(data_path)
    
    if n >=0:
        x_true_img = image.index_img(x_true_org,n)
    else:
        x_true_img = x_true_org
        
    print ("Subject Data Path Location: " + str(data_path) + "\n" + "Subject Scan: " + str(x_true_img))
    
    # prepare multidimensional arrays of ground truth and training data
    x_true = np.array(x_true_img.get_data())
    #mask = (np.random.rand(x_true.shape[0],x_true.shape[1],x_true.shape[2]) < observed_ratio).astype('int') 
    #mask = get_mask(x_true, observed_ratio)
    x_train = copy.deepcopy(x_true)
    x_train[mask==0] = 0.0
    
    A = Tensor(x_train)  # Construct Image Tensor to be Completed
    X0 = Tensor(np.array(x_true))  # Save the Ground Truth 
    x_hat_tensor, solution_errors, solution_test_errors = rim.rimrltc(A, X0, None, x_init, max_iter=1000, epsilon=1e-12, alpha=None)
    x_hat = x_hat_tensor.data

    orig_X = X0
    [abs_error, rel_error, compl_score, nrmse] = tenerror_omega(x_hat_tensor, orig_X, mask)
    print ("The Relative Error is:" + str(rel_error) + "; Observed Ratio: " + str(observed_ratio) + "; TSC Score: " + str(compl_score) + "; NRMSE: " + str(nrmse))
    
    #reconstruct orginal scan
    ten_ones = np.ones_like(mask)
    x_reconstr = mt.reconstruct(x_hat,x_true, ten_ones, mask)

    rel_err = mt.relative_error(x_reconstr,x_true)
    print "My Relative Error: " + str(rel_err)  
    
    x_hat_img = mt.reconstruct_image_affine(x_true_img, x_reconstr)
    x_miss_img = mt.reconstruct_image_affine(x_true_img,x_train)
    
    return observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, solution_test_errors, compl_score, nrmse


def complete_tensor_random_pattern3D(data_path, observed_ratio=0.9, n=-1):
    x_true_org = mt.read_image_abs_path(data_path)
    
    if n >=0:
        x_true_img = image.index_img(x_true_org,n)
    else:
        x_true_img = x_true_org
        
    print ("Subject Data Path Location: " + str(data_path) + "\n" + "Subject Scan: " + str(x_true_img))
    mask_img = compute_epi_mask(x_true_org)
    mask_img_data = np.array(mask_img.get_data())
    
    # prepare multidimensional arrays of ground truth and training data
    x_true = np.array(x_true_img.get_data())
    #mask = (np.random.rand(x_true.shape[0],x_true.shape[1],x_true.shape[2]) < observed_ratio).astype('int') 
    mask = get_mask(x_true, observed_ratio)
    epi_mask = copy.deepcopy(mask_img_data)
    
    mask[epi_mask==0] = 1
    
    x_train = copy.deepcopy(x_true)
    x_train[mask==0] = 0.0
    
    A = Tensor(x_train)  # Construct Image Tensor to be Completed
    X0 = Tensor(np.array(x_true))  # Save the Ground Truth 
    
    x_hat_tensor, solution_errors, solution_test_errors = rimrltc(A, X0,mask, max_iter=1000, epsilon=1e-12, alpha=None)
    x_hat = x_hat_tensor.data

    orig_X = X0
    [abs_error, rel_error, compl_score, nrmse] = tenerror_omega(x_hat_tensor, orig_X, mask)
    print ("The Relative Error is:" + str(rel_error) + "; Observed Ratio: " + str(observed_ratio) + "; TSC Score: " + str(compl_score) + "; NRMSE: " + str(nrmse))
    
    #reconstruct orginal scan
    ten_ones = np.ones_like(mask)
    x_reconstr = mt.reconstruct(x_hat,x_true, ten_ones, mask)

    rel_err = mt.relative_error(x_reconstr,x_true)
    print "My Relative Error: " + str(rel_err)  
    
    x_hat_img = mt.reconstruct_image_affine(x_true_img, x_reconstr)
    x_miss_img = mt.reconstruct_image_affine(x_true_img,x_train)
    
    return observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, solution_test_errors, compl_score, nrmse


def complete_tensor_random_pattern2D(data_path, observed_ratio=0.9, n=-1):
    x_true_org = mt.read_image_abs_path(data_path)
    
    if n >=0:
        x_true_img = image.index_img(x_true_org,n)
    else:
        x_true_img = x_true_org
        
    print ("Subject Data Path Location: " + str(data_path) + "\n" + "Subject Scan: " + str(x_true_img))
    
    
    # prepare multidimensional arrays of ground truth and training data
    x_true = np.array(x_true_img.get_data())
    
    print ("Subject Data Path Location: " + str(data_path) + "\n" + "Subject Scan: " + str(x_true_img))
    mask_img = compute_epi_mask(x_true_org)
    mask_img_data = np.array(mask_img.get_data())
    
    # prepare multidimensional arrays of ground truth and training data
    x_true = np.array(x_true_img.get_data())
    #mask = (np.random.rand(x_true.shape[0],x_true.shape[1],x_true.shape[2]) < observed_ratio).astype('int') 
    mask = get_mask(x_true, observed_ratio)
    epi_mask = copy.deepcopy(mask_img_data)
    
    mask[epi_mask==0] = 1
    
    x_train = copy.deepcopy(x_true)
    x_train[mask==0] = 0.0
    
    # reshape the tensor in 2D
    
    num_rows = x_true.shape[0]*x_true.shape[1]*x_true.shape[2]
    x_train2D = np.reshape(x_train, (num_rows, x_train.shape[3]))
    x_true2D = np.reshape(x_true, (num_rows,x_true.shape[3]))
    
    mask2D = np.reshape(mask, (num_rows, mask.shape[3]))
  
    
    A = Tensor(x_train2D)  # Construct Image Tensor to be Completed
    X0 = Tensor(np.array(x_true2D))  # Save the Ground Truth
    
    
    x_hat_tensor, solution_errors, solution_test_errors = rimrltc(A, X0,mask2D, max_iter=1000, epsilon=1e-12, alpha=None)
    x_hat = x_hat_tensor.data

    orig_X = X0
    [abs_error, rel_error, compl_score, nrmse] = tenerror_omega(x_hat_tensor, orig_X, mask2D)
    print ("The Relative Error is:" + str(rel_error) + "; Observed Ratio: " + str(observed_ratio) + "; TSC Score: " + str(compl_score) + "; NRMSE: " + str(nrmse))
    
    # reshape solution to the original form
    x_hat4D =  np.reshape(x_hat, (x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    
    #reconstruct orginal scan
    ten_ones = np.ones_like(mask)
    x_reconstr = mt.reconstruct(x_hat4D,x_true, ten_ones, mask)

    rel_err = mt.relative_error(x_reconstr,x_true)
    print "My Relative Error: " + str(rel_err)  
    
    x_hat_img = mt.reconstruct_image_affine(x_true_img, x_reconstr)
    x_miss_img = mt.reconstruct_image_affine(x_true_img,x_train)
    
    return observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, solution_test_errors, compl_score, nrmse

def complete_tensor_structural_pattern(true_data_path, corrupted_data_path, observed_ratio, n=-1):
    x_true_org = mt.read_image_abs_path(true_data_path)
    x_corr_org = mt.read_image_abs_path(corrupted_data_path)
    
    if n >=0:
        x_true_img = image.index_img(x_true_org,n)
    else:
        x_true_img = x_true_org
        
    print ("Subject Data Path Location: " + str(true_data_path) + "\n" + "Subject Scan: " + str(x_true_img))
    print ("Corrupted Subject Data Path Location: " + str(corrupted_data_path) + "\n" + "Subject Scan: " + str(x_corr_org))
    
    # prepare multidimensional arrays of ground truth and training data
    x_true = np.array(x_true_img.get_data())
    x_corr = np.array(x_corr_org.get_data())
   
    mask = get_structural_mask(x_corr)
    x_train = copy.deepcopy(x_true)
    x_train[mask==0] = 0.0
    
    A = Tensor(x_train)  # Construct Image Tensor to be Completed
    X0 = Tensor(np.array(x_true))  # Save the Ground Truth 
    
    x_hat_tensor, solution_errors = rimrltc(A, mask, max_iter=1000, epsilon=1e-12, alpha=None)
    x_hat = x_hat_tensor.data

    orig_X = X0
    [abs_error, rel_error] = tenerror(x_hat_tensor, orig_X, mask)
    print ("The Relative Error is:" + str(rel_error) + "; Observed Ratio: " + str(observed_ratio))
    
    #reconstruct orginal scan
    ten_ones = np.ones_like(mask)
    x_reconstr = mt.reconstruct(x_hat,x_true, ten_ones, mask)

    rel_err = mt.relative_error(x_reconstr,x_true)
    print "My Relative Error: " + str(rel_err)  
    
    x_hat_img = mt.reconstruct_image_affine(x_true_img, x_reconstr)
    x_miss_img = mt.reconstruct_image_affine(x_true_img,x_train)
    
    folder_path = "/work/pl/sch/analysis/scripts/figures/d4/structural/50"
   
    x_true_path = os.path.join(folder_path,"x_true_img")
    x_hat_path = os.path.join(folder_path,"x_hat_img")
    x_miss_path = os.path.join(folder_path,"x_miss_img")
        
   
    print("x_true_path:" + str(x_true_path))
    nib.save(x_true_img, x_true_path)
    nib.save(x_hat_img, x_hat_path)
    nib.save(x_hat_img, x_miss_path)
    
    return observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors

def complete_synthetic_image(x_true_img, observed_ratio=0.9):
       
    # prepare multidimensional arrays of ground truth and training data
    x_true = np.array(x_true_img)
    
    mask = get_mask(x_true, observed_ratio)
    x_train = copy.deepcopy(x_true)
    x_train[mask==0] = 0.0
    
    A = Tensor(x_train)  # Construct Image Tensor to be Completed
    X0 = Tensor(np.array(x_true))  # Save the Ground Truth 
    
    x_hat_tensor, solution_errors = rimrltc(A, mask, max_iter=1000, epsilon=1e-12, alpha=None)
    x_hat = x_hat_tensor.data

    orig_X = X0
    [abs_error, rel_error] = tenerror(x_hat_tensor, orig_X, mask)
    print ("The Relative Error is:" + str(rel_error) + "; Observed Ratio: " + str(observed_ratio))
    
    plt.imshow(x_hat)
    print(x_hat.shape)
    plt.show()
    
    #reconstruct orginal scan
    ten_ones = np.ones_like(mask)
    x_reconstr = mt.reconstruct(x_hat,x_true, ten_ones, mask)

    print "X_Recon"
    plt.imshow(x_reconstr)
    print(x_reconstr.shape)
    plt.show()
    
    rel_err = mt.relative_error(x_reconstr,x_true)
    print "My Relative Error: " + str(rel_err)  
    
    return observed_ratio, rel_err, x_true, x_reconstr, x_train, solution_errors
    

def get_mask(data, observed_ratio):
    
    if len(data.shape) == 3:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2]) < observed_ratio).astype('int') 
    elif len(data.shape) == 4:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2], data.shape[3]) < observed_ratio).astype('int') 
    elif len(data.shape) == 2:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1]) < observed_ratio).astype('int') 
    
    return mask_indices

def get_structural_mask(x_corr):
    
    mask_indices = np.ones_like(x_corr)
    mask_indices[x_corr == 0.0] =  0
    
    return mask_indices
        
    