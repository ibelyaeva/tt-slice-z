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

from pyten.method import rimrltc2 as rim, rimrltc, cp_als

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
    
    r = 63  # Rank for CP-based methods

    [x_hat_tensor, als_x_hat_tensor] = cp_als(A, r, mask)
    x_hat =  als_x_hat_tensor.data


    orig_X = X0
    
    [abs_error, rel_error, compl_score, nrmse] = tenerror_omega(als_x_hat_tensor, orig_X, mask)
    print ("The Relative Error is:" + str(rel_error) + "; Observed Ratio: " + str(observed_ratio) + "; TSC Score: " + str(compl_score) + "; NRMSE: " + str(nrmse))   
    
    #reconstruct orginal scan
    ten_ones = np.ones_like(mask)
    x_reconstr = mt.reconstruct(x_hat,x_true, ten_ones, mask)

    rel_err = mt.relative_error(x_reconstr,x_true)
    print "My Relative Error: " + str(rel_err)  
    
    x_hat_img = mt.reconstruct_image_affine(x_true_img, x_reconstr)
    x_miss_img = mt.reconstruct_image_affine(x_true_img,x_train)
    
    return observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img


def get_mask(data, observed_ratio):
    
    if len(data.shape) == 3:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2]) < observed_ratio).astype('int') 
    elif len(data.shape) == 4:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2], data.shape[3]) < observed_ratio).astype('int') 
    elif len(data.shape) == 2:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1]) < observed_ratio).astype('int') 
    
    return mask_indices
        
    