import tensorly as tl
import numpy as np

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from nilearn import image
import copy
from nilearn import plotting
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from collections import OrderedDict
import pandas as pd
from scipy import stats
from nilearn.image import math_img
import nibabel as nib
import copy
from nilearn import image
from tensorly.decomposition import robust_pca


from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from nilearn import plotting

def get_mask(data, observed_ratio):
    
    if len(data.shape) == 3:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2]) < observed_ratio).astype('int') 
    elif len(data.shape) == 4:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2], data.shape[3]) < observed_ratio).astype('int') 
    elif len(data.shape) == 2:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1]) < observed_ratio).astype('int') 
    
    return mask_indices

def read_image_abs_path(path):
    img = nib.load(path)
    return img

def reconstruct_image_affine(img_ref, x_hat):
    result = nib.Nifti1Image(x_hat, img_ref.affine)
    return result


subject_scan_path = "/work/pl/sch/analysis/data/COBRE001/swaAMAYER+cobre01_63001+M87100944+20110309at135133+RSTpre_V01_R01+CM.nii"
print ("Subject Path: " + str(subject_scan_path))
x_true_org = read_image_abs_path(subject_scan_path)

x_true_img = np.array(x_true_org.get_data())

mask_img = compute_epi_mask(x_true_org)
mask_img_data = np.array(mask_img.get_data())

observed_ratio = 0.95
missing_ratio = 1 - observed_ratio

mask_indices = get_mask(x_true_img, observed_ratio)
epi_mask = copy.deepcopy(mask_img_data)
    
mask_indices[epi_mask==0] = 1

norm_ground_truth = np.linalg.norm(x_true_img)
x_true_img = x_true_img * (1./norm_ground_truth)


norm_ground_truth = np.linalg.norm(x_true_img)
x_true_img = x_true_img * (1./norm_ground_truth)
ten_ones = np.ones_like(mask_indices)
x_train = copy.deepcopy(x_true_img)
x_train[mask_indices==0] = 0.0

x_init = copy.deepcopy(x_train)

x_org = reconstruct_image_affine(x_true_org, x_true_img)
x_org_img = image.index_img(x_org,1)
org_image = plotting.plot_epi(x_org_img, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)

x_miss_img = reconstruct_image_affine(x_true_org, x_train)
x_miss = image.index_img(x_miss_img,1)
x_miss_image = plotting.plot_epi(x_miss, bg_img=None,black_bg=True, cmap='jet', cut_coords=[1, -13, 32]) 

low_rank_part, sparse_part = robust_pca(x_train, reg_E=0.04, learning_rate=1.2, n_iter_max=20)

