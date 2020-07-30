import ttrecipes as tr
import ttrecipes.mpl

from nilearn.image import math_img
import nibabel as nib
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from nilearn import plotting
from nilearn import image
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import copy

import numpy as np
import tt


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

def reconstruct(x_hat,x_true, ten_ones, mask):
    x_reconstruct = np.multiply(mask,x_true) + np.multiply((ten_ones - mask), x_hat)
    return x_reconstruct

def tsc(x_hat,x_true, ten_ones, mask):
    nomin = np.linalg.norm(np.multiply((ten_ones - mask), (x_true - x_hat)))
    denom = np.linalg.norm(np.multiply((ten_ones - mask), x_true))
    score = nomin/denom
    return score 


subject_scan_path = "/analysis/data/subject1/swaAMAYER+cobre01_63001+M87100944+20110309at135133+RSTpre_V01_R01+CM.nii"
print ("Subject Path: " + str(subject_scan_path))

x_true_org = read_image_abs_path(subject_scan_path)
x_true_img = np.array(x_true_org.get_data())

observed_ratio = 0.20
missing_ratio = 1 - observed_ratio

mask_img = compute_epi_mask(x_true_org)
mask_img_data = np.array(mask_img.get_data())

mask_indices = get_mask(x_true_img, observed_ratio)
epi_mask = copy.deepcopy(mask_img_data)
    
mask_indices[epi_mask==0] = 1
mask_indices_count = np.count_nonzero(mask_indices==1)
print ("mask_indices_count:" + str(mask_indices_count))

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


x_true_tt = tt.tensor(x_true_img)


x_true_tt.erank

x_true_tt

shape = [53,63,46,144]
ranks = 5
xs_matrix = np.transpose(np.nonzero(mask_indices))
values = x_train[np.nonzero(mask_indices)]
ranks = [1, 5, 5, 5, 1]
spatial_ranks = [4,4,4,4]
completed4 = tr.core.inverse_distance_weighting(xs_matrix, values, shape, p=2)
x_rec = completed4.full()

tsc_score = tsc(x_rec,x_true_img, ten_ones, mask_indices)

print ("tcs_score: " + str(tsc_score)

x_hat = reconstruct_image_affine(x_true_org, x_rec)
x_hat_img = image.index_img(x_hat,1)
x_hat_image = plotting.plot_epi(x_hat_img, bg_img=None,black_bg=True, cmap='jet', cut_coords=[1, -13, 32])



