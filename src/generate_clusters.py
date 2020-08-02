import nibabel as nb
import numpy as np
from nilearn import plotting, image
import os
import metric_util as mt
import data_util as du
from scipy import stats
from scipy import ndimage
from nilearn.masking import apply_mask

import sys
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.cluster import hierarchy
import pandas as pd

template_folder = "/work/scratch/tensor_completion/4D/noise/template"
gm_prob_file_name ="mni_icbm152_gm_tal_nlin_asym_09c.nii"
csf_prob_file_name= "mni_icbm152_wm_tal_nlin_asym_09c.nii"
wm_prob_file_name = "mni_icbm152_wm_tal_nlin_asym_09c.nii"
icbm_file_name = "mni_icbm152_t1_tal_nlin_asym_09c.nii"

gm_prob_path = os.path.join(template_folder, gm_prob_file_name)
csf_prob_path = os.path.join(template_folder, csf_prob_file_name)
wm_prob_path = os.path.join(template_folder, wm_prob_file_name)
icbm_file_path = os.path.join(template_folder, icbm_file_name)

mask_gm_path = os.path.join(template_folder, "gm_mask.nii")
subject_img_data_gm_path = "/work/scratch/tensor_completion/4D/noise/mask/subject_gm_mask.nii"

subject_scan_path = du.get_full_path_subject1()
subject_img = mt.read_image_abs_path(subject_scan_path)

gm_prob_img = mt.read_image_abs_path(gm_prob_path)
csf_prob_img = mt.read_image_abs_path(csf_prob_path)
wm_prob_img = mt.read_image_abs_path(wm_prob_path)
icmb_prob_img = mt.read_image_abs_path(icbm_file_path)

mask_gm = image.resample_to_img(gm_prob_img, subject_img, interpolation='nearest').get_data()
mask_wm = image.resample_to_img(wm_prob_img, subject_img, interpolation='nearest').get_data()
mask_csf = image.resample_to_img(csf_prob_img, subject_img, interpolation='nearest').get_data()
mask_icmb = image.resample_to_img(icmb_prob_img, subject_img, interpolation='nearest').get_data()
subject_img_data = np.array(subject_img.get_data())

def get_gm_mask(infile, treshold = 0.2):
 gm_img = mt.read_image_abs_path(infile)
 gm_data = np.array(gm_img.get_data())
 gm_mask = (gm_data > 0.2)
 gm_mask = ndimage.binary_closing(gm_mask, iterations=2)
 gm_mask_img = image.new_img_like(gm_img, gm_mask)
 return gm_mask_img

def get_wm_mask(infile, treshold = 0.2):
 wm_img = mt.read_image_abs_path(infile)
 wm_data = np.array(wm_img.get_data())
 wm_mask = (wm_data > 0.2)
 wm_mask = ndimage.binary_closing(wm_mask, iterations=2)
 wm_mask_img = image.new_img_like(wm_img, wm_mask)
 return wm_mask_img

def get_csf_mask(infile, treshold = 0.2):
 cf_img = mt.read_image_abs_path(infile)
 cf_data = np.array(cf_img.get_data())
 cf_mask = (cf_data > 0.2)
 cf_mask = ndimage.binary_closing(cf_mask, iterations=2)
 cf_mask_img = image.new_img_like(cf_img, cf_mask)
 return cf_mask_img

def resample_mask(src_img, target_img):
    resampled_img = image.resample_to_img(src_img, target_img, interpolation='continuous')
    data = resampled_img.get_data()
    data = (data > 0).astype('float32')
    return data

def reorderMatrixRows(X,distanceMetric='euclidean',linkageMethod='average',doOptimalLeaf=False):
    "Get reordering of rows of a matrix by clustering"

    # Find (and exclude) constant voxels:
    isConstant = (np.std(X,axis=1,ddof=1)==0)
    if np.any(isConstant):
        X = X[~isConstant,:]
        print('%u constant voxels ignored' % np.sum(isConstant))

    # z-score the remaining voxels:
    X = stats.zscore(X,axis=1,ddof=1)

    print('Filtered to %u x %u time series' % (X.shape[0],X.shape[1]))

    # Compute condensed pairwise distance matrix:
    # DataFrame.corr(method='Pearson',min_periods=1)
    dij = distance.pdist(X,metric=distanceMetric)
    print('%u %s distances computed!' %(dij.shape[0],distanceMetric))

    # Check D is well-behaved:
    if not np.isfinite(dij).all():
        raise ValueError('Distance matrix contains non-finite values...')

    # Compute hierarchical linkage structure:
    Z = hierarchy.linkage(dij,method=linkageMethod,optimal_ordering=doOptimalLeaf)
    print('%u objects agglomerated using average linkage clustering!' %(X.shape[0]))

    # Get voxel ordering vector:
    if np.any(isConstant):
        # Extend to the full size
        nodeOrdering = np.zeros_like(isConstant,dtype=int)
        nodeOrdering[~isConstant] = hierarchy.leaves_list(Z)
    else:
        nodeOrdering = hierarchy.leaves_list(Z)
    return nodeOrdering



x_miss_path = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45/x_miss_45.nii"
x_miss_img = mt.read_image_abs_path(x_miss_path)

ntsteps = subject_img_data.shape[-1]
X_ts = subject_img_data.reshape(-1, ntsteps)

CF_ind = 1
GM_ind = 2
WM_ind = 3
ordering = []

distanceMetric='euclidean'
linkageMethod='average'
doOptimalLeaf=False

gm_img = get_gm_mask(gm_prob_path)
wm_img = get_wm_mask(wm_prob_path)
csf_img = get_csf_mask(csf_prob_path)

subject_img_data = subject_img.get_data()
gm_mask = resample_mask(gm_img, subject_img)


X_ts_gm = subject_img_data[gm_mask==1]

nodeOrdering = reorderMatrixRows(X_ts_gm,distanceMetric=distanceMetric,
                            linkageMethod=linkageMethod,doOptimalLeaf=doOptimalLeaf)




