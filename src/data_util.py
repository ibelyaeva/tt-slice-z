import nilearn

from nilearn import image
import nibabel as nib
import copy
from nilearn import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from math import ceil
from nilearn.datasets import MNI152_FILE_PATH
from nibabel.affines import apply_affine
from nilearn.image.resampling import coord_transform, get_bounds, get_mask_bounds
from nilearn.image import resample_img
from nilearn.masking import compute_background_mask
import ellipsoid_mask as elm
import metric_util as mt


folder_path_subject1 = "/home/ec2-user/analysis/data/subject1"
data_path_subject1   = "swaAMAYER+cobre01_63001+M87100944+20110309at135133+RSTpre_V01_R01+CM.nii"

ellipsoid_masks_folder = "/work/pl/sch/analysis/results/masked_images/ellipsoid_masks"
ellipsoid_mask1_path = "size_20_7_15.nii"
ellipsoid_mask2_path = "size_20_11_25.nii"
ellipsoid_mask3_path = "size_35_20_25.nii"

def get_folder_subject1():
    folder_path_subject1 = "/work/pl/sch/analysis/data/COBRE001"
    return folder_path_subject1

def corrupted4D_10_frames_path():
    path = "/work/pl/sch/analysis/results/masked_images/d4/10/subject1_10.nii"
    return path

def corrupted4D_20_frames_path():
    path = "/work/pl/sch/analysis/results/masked_images/d4/20/subject1_20.nii"
    return path

def corrupted4D_50_frames_path():
    path = "/work/pl/sch/analysis/results/masked_images/d4/50/subject1_50.nii"
    return path

def get_path_subject1():
    data_path_subject1   = "swaAMAYER+cobre01_63001+M87100944+20110309at135133+RSTpre_V01_R01+CM.nii"
    return data_path_subject1

def get_full_path_subject1():
    subject_abs_path  = os.path.join(get_folder_subject1(), get_path_subject1())
    return subject_abs_path

def get_ellipse_mask_img_20_7_15():
    ellipsoid = elm.EllipsoidMask(0, -18, 17, 20, 17, 15)
    path = os.path.join(ellipsoid_masks_folder, ellipsoid_mask1_path)
    mask_img = mt.read_image_abs_path(path)
    print("Ellipsoid :" + "; x_r =" + str(ellipsoid.x_r) + "; y_r=" + str(ellipsoid.y_r) + "; z_r = " + str(ellipsoid.z_r) + "; Volume =" + str(ellipsoid.volume))
    return mask_img

