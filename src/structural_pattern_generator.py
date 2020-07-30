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
from sklearn.model_selection import train_test_split
from nibabel.affines import apply_affine
from nilearn.image.resampling import coord_transform, get_bounds, get_mask_bounds
from nilearn.image import resample_img
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
import metric_util as mt
import data_util as du
import mri_draw_utils as mrd
import ellipsoid_masker as elpm
import ellipsoid_mask as em
import traceback
import random
import itertools
from random import seed
from random import sample
np.random.seed(0)

def generate_structural_missing_pattern(frames_count, start_n, shape):

    subject_scan_path = du.get_full_path_subject1()

    subj_img  = mt.read_image_abs_path(subject_scan_path)
    mask_img = compute_epi_mask(subject_scan_path)

    mask_img_data = np.array(mask_img.get_data())
    epi_mask = copy.deepcopy(mask_img_data);
    n = 0
    
    corrupted_volumes = {}
    corrupted_volumes_list = []
    corrupted_volumes_list_scan_numbers = []
    ts = create_frames(shape[len(shape) - 1], start_n, frames_count)

    mask_zero_indices = 0 
    for i in ts:
        target_img = image.index_img(subject_scan_path,ts[i])
        data = copy.deepcopy(np.array(target_img.get_data()))
        data[epi_mask == 1] = 0
        mask_zero_indices_count = np.count_nonzero(data==0)
        print ("Non Zero Count in Mask = " + str(mask_zero_indices_count))
        mask_zero_indices = mask_zero_indices + mask_zero_indices_count
        masked_image = mt.reconstruct_image_affine(subj_img, data)
        corrupted_volumes_list.append(masked_image)
        corrupted_volumes[ts[i]] = masked_image
        corrupted_volumes_list_scan_numbers.append(ts[i])

    counter = 0
    
    print ("Total Non Zero Count in Mask = " + str(mask_zero_indices))

    volumes_list = []

    counter = 0
    
    volumes_list = []
    for img in image.iter_img(subject_scan_path):
        print "Volume Index: " + str(counter)
        if counter in corrupted_volumes_list_scan_numbers:
            print "Adding corrupted volume to the list " + str(counter)
            volumes_list.append(corrupted_volumes[counter])
        else:
            print "Adding normal volume to the list " + str(counter)
            volumes_list.append(img)
        counter = counter + 1       

    # now generate corrupted 4D from the list
    print("I am here")
    x_corr_img = image.concat_imgs(volumes_list)
    print("I am here 2")
    #observed_ratio = mt.compute_observed_ratio(image_masked_by_ellipsoid)
    #print ("Effective Missing Ratio: " + str(observed_ratio))
    return x_corr_img, ts

def generate_structural_missing_pattern_random_frame(x0,y0,z0, x_r, y_r, z_r,frames_count, folder, shape):
    subject_scan_path = du.get_full_path_subject1()

    n = 0
    
    masked_img_file_path  = folder + "/" + "size_" + str(x_r) + "_" + str(y_r) + "_" + str(z_r) + "_scan_" + str(n)
    
    corrupted_volumes_list = []
    corrupted_volumes_list_scan_numbers = []
    corrupted_volumes_scan_numbers = {}
    corrupted_volumes = {}
    
    random_ts = create_random_frames(shape[len(shape) - 1], frames_count)
    
    for i in random_ts:
        target_img = image.index_img(subject_scan_path,random_ts[i])
        image_masked_by_ellipsoid = elpm.create_ellipsoid_mask(x0, y0, z0, x_r, y_r, z_r, target_img, masked_img_file_path)
        
       # ellipsoid = em.EllipsoidMask(x0, y0, z0, x_r, y_r, z_r, folder)
        #ellipsoid_volume = ellipsoid.volume()
        observed_ratio = mt.compute_observed_ratio(image_masked_by_ellipsoid)
        
        corrupted_volumes_list.append(image_masked_by_ellipsoid)
        corrupted_volumes_list_scan_numbers.append(random_ts[i])
        corrupted_volumes[random_ts[i]] = image_masked_by_ellipsoid
        #print ("Ellipsoid Volume: " + str(ellipsoid_volume) + "; Missing Ratio: " + str(observed_ratio))
    
    # now create corrupted 4d where fist n frames has ellipsoid missing across 10 frames
    counter = 0
    
    volumes_list = []
    for img in image.iter_img(subject_scan_path):
        print "Volume Index: " + str(counter)
        if counter in corrupted_volumes_list_scan_numbers:
            print "Adding corrupted volume to the list " + str(counter)
            volumes_list.append(corrupted_volumes[counter])
        else:
            print "Adding normal volume to the list " + str(counter)
            volumes_list.append(img)
        counter = counter + 1
        
    # now generate corrupted 4D from the list
    x_corr_img = image.concat_imgs(volumes_list)
    #observed_ratio = mt.compute_observed_ratio(image_masked_by_ellipsoid)
    #print ("Effective Missing Ratio: " + str(observed_ratio))
    return x_corr_img, random_ts


def random_gen(high, random_count):
    seed(1)
    # prepare a sequence
    sequence = [i for i in xrange(high)]
    print(sequence)
    subset = sample(sequence,random_count)
    return subset
        
def create_random_frames(total_scans, random_size):
    scans = {} 
    
    print ("total_scans: " + str(total_scans))
    print ("random_size: " + str(random_size))
    random_list = random_gen(total_scans - 1, random_size)
    print ("Random Sample: " + str(random_list))
    
    count = 0
    for ts in random_list:
        scans[count] = ts
        print ("Adding random frames: " + str("; Timepoint #: ") + str(ts))
        count = count + 1
    return scans
    
def create_frames(total_scans, start_n, size):

    scans = {}     
    print ("total_scans: " + str(total_scans))
    print ("size: " + str(size))
    count = 0
    for ts in range(size):
        scans[count] = start_n + ts
        print ("Adding sequential frames: " + str("; Timepoint #: ") + str(start_n + ts))
        count = count + 1

    return scans