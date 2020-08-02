import numpy as np
from scipy import stats
from scipy.special import stdtr
import tensor_util as tu
import noise_util as nu

import nilearn

from nilearn import image
import nibabel as nib
import copy
from nilearn import plotting
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from math import ceil
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
import metric_util as mt
import data_util as du
import mri_draw_utils as mrd
import traceback
import random
import itertools
from random import seed
from random import sample
from nilearn.image import math_img
import tensor_util as tu
np.random.seed(0)

subject_scan_path = du.get_full_path_subject1()

x_true_norm_path = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/normal/x_true_norm"
x_true_path = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/normal/x_true"
file_path45 = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45"
file_path45_80 = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45-80"
file_path45_80_100 = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45-80-100"

#sp.slice_wise_fft(subject_scan_path, folder, spike_thres=4.)
def update_volume(x_img4d, x_img3d, scan_number):
        
        corrupted_volumes_list_scan_numbers = []
        replaced_frames = {}
        
        counter = 0    
        volumes_list = []
        for img in image.iter_img(x_img4d):
            print "Volume Index: " + str(counter)
            if counter in scan_number:
                print "Adding corrupted volume to the list " + str(counter)
                volumes_list.append(x_img3d)
            else:
                print "Adding normal volume to the list " + str(counter)
                volumes_list.append(img)
            counter = counter + 1
        
        x_img = image.concat_imgs(volumes_list)
        return x_img

def createtrNoisedImage(img_source, snrdb, scan_numbers, file_path):   
    
    x_img_updated = img_source
    prefix = ""
    for i in scan_numbers:
        img_data = np.array(image.index_img(img_source, i).get_data())
        img = image.index_img(img_source,i)
        x_noised = nu.add_richian_noise(img, img_data, snrdb)
        x_noised_img = mt.reconstruct_image_affine(img_source, x_noised)
        x_img_updated = update_volume(x_img_updated, x_noised_img, [i])
        prefix = prefix + "_" + str(i)
    
    file_name = "x_miss" + prefix
    path = os.path.join(file_path, file_name)
        
    nib.save(x_img_updated,path)
    return x_img_updated
    
def createTR45(x_true_img, file_path, snr=30):
    print ("snr = " + str(snr))
    x_noised45 = createtrNoisedImage(x_true_img, snr, [45],file_path)
    
def createTR4580(x_true_img, file_path, snr=30):
    print ("snr = " + str(snr))
    x_noised4580 = createtrNoisedImage(x_true_img, snr, [45,80],file_path)
    
def createTR4580100(x_true_img, file_path, snr=30):
    print ("snr = " + str(snr))
    x_noised4580100 = createtrNoisedImage(x_true_img, snr, [45,80,100],file_path)
    
    
if __name__ == "__main__":
    
    x_true_img = mt.read_image_abs_path(subject_scan_path)
    data, data_norm = tu.normalize_data(np.array(x_true_img.get_data()))
    x_true_img_norm = mt.reconstruct_image_affine(x_true_img, data)
    nib.save(x_true_img_norm ,x_true_norm_path)
    nib.save(x_true_img ,x_true_path)
    createTR45(x_true_img_norm, file_path45, snr=25)
    #createTR4580(x_true_img,file_path45_80, snr=25)
    #createTR4580100(x_true_img,file_path45_80_100, snr=25)

      
