import texfig

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
from pyten.method import *
from datetime import datetime
import file_service as fs
import csv

import metric_util as mt
import completion_riemannian as rm
import data_util as du
import mri_draw_utils as mrd
import solution as sl
import solution_writer as sw

import configparser
from os import path
import logging
import metadata as mdt
import complete_tensor as ct
import tensor_util as tu

def get_image_by_index(x, index):
    return image.index_img(x, index)
    

def plot_solution(folder_path, images_folder, mr, observed_ratio, d, tsc_score, tcs_z_score):
    ext = ".nii"
    x_true_path = os.path.join(folder_path,"x_true_img_" + str(mr) + ext)
    x_miss_path = os.path.join(folder_path,"x_miss_img_" + str(mr)  + ext)
    x_hat_path = os.path.join(folder_path,"x_hat_img_" + str(mr)  + ext)
    
    ground_truth_img = mt.read_image_abs_path(x_true_path)
    x_miss_img = mt.read_image_abs_path(x_miss_path)
    x_hat_img = mt.read_image_abs_path(x_hat_path)
    
    title = str(d) + "D fMRI Tensor Completion"
    
    time0 = 0
    time1 = 1
    time119 = 119
    time143 = 143
    
    mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(ground_truth_img, 0), image.index_img(x_hat_img,0), image.index_img(x_miss_img, 0), title,
                                             tsc_score,observed_ratio, tsc_score, tcs_z_score, 2, coord=None, folder=images_folder, iteration = -1, time=0)
        
    mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(ground_truth_img, 69), image.index_img(x_hat_img,69), image.index_img(x_miss_img, 69), title,
                                             tsc_score, observed_ratio, tsc_score, tcs_z_score, 2, coord=None, folder=images_folder, iteration = -1, time=69)
        
    mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(ground_truth_img, 119), image.index_img(x_hat_img,119), image.index_img(x_miss_img, 119),title,
                                             tsc_score, observed_ratio, tsc_score, tcs_z_score, 2, coord=None, folder=images_folder, iteration = -1, time = 119)
        
    mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(ground_truth_img, 143), image.index_img(x_hat_img,143), image.index_img(x_miss_img, 143),title,
                                             tsc_score, observed_ratio, tsc_score, tcs_z_score, 2, coord=None, folder=images_folder, iteration = -1, time=143)
    


def plot_mr(folder_path, image_path,  mr, observed_ratio, tcs, tcs_z):
    #suffix = "random/images/final"
    #dir_path = os.path.join(folder_path, suffix)
    #fs.ensure_dir(dir_path)
    plot_solution(folder_path, image_path, mr, observed_ratio, 4, tcs, tcs_z)

def plot_mr10(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 10, 0.9,  tcs, tcs_z)
    
def plot_mr20(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 20, 0.8,  tcs, tcs_z)
    
def plot_mr30(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 30, 0.7,  tcs, tcs_z)
    
def plot_mr40(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 40, 0.6,  tcs, tcs_z)
    
def plot_mr50(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 50, 0.5,  tcs, tcs_z)
    
def plot_mr60(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 60, 0.4,  tcs, tcs_z)
    
def plot_mr70(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 70, 0.3,  tcs, tcs_z)
    
def plot_mr80(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 80, 0.2,  tcs, tcs_z)
    
def plot_mr90(folder_path, image_path, tcs, tcs_z):
    plot_mr(folder_path, image_path, 90, 0.1,  tcs, tcs_z)
    
if __name__ == "__main__":
    pass
    #complete_random_2D()
    
    #MR10%
    folder10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/"
    mr10path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/scans/final/mr/10"
    suffix = "random/images/final"
    image10_path = os.path.join(folder10_path, suffix)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    #plot_mr10(mr10path, image10_path, 0.002500909846, 0.004216705)
    
    #MR20
    folder20_path = "/work/scratch/tensor_completion/4D/run_2018-07-20_07_36_13/"
    mr20path = "/work/scratch/tensor_completion/4D/run_2018-07-20_07_36_13/d4/random/scans/final/mr/20"
    suffix = "random/images/final"
    image20_path = os.path.join(folder20_path, suffix)
    fs.ensure_dir(image20_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr20(mr20path, image20_path, 0.002613617, 0.004313019)
    
    
    #MR30
    folder30_path = "/work/scratch/tensor_completion/4D/run_2018-07-20_09_03_54/"
    mr30path = "/work/scratch/tensor_completion/4D/run_2018-07-20_09_03_54/d4/random/scans/final/mr/30"
    suffix = "random/images/final"
    image30_path = os.path.join(folder30_path, suffix)
    fs.ensure_dir(image30_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr30(mr30path, image30_path, 0.002711214, 0.004554085)
    
    #MR30
    folder30_path = "/work/scratch/tensor_completion/4D/run_2018-07-20_09_03_54/"
    mr30path = "/work/scratch/tensor_completion/4D/run_2018-07-20_09_03_54/d4/random/scans/final/mr/30"
    suffix = "random/images/final"
    image30_path = os.path.join(folder30_path, suffix)
    fs.ensure_dir(image30_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr30(mr30path, image30_path, 0.002711214, 0.004554085)
    
    #MR40#
    folder40_path = "/work/scratch/tensor_completion/4D/run_2018-07-20_11_07_50/"
    mr40path = "/work/scratch/tensor_completion/4D/run_2018-07-20_11_07_50/d4/random/scans/final/mr/40"
    suffix = "random/images/final"
    image40_path = os.path.join(folder40_path, suffix)
    fs.ensure_dir(image40_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr40(mr40path, image40_path, 0.00290222, 0.004560752)
    
    
    #MR50#
    folder50_path = "/work/scratch/tensor_completion/4D/run_2018-07-20_15_00_16/"
    mr50path = "/work/scratch/tensor_completion/4D/run_2018-07-20_15_00_16/d4/random/scans/final/mr/50"
    suffix = "random/images/final"
    image50_path = os.path.join(folder50_path, suffix)
    fs.ensure_dir(image50_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr50(mr50path, image50_path, 0.00290222, 0.004560752)
    
    
    #MR60#
    folder60_path = "/work/scratch/tensor_completion/4D/run_2018-07-20_16_11_43/"
    mr60path = "/work/scratch/tensor_completion/4D/run_2018-07-20_16_11_43/d4/random/scans/final/mr/60"
    suffix = "random/images/final"
    image60_path = os.path.join(folder60_path, suffix)
    fs.ensure_dir(image60_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr60(mr60path, image60_path, 0.003524534, 0.00502384)
    
    #MR70#
    folder70_path = "/work/scratch/tensor_completion/4D/run_2018-07-20_22_15_54/"
    mr70path = "/work/scratch/tensor_completion/4D/run_2018-07-20_22_15_54/d4/random/scans/final/mr/70"
    suffix = "random/images/final"
    image70_path = os.path.join(folder70_path, suffix)
    fs.ensure_dir(image70_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr70(mr70path, image70_path, 0.004206054   , 0.005138284)
    
    
    #MR80#
    folder80_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_06_33_14/"
    mr80path = "/work/scratch/tensor_completion/4D/run_2018-07-21_06_33_14/d4/random/scans/final/mr/80"
    suffix = "random/images/final"
    image80_path = os.path.join(folder80_path, suffix)
    fs.ensure_dir(image80_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr80(mr80path, image80_path, 0.004206054, 0.005138284)
    
    #MR90#
    folder90_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_18_03_23/"
    mr90path = "/work/scratch/tensor_completion/4D/run_2018-07-21_18_03_23/d4/random/scans/final/mr/90"
    suffix = "random/images/final"
    image90_path = os.path.join(folder90_path, suffix)
    fs.ensure_dir(image90_path)
    #fs.ensure_dir(dir_path)
    #image10_path = "/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/images/final"
    plot_mr90(mr90path, image90_path, 0.015419008,  0.009015141)
    

        
    
    
    
   