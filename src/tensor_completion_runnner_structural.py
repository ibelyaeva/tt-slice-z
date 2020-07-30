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

import metric_util as mt
import completion_riemannian as rm
import data_util as du
import mri_draw_utils as mrd
import solution as sl
import solution_writer as sw
import ellipsoid_masker as elpm
import ellipsoid_mask as em
import traceback
    
def complete_staructural4D_10():
    
    print ("4D Structural Value Pattern Simulations Completed - 10 Frames Missing started.")
    corrupted_subject_scan_path = du.corrupted4D_10_frames_path()
    subject_scan_path = du.get_full_path_subject1()
    print "Subject Path: " + str(subject_scan_path)
    print "Corrupted Subject Path 10 Frames: " + str(corrupted_subject_scan_path)
    
    title = 'fMRI 4D Scan Completion. Pattern - Structural Values. # Frames Corrupted = 10 of 144. Miss Ratio: 7%'
    pattern = 'structural'
    solution_path = "/work/pl/sch/analysis/scripts/csv_data/solution/structural/d4"
    
    folder = "solution/structural/d4"
    all_solutions = sw.SolutionWriter(folder)
    
    subject_img = mt.read_image_abs_path(subject_scan_path)
    subject_scan_img = mt.read_image_abs_path(corrupted_subject_scan_path)
    
    miss_ratio = mt.compute_observed_ratio(subject_scan_img)
    observed_ratio_comp = 1 - miss_ratio
    print ("Corrupted 4D Volume: 10 Frames " + "; Missing Ratio: " + str(miss_ratio))
    
    observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors = rm.complete_tensor_structural_pattern(subject_scan_path, corrupted_subject_scan_path, observed_ratio_comp, n=-1)
    
    x_true = image.index_img(x_true_img,1)
    x_hat = image.index_img(x_hat_img,1)
    x_miss = image.index_img(x_miss_img,1)
    
    # where save the plot    
    folder = "/work/pl/sch/analysis/scripts/figures/d4/structural/10"
    x0, y0, z0 = (0 ,-18 , 17)
    cut_coords = [x0, y0, z0]
    
    x_r, y_r, z_r = (20, 17, 15)
    radius = [x_r, y_r, z_r]
    
    try:
        mrd.draw_original_vs_reconstructed4(x_true, x_hat, x_miss, title, rel_err, observed_ratio_comp, coord=cut_coords, folder=folder, radius=radius)
    except Exception as e:
        print("in draw: " + str(traceback.format_exc()))
         
    
    missing_frame_count = 10
    
    try:
        compl_sol = save_structural_solution(observed_ratio_comp, 4, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, pattern, solution_path,  missing_frame_count)
    except Exception as e:
        print("in save solution: " + str(traceback.format_exc()))
       
    print ("Saving All Solutions ... Done.")
    
    print ("4D Structural Value Pattern Simulations Completed - 10 Frames Missing... Done.")
    
def complete_staructural4D_by_ellipse_volume():
    
    print ("4D Structural Value Pattern Simulations by Ellipse Volume Missing started.")
    folder_path = "/work/pl/sch/analysis/results/masked_images/ellipsoid_masks/type2/scans"
    
        
    path1 = "size_25_20_15_scan_0.nii"
    path2 = "size_28_23_18_scan_0.nii"
    path3 = "size_30_25_20_scan_0.nii"
    path4 = "size_33_28_23_scan_0.nii"
       
    path1_full = os.path.join(folder_path, path1)
    path2_full = os.path.join(folder_path, path2)
    path3_full = os.path.join(folder_path, path3)
    path4_full = os.path.join(folder_path, path4)
       
    corrupted_volume_list = []
    
    volume1 = mt.read_image_abs_path(path1_full)
    volume2 = mt.read_image_abs_path(path2_full)
    volume3 = mt.read_image_abs_path(path3_full)
    volume4 = mt.read_image_abs_path(path4_full)
    
    corrupted_volume_list.append(path1_full)
    corrupted_volume_list.append(path2_full)
    corrupted_volume_list.append(path3_full)
    corrupted_volume_list.append(path4_full)
   
   
    radius_list = []
    radius_list.append((25,20,15))
    radius_list.append((28,23,18))
    radius_list.append((30,25,20))
    radius_list.append((33,28,23))
   
    
    subject_scan_path = du.get_full_path_subject1()
    print "Subject Path: " + str(subject_scan_path)
    
    folder = "/work/pl/sch/analysis/scripts/figures/d4/structural/volume"
    all_solutions = sw.SolutionWriter(folder)
    
    counter = 0
    for item in corrupted_volume_list:
        subject_scan_img = mt.read_image_abs_path(item)
    
        miss_ratio = mt.compute_observed_ratio(subject_scan_img)
        observed_ratio_comp = 1 - miss_ratio
        print ("Corrupted 4D Volume: " + "; Missing Ratio: " + str(miss_ratio))
        title = 'fMRI 4D Scan Completion. Pattern - Structural Values.'
        pattern = 'structural'
        
        solution_path = "/work/pl/sch/analysis/scripts/csv_data/solution/structural/d4/volume"
                        
        observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors = rm.complete_tensor_structural_pattern(subject_scan_path, item, observed_ratio_comp, n=3)
        
        # where save the plot    
        folder = "/work/pl/sch/analysis/scripts/figures/d4/structural/volume"
        x0, y0, z0 = (10 ,10 , 9.0)
        cut_coords = [x0, y0, z0]
    
        x_r, y_r, z_r = radius_list[counter]
        radius = [x_r, y_r, z_r]
        
        try:
            mrd.draw_original_vs_reconstructed4(x_true_img, x_hat_img, x_miss_img, title, rel_err, observed_ratio_comp, coord=cut_coords, folder=folder, radius=radius)
        except Exception as e:
            print("in draw: " + str(traceback.format_exc()))
         
    
        missing_frame_count = 10
    
        try:
            compl_solution = save_structural_solution(observed_ratio_comp, 4, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, pattern, solution_path,  missing_frame_count)
        except Exception as e:
            print("in save solution: " + str(traceback.format_exc()))
        
        print ("Saving Solution ... Done.")
    
        counter = counter + 1
        
        all_solutions.solutions.append(compl_solution)
    
    print ("Saving All Solutions ...")
    all_solutions.write_all()
    print ("Saving All Solutions ... Done.")
        
    print ("4D Structural Value Pattern Simulations Completed - By Volume... Done.")
   
    
def complete_staructural4D_50():
    
    print ("4D Structural Value Pattern Simulations Completed - 50 Frames Missing started.")
    corrupted_subject_scan_path = du.corrupted4D_50_frames_path()
    subject_scan_path = du.get_full_path_subject1()
    print "Subject Path: " + str(subject_scan_path)
    print "Corrupted Subject Path 50 Frames: " + str(corrupted_subject_scan_path)
    
    title = 'fMRI 4D Scan Completion. Pattern - Structural Values. \n # Frames Corrupted = 50 of 144. Miss Ratio: 34.7%'
    pattern = 'structural'
    solution_path = "/work/pl/sch/analysis/scripts/csv_data/solution/structural/d4/50"
    
    folder = "solution/structural/d4"
      
    subject_img = mt.read_image_abs_path(subject_scan_path)
    subject_scan_img = mt.read_image_abs_path(corrupted_subject_scan_path)
    
    miss_ratio = mt.compute_observed_ratio(subject_scan_img)
    observed_ratio_comp = 1 - miss_ratio
    print ("Corrupted 4D Volume: 50 Frames " + "; Missing Ratio: " + str(miss_ratio))
    
    observed_ratio, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors = rm.complete_tensor_structural_pattern(subject_scan_path, corrupted_subject_scan_path, observed_ratio_comp, n=-1)
    
    x_true = image.index_img(x_true_img,1)
    x_hat = image.index_img(x_hat_img,1)
    x_miss = image.index_img(x_miss_img,1)
    
    # where save the plot    
    folder = "/work/pl/sch/analysis/scripts/figures/d4/structural/50"
    x0, y0, z0 = (0 ,-18 , 17)
    cut_coords = [x0, y0, z0]
    
    x_r, y_r, z_r = (20, 17, 15)
    radius = [x_r, y_r, z_r]
    
    try:
        mrd.draw_original_vs_reconstructed4(x_true, x_hat, x_miss, title, rel_err, observed_ratio_comp, coord=cut_coords, folder=folder, radius=radius)
    except Exception as e:
        print("in draw: " + str(traceback.format_exc()))
         
    
    missing_frame_count = 50
    
    try:
        save_structural_solution(observed_ratio_comp, 4, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, pattern, solution_path,  missing_frame_count)
    except Exception as e:
        print("in save solution: " + str(traceback.format_exc()))
        
    print ("Saving All Solutions ... Done.")
    
    print ("4D Structural Value Pattern Simulations Completed - 50 Frames Missing... Done.")

def plot_4D_10():
    
    folder_path = "/work/pl/sch/analysis/scripts/csv_data/solution/structural/d4/10"
    
    x_true_path = os.path.join(folder_path,"x_true_img_10.nii")
    x_hat_path = os.path.join(folder_path,"x_hat_img_10.nii")
    x_miss_path = os.path.join(folder_path,"x_miss_img_10.nii")
    
    x_true_img = mt.read_image_abs_path(x_true_path)
    x_hat_img = mt.read_image_abs_path(x_hat_path)
    x_miss_img = mt.read_image_abs_path(x_miss_path)
    
    title = 'fMRI 4D Scan Completion. Pattern - Structural Values. \n # Frames Corrupted = 10 of 144. Miss Ratio: 7%'
    
    x0, y0, z0 = (0 ,-18 , 17)
    cut_coords = [x0, y0, z0]
    
    x_r, y_r, z_r = (20, 17, 15)
    radius = [x_r, y_r, z_r]
    
    x_true = image.index_img(x_true_img,1)
    x_hat = image.index_img(x_hat_img,1)
    x_miss = image.index_img(x_miss_img,1)
    
    folder = "/work/pl/sch/analysis/scripts/figures/d4/structural/10"
    mrd.draw_original_vs_reconstructed4(x_true, x_hat, x_miss, title, 0.00013, 0.93, coord=cut_coords, folder=folder, radius = radius)
    
def plot_4D_50():
    
    folder_path = "/work/pl/sch/analysis/scripts/csv_data/solution/structural/d4/50"
    
    x_true_path = os.path.join(folder_path,"x_true_img_50.nii")
    x_hat_path = os.path.join(folder_path,"x_hat_img_50.nii")
    x_miss_path = os.path.join(folder_path,"x_miss_img_50.nii")
    
    x_true_img = mt.read_image_abs_path(x_true_path)
    x_hat_img = mt.read_image_abs_path(x_hat_path)
    x_miss_img = mt.read_image_abs_path(x_miss_path)
    
    title = 'fMRI 4D Scan Completion. Pattern - Structural Values. \n # Frames Corrupted = 50 of 144. Miss Ratio: 34.7%'
    
    
    x0, y0, z0 = (0 ,-18 , 17)
    cut_coords = [x0, y0, z0]
    
    x_r, y_r, z_r = (20, 17, 15)
    radius = [x_r, y_r, z_r]
    
    x_true = image.index_img(x_true_img,1)
    x_hat = image.index_img(x_hat_img,1)
    x_miss = image.index_img(x_miss_img,1)
    
    folder = "/work/pl/sch/analysis/scripts/figures/d4/structural/50"
    mrd.draw_original_vs_reconstructed4(x_true, x_hat, x_miss, title, 0.00279, 0.67, coord=cut_coords, folder=folder, radius = radius)
    
    
def save_solution(observed_ratio, d, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, pattern, solution_path):
    
    print ("Saving Solution ...")
    solution = sl.Solution(pattern, d, observed_ratio, x_true_img, x_hat_img, x_miss_img, rel_err, solution_errors, solution_path)
    solution.save_solution_scans()
    solution.write_summary()
    solution.write_cost()
    
    print ("Solution saved @ " + str(solution_path))
    
    return solution

def save_structural_solution(observed_ratio, d, rel_err, x_true_img, x_hat_img, x_miss_img, solution_errors, pattern, solution_path,  missing_frame_count):
    
    print ("Saving Solution ...")
    solution = sl.Solution(pattern, d, observed_ratio, x_true_img, x_hat_img, x_miss_img, rel_err, solution_errors, solution_path)
    solution.save_solution_structural_scans(solution_path, missing_frame_count)
    solution.write_summary()
    solution.write_cost()
    
    print ("Solution saved @ " + str(solution_path))
    
    return solution

def generate_structural_missing_pattern():
    subject_scan_path = du.get_full_path_subject1()
        
    print ("3D Random Missing Value Pattern Simulations has started...")
    print "Subject Path: " + str(subject_scan_path)
    
    n = 0
    # type 1 (center is the center of the image), corrupt first 10 frames
    x0, y0, z0 = (0 ,-18 , 17)
    x_r, y_r, z_r = (20, 17, 15)
    
    print "===Type 1 Experiments===="
    
    target_img = image.index_img(subject_scan_path,n)
    
    type_1_folder_path = "/work/pl/sch/analysis/results/masked_images/ellipsoid_masks/type1"
    masked_img_file_path  = type_1_folder_path + "/" + "size_" + str(x_r) + "_" + str(y_r) + "_" + str(z_r) + "_scan_" + str(n)
    
    corrupted_volumes_list = []
    corrupted_volumes_list_scan_numbers = []
    
    for i in xrange(10):
        masked_img_file_path  = type_1_folder_path + "/" + "size_" + str(x_r) + "_" + str(y_r) + "_" + str(z_r) + "_scan_" + str(i)
        target_img = image.index_img(subject_scan_path,i)
        image_masked_by_ellipsoid = elpm.create_ellipsoid_mask(x0, y0, z0, x_r, y_r, z_r, target_img, masked_img_file_path)
        
        masked_img_file_path = masked_img_file_path + ".nii"
        ellipsoid = em.EllipsoidMask(x0, y0, z0, x_r, y_r, z_r, masked_img_file_path)
        ellipsoid_volume = ellipsoid.volume()
        observed_ratio = mt.compute_observed_ratio(image_masked_by_ellipsoid)
        
        #corrupted_volumes_list.append(image_masked_by_ellipsoid)
        #corrupted_volumes_list_scan_numbers.append(i)
        print ("Ellipsoid Volume: " + str(ellipsoid_volume) + "; Missing Ratio: " + str(observed_ratio))
    
    # now create corrupted 4d where fist 10 frames has ellipsoid missing across 10 frames
    counter = 0
    
    volumes_list = []
    for img in image.iter_img(subject_scan_path):
        print "Volume Index: " + str(counter)
        if counter in corrupted_volumes_list_scan_numbers:
            print "Adding corrupted volume to the list " + str(counter)
            volumes_list.append(corrupted_volumes_list[counter])
        else:
            print "Adding normal volume to the list " + str(counter)
            volumes_list.append(img)
        counter = counter + 1
        
    # now generate corrupted 4D from the list
    corrupted4d_10 = image.concat_imgs(volumes_list)
    print "Corrupted 4D - 10 frames: " + str(corrupted4d_10)
    observed_ratio4D_10 = mt.compute_observed_ratio(corrupted4d_10)
    print ("Corrupted 4D - 10 Volume: " + "; Missing Ratio: " + str(observed_ratio4D_10))
    corr_file_path4D = du.corrupted4D_10_frames_path()
    nib.save(corrupted4d_10, corr_file_path4D)
    
    # generate 50 corrupted frames
    corrupted_volumes_list = []
    corrupted_volumes_list_scan_numbers = []
    
    for i in xrange(50):
        masked_img_file_path  = type_1_folder_path + "/" + "size_" + str(x_r) + "_" + str(y_r) + "_" + str(z_r) + "_scan_" + str(i)
        target_img = image.index_img(subject_scan_path,i)
        image_masked_by_ellipsoid = elpm.create_ellipsoid_mask(x0, y0, z0, x_r, y_r, z_r, target_img, masked_img_file_path)
        
        masked_img_file_path = masked_img_file_path + ".nii"
        ellipsoid = em.EllipsoidMask(x0, y0, z0, x_r, y_r, z_r, masked_img_file_path)
        ellipsoid_volume = ellipsoid.volume()
        observed_ratio = mt.compute_observed_ratio(image_masked_by_ellipsoid)
        
        corrupted_volumes_list.append(image_masked_by_ellipsoid)
        corrupted_volumes_list_scan_numbers.append(i)
        print ("Ellipsoid Volume: " + str(ellipsoid_volume) + "; Missing Ratio: " + str(observed_ratio))
    
    # now create corrupted 4d where fist 10 frames has ellipsoid missing across 10 frames
    counter = 0
    
    volumes_list = []
    for img in image.iter_img(subject_scan_path):
        print "Volume Index: " + str(counter)
        if counter in corrupted_volumes_list_scan_numbers:
            print "Adding corrupted volume to the list " + str(counter)
            volumes_list.append(corrupted_volumes_list[counter])
        else:
            print "Adding normal volume to the list " + str(counter)
            volumes_list.append(img)
        counter = counter + 1
        
    # now generate corrupted 4D from the list
    corrupted4d_50 = image.concat_imgs(volumes_list)
    print "Corrupted 4D - 50 frames: " + str(corrupted4d_50)
    observed_ratio4D_50 = mt.compute_observed_ratio(corrupted4d_50)
    print ("Corrupted 4D - 50 Volume: " + "; Missing Ratio: " + str(observed_ratio4D_50))
    corr_file_path4D_50 = du.corrupted4D_50_frames_path()
    nib.save(corrupted4d_50, corr_file_path4D_50)
        
            
    # type 2 (the center is shifted)
    type_2_folder_path = "/work/pl/sch/analysis/results/masked_images/ellipsoid_masks/type2"
    #/work/pl/sch/analysis/results/masked_images/sizex1_25_20_15
    
    x0, y0, z0 = (10 ,10 , 9.0)
    x_r, y_r, z_r = (25, 20,15)
            
    print "===Type 2 Experiments===="
    
    for i in (0, 3, 5):
        x_r = x_r + i
        y_r = y_r + i
        z_r = z_r + i
        n = 0
        target_img = image.index_img(subject_scan_path,n)
        masked_img_file_path  = type_2_folder_path + "/" + "size_" + str(x_r) + "_" + str(y_r) + "_" + str(z_r) + "_scan_" + str(n)
        image_masked_by_ellipsoid = elpm.create_ellipsoid_mask(x0, y0, z0, x_r, y_r, z_r, target_img, masked_img_file_path)
        
        masked_img_file_path = masked_img_file_path + ".nii"
        ellipsoid = em.EllipsoidMask(x0, y0, z0, x_r, y_r, z_r, masked_img_file_path)
        ellipsoid_volume = ellipsoid.volume()
        observed_ratio = mt.compute_observed_ratio(image_masked_by_ellipsoid)
                
        print ("Ellipsoid Volume: " + str(ellipsoid_volume) + "; Missing Ratio: " + str(observed_ratio))
  
if __name__ == "__main__":
    #generate_structural_missing_pattern()
    #complete_staructural4D_10()
    #plot_4D_10()
    #plot_4D_50()
    #complete_staructural4D_50()
    complete_staructural4D_by_ellipse_volume()
    
    