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
import matplotlib.gridspec as gridspec
import matplotlib

import metric_util as mc
import math

import mri_draw_utils as mrd

def draw_n_frames(x_true_img, x_hat_img, x_miss_img, n, plot_title, relative_error, observed_ratio, coord=None, folder=None, radius = None):
    
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = n
    
    fg_color = 'white'
    bg_color = 'black'


    if coord:
        x0, y0, z0 = coord
        
    if radius:
        x_r, y_r, z_r = radius
              
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=11)
    
    miss_title = None
    if coord and radius:
        miss_title = "3D Ellipsoid Mask" + " Center: " + str(x0) + ", "+ str(y0) + ", "+  str(z0)+ "; Radius: " + str(x_r) + ", "+ str(y_r) + ", "+  str(z_r)
    
    if plot_title and miss_title:
        fig_title = plot_title + "\n" + miss_title
        fig.suptitle(fig_title, color=fg_color, fontweight='normal', fontsize=11)
    
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)

    
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = mrd.formatted_percentage(missing_ratio, 2)                    
    relative_error_str = "{0:.5f}".format(relative_error)    
    
    print("RSE= " + relative_error_str)
    
    miss_title = None
    if coord and radius:
        miss_title = "3D Ellipsoid Mask" + " Center: " + str(x0) + ", "+ str(y0) + ", "+  str(z0)+ "; Radius: " + str(x_r) + ", "+ str(y_r) + ", "+  str(z_r)
    
    grid_size = (grid_rows, grid_cols)
    cut = [x0, z0]
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            ax = fig.add_subplot(grid[i, j])
            if i == 0:
                img = image.index_img(x_true_img, j)
            elif i == 1:
                img = image.index_img(x_miss_img, j)       
            else:
                img = image.index_img(x_hat_img, j)
                
                ax.set_title("RSE=" + str(relative_error_str))
                
            row_img = plotting.plot_epi(img, bg_img=None,black_bg=True, display_mode='xz', figure= fig, axes = ax, cmap='jet', cut_coords=cut)
            
            if i == 2:
                x_hat_title = "RSE=" + str(relative_error_str)
                row_img.title(x_hat_title, size=6)
            if i == 1:
                row_img.add_contours(img, levels=[0.1, 0.5, 0.7, 0.9], filled=True, colors='b')
                coords = [(x0, y0, z0)]
                row_img.add_markers(coords, marker_color='b', marker_size=25) 
                x_miss_title = "Timepoint=" + str(j)
                row_img.title(x_miss_title, size=6) 
                
    
    if folder and radius:
        fig_id =  str(folder) + "/" + "_"+ str(x_r) + "_"+ str(y_r) + "_"+  str(z_r)+"_timeframes_" + str(n)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
        
    mrd.save_fig_abs_path(fig_id)
    
def draw_n_frames_syntetic_img(x_true_img_list, x_hat_img_list, x_miss_img_list, n, plot_title, relative_error_list, observed_ratio_list, folder=None, file_name = None):
    
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = n
    
    fg_color = 'white'
    bg_color = 'black'


                
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=11)
    
    miss_title = None
    
    
    if plot_title and miss_title:
        fig_title = plot_title + "\n" + miss_title
        fig.suptitle(fig_title, color=fg_color, fontweight='normal', fontsize=11)
    
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
          
    
    grid_size = (grid_rows, grid_cols)
    org_img_ax = fig.add_subplot(grid[0, 0])
    org_img_ax.set_title("Original Image")
    org_img_ax.imshow(x_true_img_list[0])
    
   
    for i in range(grid_size[0]+1):
        for j in range(grid_size[1]):
            ax = fig.add_subplot(grid[i, j])
            if i == 0:
                
                missing_ratio = (1.0 - observed_ratio_list[j])
                missing_ratio_str = mrd.formatted_percentage(missing_ratio, 2)  
                row_img = x_miss_img_list[j]   
                ax.set_title("MR=" + str(missing_ratio_str))  
            else:
                row_img = x_hat_img_list[j]
                
                relative_error_str = "{0:.5f}".format(relative_error_list[j]) 
                ax.set_title("RSE=" + str(relative_error_str))
                       
            row = ax.imshow(row_img)
            
                
    
    if folder and file:
        fig_id =  str(folder) + "/" + str(file_name) + "_missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
        
    mrd.save_fig_abs_path(fig_id)
    
def draw_4d_10():
    folder_path = "/work/pl/sch/analysis/scripts/csv_data/solution/structural/d4/10"
    
    x_true_path = os.path.join(folder_path,"x_true_img_10.nii")
    x_hat_path = os.path.join(folder_path,"x_hat_img_10.nii")
    x_miss_path = os.path.join(folder_path,"x_miss_img_10.nii")
    
    x_true_img = mc.read_image_abs_path(x_true_path)
    x_hat_img = mc.read_image_abs_path(x_hat_path)
    x_miss_img = mc.read_image_abs_path(x_miss_path)
    
    title = 'fMRI 4D Scan Completion. Pattern - Structural Values. \n # Frames Corrupted = 10 of 144. Miss Ratio: 7%'
    
    x0, y0, z0 = (0 ,-18 , 17)
    cut_coords = [x0, y0, z0]
    
    x_r, y_r, z_r = (20, 17, 15)
    radius = [x_r, y_r, z_r]
    
    folder = "/work/pl/sch/analysis/scripts/figures/d4/structural/10"
       
    draw_n_frames(x_true_img, x_hat_img, x_miss_img, 2, title, 0.00013, 0.93, coord=cut_coords, folder=folder, radius = radius)
    
def draw_4d_50():
    folder_path = "/work/pl/sch/analysis/scripts/csv_data/solution/structural/d4/50"
    
    x_true_path = os.path.join(folder_path,"x_true_img_50.nii")
    x_hat_path = os.path.join(folder_path,"x_hat_img_50.nii")
    x_miss_path = os.path.join(folder_path,"x_miss_img_50.nii")
    
    x_true_img = mc.read_image_abs_path(x_true_path)
    x_hat_img = mc.read_image_abs_path(x_hat_path)
    x_miss_img = mc.read_image_abs_path(x_miss_path)
    
    title = 'fMRI 4D Scan Completion. Pattern - Structural Values. \n # Frames Corrupted = 50 of 144. Miss Ratio: 34.7%'
    
    x0, y0, z0 = (0 ,-18 , 17)
    cut_coords = [x0, y0, z0]
    
    x_r, y_r, z_r = (20, 17, 15)
    radius = [x_r, y_r, z_r]
    
    folder = "/work/pl/sch/analysis/scripts/figures/d4/structural/50"
       
    draw_n_frames(x_true_img, x_hat_img, x_miss_img, 2, title, 0.00279, 0.67, coord=cut_coords, folder=folder, radius = radius)
    
if __name__ == "__main__":
    
    draw_4d_10()
    draw_4d_50()
    
    
    
    