import nilearn
from nilearn import image
import nibabel as nib
import copy
from nilearn import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from nilearn.image.resampling import coord_transform, get_bounds, get_mask_bounds
from nilearn.masking import compute_background_mask
import metric_util as mu
import mri_draw_utils as mrd

def get_xyz(i, j, k, epi_img):
    M = epi_img.affine[:3, :3]
    abc = epi_img.affine[:3, 3]
    return M.dot([i, j, k]) + abc

def xyz_coord(img):
    affine = img.affine.copy()
    data_coords = list(np.ndindex(img.shape[:3]))
    data_coords = np.asarray(list(zip(*data_coords)))
    data_coords = coord_transform(data_coords[0], data_coords[1],
                                  data_coords[2], affine)
    data_coords = np.asarray(data_coords).T
    return data_coords

def reconstruct_image_affine(img_ref, x_hat):
    img_ref
    result = nib.Nifti1Image(x_hat, img_ref.affine)
    return result

def get_box_coord(img):
    box_coordinates = get_mask_bounds(img)
    x_min = box_coordinates[0]
    x_max = box_coordinates[1]
    y_min = box_coordinates[2]
    y_max = box_coordinates[3]
    z_min = box_coordinates[4]
    z_max = box_coordinates[5]
    return x_min, x_max, y_min, y_max, z_min, z_max

def ellipsoid_masker(x_r, y_r, z_r, x0, y0, z0, img):
    # compute background mask
    brain_mask = compute_background_mask(img)
    # build a mesh grid as per original image
    x_min, x_max, y_min, y_max, z_min, z_max = get_box_coord(img)
    print "Box Coordinates: " + "; x_min: " + str(x_min) + "; x_max: "  + str(x_max) + "; y_min: " + str(y_min) +  "; y_max: " + str(y_max) + "; z_min " + str(z_min) + "; z_max: "  + str(z_max)
    x_spacing = abs(img.affine[0,0])
    y_spacing = abs(img.affine[1,1])
    z_spacing = abs(img.affine[2,2])
    print "X-spacing: " +str(x_spacing) + "; Y-spacing: " + str(y_spacing) + "; Z-spacing: " + str(z_spacing)
    
     # build a mesh grid as per original image
    x, y, z = np.mgrid[x_min:x_max+1:x_spacing, y_min:y_max+1:y_spacing, z_min:z_max+1:z_spacing]
    
    # analytical ellipse equation
    xx, yy, zz = np.nonzero((((x - x0) / float(x_r)) ** 2 +
               ((y - y0) / float(y_r)) ** 2 +
               ((z - z0) / float(z_r)) ** 2) <= 1)
    # create array with the same shape as brain mask to mask entries of interest
    activation_mask = np.zeros(img.get_data().shape)
    activation_mask[xx, yy, zz] = 1
    activation_img = nib.Nifti1Image(activation_mask, affine=img.affine)
    return activation_img    

def apply_ellipse_mask(mask_img, img):
    data = copy.deepcopy(img.get_data())
    data[mask_img.get_data() > 0] = 0
    masked_image = reconstruct_image_affine(img, data)
    return masked_image

def create_image_with_ellipse_mask(x_r, y_r, z_r, x0, y0, z0, img):
    mask = ellipsoid_masker(x_r, y_r, z_r, x0, y0, z0, img)
    image_masked = apply_ellipse_mask(mask,img)
    return image_masked  
    

def read_image(folder, path):
    img = nib.load(folder + "/" + path)
    return img

def get_data(img):
    data = img.get_data()
    return data

def create_ellipsoid_mask(x0, y0, z0, x_r, y_r, z_r, target_img, mask_path):
    image_masked_by_ellipsoid = create_image_with_ellipse_mask(x_r, y_r, z_r, x0, y0, z0, target_img)
    print "Saving Ellpse Mask @: " + str(mask_path)
    nib.save(image_masked_by_ellipsoid,mask_path)
    mask_path = mask_path + ".nii"
    #plot_ellipse_mask(mask_path,x0, y0, z0, x_r, y_r, z_r)
    return image_masked_by_ellipsoid

def plot_ellipse_mask(mask_path, x0, y0, z0, x_r, y_r, z_r):
    img = mu.read_image_abs_path(mask_path)
    title = "3D Ellipsoid" + " Center: " + str(x0) + ", "+ str(y0) + ", "+  str(z0)+ "; Radius: " + str(x_r) + ", "+ str(y_r) + ", "+  str(z_r)
    
    fg_color = 'white'
    bg_color = 'black'
    
    plot_title = "Structural Missing Pattern in fMRI Scan. \n Missing Voxels enclosed by ellipsoid." + "\n" + str(title)
    
    fig = plt.figure()
    fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=9)
    
                   
    cut_coords = [x0, y0, z0]           
    display = plotting.plot_epi(img, bg_img=None,black_bg=True, cmap='jet', cut_coords=cut_coords)  
    display.title(plot_title, size=6)
    display.add_contours(img, levels=[0.1, 0.5, 0.7, 0.9], colors='r')
    coords = [(x0, y0, z0)]
    display.add_markers(coords, marker_color='y', marker_size=5)
    
    fig_id = os.path.splitext(mask_path)[0]
    mrd.save_fig(fig_id, tight_layout=False)