import animate_mri as mri_vis
import data_util as du
import metric_util as mt
import configparser
from os import path
import logging
import metadata as mdt
from mayavi import mlab
import  moviepy.editor as mpy

subject_scan_path = du.get_full_path_subject1()
print "Subject Path: " + str(subject_scan_path)
x_true_path = '/work/rs/test/x_true_img_50.nii'
x_true_org = mt.read_image_abs_path(subject_scan_path)

meta = mdt.Metadata('random', 4)
root_dir = "/work/pl/sch/analysis/my_work/medical-image-analyis/doc/figures/results/animation"
#solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder = meta.init_meta_data(root_dir)
solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration  = meta.init_meta_data(root_dir)

print 'Movies dir: ' + movies_folder
mri_animator =  mri_vis.MRIAnimator(movies_folder, 'Original fMRI brain volume', x_true_org, x_true_org)
#mri_animator.animate_volume(x_true_org, 0)
   
#mri_animator.save_anim_to_file(x_true_org, 0)

x_miss_img_path = '/work/str/4D/mr/15/size1/scans/final/mr/15/15/x_miss_img_15.nii'
x_miss_img = mt.read_image_abs_path(x_miss_img_path)
mri_animator =  mri_vis.MRIAnimator(movies_folder, '', x_miss_img, x_miss_img, observed_ratio=0.6)

#/work/rs/x_miss_img_60

#mri_animator.animate_volume(x_miss_img, 19)

mri_animator.save_anim_to_file(x_miss_img, 19)


x_hat_img_path = '/work/rs/x_hat_img_60.nii'
x_hat_img = mt.read_image_abs_path(x_hat_img_path)
#mri_animator =  mri_vis.MRIAnimator(movies_folder, 'Recovered brain volume. RSE=1.28E-3', x_hat_img, x_hat_img, observed_ratio=0.4)

#mri_animator.save_anim_to_file(x_hat_img, 0)