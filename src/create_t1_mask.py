import nibabel as nb
import numpy as np
from nilearn import plotting, image
import os
import metric_util as mt
import data_util as du
from scipy import stats

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

mask_gm_mask = mask_gm > 0.5
mask_gm[mask_gm <0.5] = 0.0

subject_img_data = np.array(subject_img.get_data())

subject_img_data[mask_gm==0.0] = 0
mask_gm_img = mt.reconstruct_image_affine(subject_img, mask_gm)

subject_data_gm = stats.zscore(subject_img_data, 2)
subject_data_gm_img= mt.reconstruct_image_affine(subject_img, subject_img_data)

subject_data_gm_mean_img = image.mean_img(subject_data_gm_img)
subject_data_gm_mean = np.array(subject_data_gm_mean_img.get_data())

co = np.corrcoef(X_z[ind,:],subject_data_gm_mean)

print ("I am here")
#nib.save(mask_gm_img, mask_gm_path)
#nib.save(subject_data_gm_mask_img, subject_img_data_gm_path)