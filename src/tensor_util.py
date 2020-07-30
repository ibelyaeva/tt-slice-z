import numpy as np
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from nilearn.image import math_img
from nilearn import image
from scipy import stats
import copy
from math import isnan
from statsmodels.robust.scale import mad
from mpmath import mpf, mpc, mp


def get_max_rank(x):
    tensor_shape = x.shape
    
    if len(x.shape) >=4:
        max_rank = np.max(tensor_shape[0:len(x.shape)-1])
    elif len(x.shape) ==2:
        max_rank = np.linalg.matrix_rank(x)
    else:
        max_rank = np.max(tensor_shape)
    return max_rank

def get_tensor_shape(x):
    return x.shape

def get_tensor_shape_as_list(x):
    return list(x.shape)

def get_ten_ones(x):
    ten_ones = np.ones_like(x)
    return ten_ones

def get_mask(data, observed_ratio):
    
    if len(data.shape) == 3:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2]) < observed_ratio).astype('int') 
    elif len(data.shape) == 4:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2], data.shape[3]) < observed_ratio).astype('int') 
    elif len(data.shape) == 2:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1]) < observed_ratio).astype('int') 
    return mask_indices
        
def get_mask4D(data, observed_ratio):
    
    mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2], data.shape[3]) < observed_ratio).astype('int') 
    
    return mask_indices

def get_mask3D(data, observed_ratio):
    
    mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2]) < observed_ratio).astype('int') 
    
    return mask_indices

def get_mask_with_epi(x, x_img, observed_ratio,d):
    mask_img = compute_epi_mask(x_img)
    mask_img_data = np.array(mask_img.get_data())
    
    epi_mask = copy.deepcopy(mask_img_data)
    
    #if d==3:
    #    mask_indices = get_mask3D(x, observed_ratio)
    #else:
    mask_indices = get_mask4D(x, observed_ratio)
    
    mask_indices[epi_mask==0] = 1
   
    epi_mask = copy.deepcopy(mask_img_data)
  
    return mask_indices

def normalize_data(x):
    norm_x = np.linalg.norm(x)
    x_norm = copy.deepcopy(x)
    x_norm = x_norm * (1./norm_x)
    return x_norm, norm_x

def init_random(x):
    x_init = (2*np.random.random_sample(x.shape) - 1).astype('float32')
    return x_init

def create_sparse_observation(x, mask_indices):
    sparse_observation = copy.deepcopy(x)
    sparse_observation[mask_indices==0] = 0.0
    return sparse_observation

def is_nan(x):
    return (x is np.nan or x != x)

def loss_func(x, y):
    return 0.5*(np.linalg.norm(x - y)** 2)

def get_z_scored_image(x):
    z_scored_image = math_img("(yt - np.mean(yt))/np.std(yt)", yt=x)
    return z_scored_image

def get_z_scored_mask(x_img, z_score_cut_off):
    ground_truth_z_score = stats.zscore(get_z_scored_image(x_img).get_data())
    mask_z_score_indices = (abs(ground_truth_z_score) > z_score_cut_off).astype('int') 
    print ("Z-score indices count: " + str(get_mask_z_indices_count(mask_z_score_indices)))
    return mask_z_score_indices
    
def get_z_score_robust_mask(x_img, z_score_cut_off):
    mu = np.median(np.array(x_img.get_data()), axis=3)
    sigma = np.stack([mad(np.array(x_img.get_data()), axis=3)] * np.array(x_img.get_data()).shape[-1], -1)
    idxs = np.where(np.abs(sigma) > 1e-10)
    ground_truth_z_score = np.array(x_img.get_data()) - mu[..., np.newaxis]
    ground_truth_z_score[idxs] /= sigma[idxs]
    mask_z_score_indices = (abs(ground_truth_z_score) > z_score_cut_off).astype('int')
    print ("Z-score indices count: " + str(get_mask_z_indices_count(mask_z_score_indices)))
    return mask_z_score_indices

def get_mask_z_indices_count(mask_z_score):
    mask_z_indices_count = np.count_nonzero(mask_z_score==1)
    return mask_z_indices_count

def tsc_z_score(x_hat,x_true, ten_ones, mask, z_score_mask):
    x_true_ind = np.multiply(x_true, z_score_mask)
    x_hat_ind = np.multiply(x_hat, z_score_mask)
    nomin = np.linalg.norm(np.multiply((ten_ones - mask), (x_true_ind -  x_hat_ind)))
    denom = np.linalg.norm(np.multiply((ten_ones - mask), x_true_ind))
    
    print ("Z Nomin:" + str(mpf(nomin)))
    print ("Z DeNomin:" + str(mpf(denom)))
    
    mp.dps = 50
    
    score = 0
    try:
        score = mpf(nomin)/mpf(denom)
    except Exception as e:
        score = 0
    except Warning as w:
        score = 0
        
    if isNaN(score):
        score = 0        
    return score 

def isNaN(num):
    return num != num

def get_z_score_robust_spatial_mask(x_img, z_score_cut_off):
    mu = np.median(np.array(x_img.get_data()))
    sigma = np.stack([mad(np.array(x_img.get_data()))] * np.array(x_img.get_data()).shape[-1], -1)
    idxs = np.where(np.abs(sigma) > 1e-10)
    ground_truth_z_score = np.array(x_img.get_data()) - mu[..., np.newaxis]
    ground_truth_z_score[idxs] /= sigma[idxs]
    mask_z_score_indices = (abs(ground_truth_z_score) > z_score_cut_off).astype('int')
    print ("Z-score indices count: " + str(get_mask_z_indices_count(mask_z_score_indices)))
    return mask_z_score_indices

#z_score_d_mask = get_z_score_robust_spatial_mask(spike_6_img, 4) 
    

    