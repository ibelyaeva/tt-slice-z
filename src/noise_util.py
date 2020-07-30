import numpy as np
import tensorflow as tf
import t3f
tf.set_random_seed(0)
np.random.seed(0)
import matplotlib.pyplot as plt
import metric_util as mt
import data_util as du
from t3f import shapes
from nilearn import image
import nibabel as nib
from math import sqrt
import metric_util
from dipy.sims.voxel import add_noise
from nilearn.masking import compute_epi_mask

def generate_system_noise_roi_mask(img, snr_db, mask):
    snr = sqrt(np.power(10.0, snr_db / 10.0))
    print ("snr: " + str(snr))
    data = np.array(img.get_data())
    signal = data[mask > 0].reshape(-1)
    sigma_n = signal.mean() / snr
    print ("sigma_n: " + str(sigma_n))
    n_1 = np.random.normal(size=data.shape, scale=sigma_n)
    n_2 = np.random.normal(size=data.shape, scale=sigma_n)
    stde_1 = n_1 / sqrt(2.0)
    stde_2 = n_2 / sqrt(2.0)
    im_noise = np.sqrt((data + stde_1)**2 + (stde_2)**2)
    im_noise[mask == 0] = 0
    noise_idxs = np.where(im_noise > 0)
    data[noise_idxs] = im_noise[noise_idxs]
    return data, im_noise

def create_noisy_image(x, snr_db, mask):
    x_noisy, noise_mask = generate_system_noise_roi_mask(x, snr_db, mask)
    x_noisy_img = mt.reconstruct_image_affine(x, x_noisy)
    noise_mask_img = mt.reconstruct_image_affine(x, noise_mask)
    return x_noisy_img, noise_mask_img

def add_gaussian_noise(img, img_data, snr_db):
    snr = sqrt(np.power(10.0, snr_db / 10.0))
    signal_mean = np.array(image.mean_img(img).get_data()).mean()
    noise_img = add_noise(img_data, snr, signal_mean, noise_type='gaussian')
    return  noise_img

def add_richian_noise(img, img_data, snr_db):
    snr = sqrt(np.power(10.0, snr_db / 10.0))
    signal_mean = np.array(image.mean_img(img).get_data()).mean()
    noise_img = add_noise(img_data, snr, signal_mean, noise_type='rician')
    return  noise_img

def add_rayleigh_noise(img, img_data, snr_db):
    snr = sqrt(np.power(10.0, snr_db / 10.0))
    signal_mean = np.array(image.mean_img(img).get_data()).mean()
    noise_img = add_noise(img_data, snr, signal_mean, noise_type='rayleigh')
    return  noise_img

def tSTD(M, x, axis=0):
    stdM = np.std(M, axis=axis)
    stdM[stdM == 0] = x
    stdM[np.isnan(stdM)] = x
    return stdM

def SNR(img, img_data):
    epi_mask_img = compute_epi_mask(img)
    epi_img_data = np.array(epi_mask_img.get_data())
    signal = img_data[epi_img_data > 0]
    signal_mean = signal.mean()
    sigma = np.std(signal)
    n = signal.size   
    return float(signal_mean/ (sigma * sqrt(n / (n - 1))))

def SNRDb(img, img_data):
    epi_mask_img = compute_epi_mask(img)
    epi_img_data = np.array(epi_mask_img.get_data())
    signal = img_data[epi_img_data > 0]
    signal_mean = signal.mean()
    sigma = np.std(signal)
    n = signal.size
    result =  float(signal_mean/ (sigma * np.sqrt(n / (n - 1))))
    return result

def init_random(x):
    init = (2*np.random.random_sample(x.shape) - 1).astype('float32')
    return init
