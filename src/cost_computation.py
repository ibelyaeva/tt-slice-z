import tensorflow as tf
import numpy as np
import t3f
tf.set_random_seed(0)
np.random.seed(0)
from t3f import shapes
from t3f import ops

from t3f import initializers
from t3f import approximate
import pandas as pd
from scipy import stats
from nilearn.image import math_img

def frobenius_norm_tf_squared(x):
    return tf.reduce_sum(x ** 2)

def compute_loss(x, sparsity_mask, sparse_observation):
    return 0.5*frobenius_norm_tf_squared(sparsity_mask * t3f.full(x) - sparse_observation)

def tsc(x_hat,x_true, ten_ones, mask):
    nomin = np.linalg.norm(np.multiply((ten_ones - mask), (x_true - x_hat)))
    denom = np.linalg.norm(np.multiply((ten_ones - mask), x_true))
    score = nomin/denom
    return score

def innerProduct(x, y):
    result = tf.reduce_sum(tf.multiply(x, y))
    return result

def compute_step_size(n_omega, grad):
    result = -innerProduct(n_omega,grad)/(innerProduct(n_omega, n_omega))
    return result

def frobenius_norm_tf(x):
    return tf.reduce_sum(x ** 2) ** 0.5

def reconstruct_tf(x_hat, sparse_observation, tf_ones, sparsity_mask):
    x_reconstruct = sparse_observation + (tf_ones - sparsity_mask)*x_hat
    return x_reconstruct

def tsc_tf(x_hat, tf_ones, sparsity_mask, sparse_observation, denom_tsc_tf):
    x_rec = reconstruct_tf(x_hat, sparse_observation, tf_ones, sparsity_mask)
    nomin = frobenius_norm_tf((tf_ones - sparsity_mask)*(x_rec - sparse_observation))
    score = nomin/denom_tsc_tf
    return score  