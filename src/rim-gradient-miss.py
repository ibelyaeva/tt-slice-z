import tensorflow as tf
import numpy as np
import t3f
tf.set_random_seed(0)
np.random.seed(0)
import matplotlib.pyplot as plt
import metric_util as mt
import data_util as du
from t3f import shapes
from nilearn import image
from skimage.measure import compare_ssim as ssim
from tensorflow.python.util import nest
import copy
from nilearn import plotting
import mri_draw_utils as mrd

def frobenius_norm_tf(x):
    return tf.reduce_sum(x ** 2) ** 0.5


def relative_error1(x_hat,x_true):
    percent_error = frobenius_norm_tf(x_hat - x_true) / frobenius_norm_tf(x_true)
    return percent_error

subject_scan_path = du.get_full_path_subject1()
print "Subject Path: " + str(subject_scan_path)
x_true_org = mt.read_image_abs_path(subject_scan_path)


observed_ratio = 0.9
x_true_img = np.array(x_true_org.get_data())
mask_indices = (np.random.rand(x_true_img.shape[0],x_true_img.shape[1],x_true_img.shape[2], x_true_img.shape[3]) < observed_ratio).astype('int') 
ten_ones = np.ones_like(mask_indices)
x_train = copy.deepcopy(x_true_img)
x_train[mask_indices==0] = 0.0
x_train[mask_indices == 0] = np.mean(x_train[mask_indices == 1])
#x_train[mask_indices == 0] = np.mean(x_train[mask_indices == 1])

x_init = copy.deepcopy(x_train)
#shape = (1, 53, 63, 63, 1)
#x_init_tf = t3f.random_tensor(shape, tt_rank=63)

ground_truth = copy.deepcopy(x_true_img)
sparse_observation = copy.deepcopy(ground_truth)
sparse_observation[mask_indices==0] = 0.0

x_miss_img = mt.reconstruct_image_affine(x_true_org, x_train)
x_miss = image.index_img(x_miss_img,1)

x_miss_image = plotting.plot_epi(x_miss, bg_img=None,black_bg=True, cmap='jet', cut_coords=None) 
ground_truth_tf = t3f.to_tt_tensor(sparse_observation, max_tt_rank=63)

A = t3f.get_variable('A', initializer=ground_truth_tf, trainable=False)
print A


x_train_tf = t3f.to_tt_tensor(x_init, max_tt_rank=63)

X = t3f.get_variable('X', initializer=x_train_tf)

def train_it_rel_cost(cost_hist, k):
    res = np.abs(cost_hist[k] - cost_hist[k-1])/np.abs(cost_hist[k-1])
    return res


# Algorithm
gradF = X - A
riemannian_grad = t3f.riemannian.project(gradF, X)
F = 0.5 * t3f.frobenius_norm_squared(X - A)
alpha = 0.001
gradnorm = t3f.frobenius_norm(X - A,epsilon=1e-06)/(t3f.frobenius_norm(A, epsilon=1e-06))

train_step = t3f.assign(X, t3f.round(X - alpha * riemannian_grad, max_tt_rank=63))
rel_error1 = relative_error1(t3f.full(X), t3f.full(A))


eps = 1.0e-5
epsilon_train=1e-5


eps

def draw(omega, x_true, x_hat, rel_error):
    images_folder = "/work/scratch/tt1"
    ten_ones = np.ones_like(omega)
    x_reconstr1 = mt.reconstruct2(x_hat,ground_truth, omega)
    x_hat_img = mt.reconstruct_image_affine(x_true_org, x_reconstr1)
    x_hat_est = image.index_img(x_hat_img,1)
    
    x_true_org_img = image.index_img(x_true,1)
    x_miss_img = mt.reconstruct_image_affine(x_true_org, x_train)
    x_miss = image.index_img(x_miss_img,1)
    mrd.draw_original_vs_reconstructed_rim(x_true_org_img,x_hat_est, x_miss, "Rim Completion",
                                             rel_error, observed_ratio, coord=None, folder=images_folder)
    

sess = tf.Session()
sess.run(tf.global_variables_initializer())
log = []
train_loss_hist = []
#for i in range(1000):
gradnorm_val = sess.run([gradnorm])
i = 0
while gradnorm_val > eps:    
    i = i + 1
    F_v, rel_error1_v, gradnorm_val, estimated_val, _ = sess.run([F, rel_error1, gradnorm, t3f.full(X), train_step.op ])
    
    train_loss_hist.append(gradnorm_val)
    if i > 1:
        diff_train = np.abs(train_loss_hist[i - 1] - train_loss_hist[i-2])/np.abs(train_loss_hist[i-1])
        print (F_v, i, gradnorm_val, rel_error1_v, diff_train)
        if diff_train <= epsilon_train:
            print "Breaking after " + str(i) + " iterations"
            break
    log.append(F_v)


draw(mask_indices, x_true_org, estimated_val, gradnorm_val)

estimated_val = sess.run(t3f.full(X))
ten_ones = np.ones_like(mask_indices)
x_reconstr3 = mt.mt.reconstruct2(estimated_val,ground_truth,  mask_indices)



def relative_error(x_hat,x_true):
    percent_error = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    return percent_error

rel_error = relative_error(x_reconstr,ground_truth)
print "My Rel Error: " + str(rel_error)




x_miss_img = mt.reconstruct_image_affine(x_true_org, x_train)


x_miss = image.index_img(x_miss_img,1)
images_folder = "/work/scratch/tt"

x_miss_image = plotting.plot_epi(x_miss, bg_img=None,black_bg=True, cmap='jet', cut_coords=None) 
x_hat_img = mt.reconstruct_image_affine(x_true_org, x_reconstr3)
x_hat = image.index_img(x_hat_img,1)
recovered_image = plotting.plot_epi(x_hat, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)
x_true_org_img = image.index_img(x_true_org,1)
org_image = plotting.plot_epi(x_true_org_img, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)
plotting.show()

