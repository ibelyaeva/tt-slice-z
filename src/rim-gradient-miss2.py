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
from t3f import ops


def frobenius_norm_tf(x):
    return tf.reduce_sum(x ** 2) ** 0.5

def relative_error1(x_hat,x_true):
    percent_error = frobenius_norm_tf(x_hat - x_true) / frobenius_norm_tf(x_true)
    return percent_error

def train_it_rel_cost(cost_hist, k):
    res = np.abs(cost_hist[k] - cost_hist[k-1])/np.abs(cost_hist[k])
    return res

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

x_init = copy.deepcopy(x_train)
ground_truth = copy.deepcopy(x_true_img)

sparse_observation = copy.deepcopy(ground_truth)
sparse_observation[mask_indices==0] = 0.0

norm_sparse_observation = np.linalg.norm(sparse_observation)
print norm_sparse_observation

#mask_indices_tf = t3f.to_tt_tensor(mask_indices.astype('float32'), max_tt_rank=63)

x_miss_img = mt.reconstruct_image_affine(x_true_org, x_train)
x_miss = image.index_img(x_miss_img,1)
x_miss_image = plotting.plot_epi(x_miss, bg_img=None,black_bg=True, cmap='jet', cut_coords=None) 
#plotting.show()

#ground_truth_tf = t3f.to_tt_tensor(ground_truth, max_tt_rank=63)
#A = t3f.get_variable('A', initializer=ground_truth_tf, trainable=False)

ground_truth = tf.get_variable('ground_truth', initializer=ground_truth, trainable=False)
sparsity_mask = tf.get_variable('sparsity_mask', initializer=mask_indices, trainable=False)
sparsity_mask = tf.cast(sparsity_mask,tf.float32)
sparse_observation = ground_truth * sparsity_mask

x_train_tf = t3f.to_tt_tensor(x_init, max_tt_rank=63)
normAOmegavar = tf.get_variable('normAOmega', initializer=norm_sparse_observation, trainable=False)
X = t3f.get_variable('X', initializer=x_train_tf)

print X


grad_full = sparsity_mask * t3f.full(X) - sparse_observation
grad_t3f = t3f.to_tt_tensor(grad_full, max_tt_rank=63)

loss = 0.5 * t3f.frobenius_norm_squared(grad_t3f)
gradnorm_omega = tf.sqrt((2*loss))/(normAOmegavar)

riemannian_grad = t3f.riemannian.project(grad_t3f, X)
riemannian_grad_norm = t3f.flat_inner(riemannian_grad, riemannian_grad)

#rel_error1 = relative_error1(t3f.full(X), t3f.full(sparse_observation))

eps = 1e-3
epsilon_train=1e-5

print grad_full
print grad_t3f

#sess.run(normAOmegavar)
#sess.run(loss)
#sess.run(gradnorm_omega)
#sess.run(riemannian_grad_norm)
alpha = 0.3
train_step = t3f.assign(X, t3f.round(X - alpha * riemannian_grad, max_tt_rank=63))

print "Epsilon:"  + str(eps)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log = []
train_loss_hist = []
gradnorm_val = sess.run([gradnorm_omega])
i = 0
while gradnorm_val > eps:    
    i = i + 1
    F_v, gradnorm_val,riemannian_grad_norm_val, _ = sess.run([loss, gradnorm_omega, riemannian_grad_norm, train_step.op])
    
    train_loss_hist.append(gradnorm_val)
    if i > 1:
        diff_train = np.abs(train_loss_hist[i - 1] - train_loss_hist[i-2])/np.abs(train_loss_hist[i-1])
        print (F_v, i, gradnorm_val, riemannian_grad_norm_val, diff_train)
        if diff_train <= epsilon_train:
            print "Breaking after " + str(i) + " iterations"
            break
    log.append(F_v)

estimated_val = sess.run(t3f.full(X))
ten_ones = np.ones_like(mask_indices)

x_reconstr = mt.reconstruct(estimated_val,ground_truth, ten_ones, mask_indices)

def relative_error(x_hat,x_true):
    percent_error = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    return percent_error

rel_error = relative_error(estimated_val,ground_truth)
print "Relative Error:" + str(rel_error)

x_miss_img = mt.reconstruct_image_affine(x_true_org, x_train)
x_miss = image.index_img(x_miss_img,1)
x_miss_image = plotting.plot_epi(x_miss, bg_img=None,black_bg=True, cmap='jet', cut_coords=None) 
x_hat_img = mt.reconstruct_image_affine(x_true_org, x_reconstr)
x_hat = image.index_img(x_hat_img,1)
recovered_image = plotting.plot_epi(x_hat, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)
x_true_org_img = image.index_img(x_true_org,1)
org_image = plotting.plot_epi(x_true_org_img, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)

plotting.show()


