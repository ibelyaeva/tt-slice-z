
# coding: utf-8

# In[1]:


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
from skimage.measure import compare_ssim as ssim
from tensorflow.python.util import nest
import copy



subject_scan_path = du.get_full_path_subject1()
print "Subject Path: " + str(subject_scan_path)
x_true_org = mt.read_image_abs_path(subject_scan_path)

observed_ratio = 0.1
x_true_img = np.array(x_true_org.get_data())
mask_indices = (np.random.rand(x_true_img.shape[0],x_true_img.shape[1],x_true_img.shape[2], x_true_img.shape[3]) < observed_ratio).astype('int') 
ten_ones = np.ones_like(mask_indices)
x_train = copy.deepcopy(x_true_img)
x_train[mask_indices==0] = 0.0
x_train[mask_indices == 0] = np.mean(x_train[mask_indices == 1])
x_init = x_train

shape = (53,63,46,144)
# Fix random seed so the results are comparable between runs.
tf.set_random_seed(0)
# Generate ground truth tensor A. To make sure that it has low TT-rank,
# let's generate a random tt-rank 5 tensor and apply t3f.full to it to convert to actual tensor.
#ground_truth = t3f.full(t3f.random_tensor(shape, tt_rank=5))
ground_truth = x_true_img
# Make a (non trainable) variable out of ground truth. Otherwise, it will be randomly regenerated on each sess.run.
ground_truth = tf.get_variable('ground_truth', initializer=ground_truth, trainable=False)
noise = 1e-2 * tf.get_variable('noise', initializer=tf.random_normal(shape), trainable=False)
noisy_ground_truth = ground_truth
# Observe 25% of the tensor values.
#sparsity_mask = tf.cast(tf.random_uniform(shape) <= 0.60, tf.float32)
sparsity_mask = tf.get_variable('sparsity_mask', initializer=mask_indices, trainable=False)
sparsity_mask = tf.cast(sparsity_mask,tf.float32)
sparse_observation = noisy_ground_truth * sparsity_mask


# ### Initialize the variable and compute the loss



def frobenius_norm_tf(x):
    return tf.reduce_sum(x ** 2) ** 0.5



def relative_error1(x_hat,x_true):
    percent_error = (frobenius_norm_tf(x_hat - x_true))**2 / (frobenius_norm_tf(x_true)**2)
    return percent_error



def tsc(x_hat,x_true, ten_ones, mask):
    nomin = np.linalg.norm(np.multiply((ten_ones - mask), (x_true - x_hat)))
    denom = np.linalg.norm(np.multiply((ten_ones - mask), x_true))
    score = nomin/denom
    return score  


def train_it_rel_cost(cost_hist):
    res = np.abs(cost_hist[i] - cost_hist[i-1])/np.abs(cost_hist[i-1])
    return res


observed_total = tf.reduce_sum(sparsity_mask)
total = np.prod(shape)
ranks_a = np.array([53,63,46,144,1])
tt_with_ranks = t3f.to_tt_tensor(x_true_img, max_tt_rank=144)
ranks = shapes.tt_ranks(tt_with_ranks)
initialization = t3f.random_tensor(shape, tt_rank=10)

x_init_tf = t3f.to_tt_tensor(x_init, max_tt_rank=45)
tt_with_ranks = t3f.to_tt_tensor(x_true_img, max_tt_rank=144)
estimated = t3f.get_variable('estimated', initializer=x_init_tf)
#estimated = t3f.get_variable('estimated', initializer=initialization)
# Loss is MSE between the estimated and ground-truth tensor as computed in the observed cells.
loss = 0.5 * tf.reduce_sum((sparsity_mask * t3f.full(estimated) - sparse_observation)**2)
#ssim_loss = ssim(sparsity_mask * t3f.full(estimated), sparse_observation)
# Test loss is MSE between the estimated tensor and full (and not noisy) ground-truth tensor A.
#msssim_index = MultiScaleSSIM(sparsity_mask * t3f.full(estimated), t3f.full(estimated))
nch = tf.shape(sparse_observation)[-1]
#msssim_index = tf_ms_ssim(t3f.full(estimated), t3f.full(estimated))
#ssim_tf = tf.reduce_mean(structural_similarity(sparsity_mask * t3f.full(estimated), sparse_observation))
test_loss = 0.5 * tf.reduce_sum((t3f.full(estimated) - ground_truth)**2)
rel_error1 = relative_error1(sparsity_mask * t3f.full(estimated), sparse_observation)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta2 = 0.98, epsilon=1e-8)
step = optimizer.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_loss_hist = []
test_loss_hist = []

rel_error_hist = []

error_epsilon = 1e-5
tol = 1e-12
epsilon1=1e-12*2
sim_index_max = 0.999999
for i in range(10000):
    _, tr_loss_v, test_loss_v, rel_error1_v, nch_v, ranks_v, estimated_val, ground_truth_val = sess.run([step, loss, test_loss,rel_error1, nch, ranks, t3f.full(estimated), ground_truth])
    train_loss_hist.append(tr_loss_v)
    test_loss_hist.append(test_loss_v)
    rel_error_hist.append(rel_error1_v)
    
    tsc_score = tsc(estimated_val,ground_truth_val, ten_ones, mask_indices)
    
    #if i % 100 == 0:
    #    ssim_index = ssim(ground_truth_val, estimated_val,
    #              data_range=estimated_val.max() - estimated_val.min())
    #    print "SSIM Index = " + str(ssim_index)
    
    if i > 1:
       diff_train = train_it_rel_cost(rel_error_hist)
       print "Train Cost Diff = " + str(diff_train)
       print "TSC Score = " + str(tsc_score)
       if diff_train<= epsilon1 and (rel_error1_v<= tol or tsc_score<= error_epsilon):
            print("Train Cost Diff = " + str(diff_train))
            break
 
    
    
      
    print(i, tr_loss_v, test_loss_v, rel_error1_v, nch_v, ranks_v)


plt.loglog(train_loss_hist, label='train')
plt.loglog(test_loss_hist, label='test')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss value')
plt.title('SGD completion')
plt.legend()

ground_truth_val = ground_truth.eval(session=sess)

estimated_val = sess.run(t3f.full(estimated))


def relative_error(x_hat,x_true):
    percent_error = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    return percent_error

rel_error = relative_error(estimated_val,ground_truth_val)

rel_error


estimated_val.shape


from nilearn import image


shape = (53,63,46,144)



sparse_observation_val=sparse_observation.eval(session=sess)


x_miss_img = mt.reconstruct_image_affine(x_true_org, sparse_observation_val)


x_miss = image.index_img(x_miss_img,1)


x_hat_img = mt.reconstruct_image_affine(x_true_org, estimated_val)

x_hat = image.index_img(x_hat_img,1)


from nilearn import plotting


x_true_org_img = image.index_img(x_true_org,1)


org_image = plotting.plot_epi(x_true_org_img, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)


recovered_image = plotting.plot_epi(x_hat, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)


x_miss_image = plotting.plot_epi(x_miss, bg_img=None,black_bg=True, cmap='jet', cut_coords=None) 


sparse_observation_val.dtype


ssim_index = ssim(ground_truth_val, estimated_val,
                  data_range=estimated_val.max() - estimated_val.min())


print("SSIM Index = " + str(ssim_index))

type(rel_error1_v)


tsc_score = tsc(estimated_val,ground_truth_val, ten_ones, mask_indices)


tsc_score

plotting.show()




