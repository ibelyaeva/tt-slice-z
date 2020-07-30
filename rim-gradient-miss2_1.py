
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import t3f
tf.set_random_seed(0)
np.random.seed(0)
get_ipython().magic(u'matplotlib inline')
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
import mri_draw_utils as mrd


# In[2]:


#sess = tf.InteractiveSession()


# In[3]:


def frobenius_norm_tf_squared(x):
    return tf.reduce_sum(x ** 2)


# In[4]:


def frobenius_norm_tf(x):
    return tf.reduce_sum(x ** 2) ** 0.5


# In[5]:


def relative_error1(x_hat,x_true):
    percent_error = frobenius_norm_tf(x_hat - x_true) / frobenius_norm_tf(x_true)
    return percent_error


# In[6]:


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


# In[7]:


subject_scan_path = du.get_full_path_subject1()
print "Subject Path: " + str(subject_scan_path)
x_true_org = mt.read_image_abs_path(subject_scan_path)


# In[ ]:





# In[8]:


observed_ratio = 0.4
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


# In[9]:


ten_ones = np.ones_like(mask_indices)


# In[10]:


norm_sparse_observation = np.linalg.norm(sparse_observation)
print norm_sparse_observation


# In[11]:


mask_indices_tf = t3f.to_tt_tensor(mask_indices.astype('float32'), max_tt_rank=63)


# In[12]:


x_miss_img = mt.reconstruct_image_affine(x_true_org, x_train)


# In[13]:


x_miss = image.index_img(x_miss_img,1)


# In[14]:


x_miss_image = plotting.plot_epi(x_miss, bg_img=None,black_bg=True, cmap='jet', cut_coords=None) 


# In[15]:


images_folder = "/work/scratch/tt1"
mrd.draw_original_vs_reconstructed_rim(image.index_img(x_true_org,1),image.index_img(x_miss_img,1), image.index_img(x_miss_img,1), "Rim Completion",
                                             observed_ratio, observed_ratio, coord=None, folder=images_folder)


# In[16]:


ground_truth_tf = t3f.to_tt_tensor(ground_truth, max_tt_rank=63)


# In[17]:


A = t3f.get_variable('A', initializer=ground_truth_tf, trainable=False)


# In[18]:


ground_truth = tf.get_variable('ground_truth', initializer=ground_truth, trainable=False)
sparsity_mask = tf.get_variable('sparsity_mask', initializer=mask_indices, trainable=False)
sparsity_mask = tf.cast(sparsity_mask,tf.float32)
sparse_observation = ground_truth * sparsity_mask


# In[19]:


#sparsity_mask = t3f.get_variable('sparsity_mask', initializer=mask_indices_tf, trainable=False)
#sparsity_mask = t3f.cast(sparsity_mask, tf.float32)
#sparse_observation_tf = t3f.to_tt_tensor(sparse_observation, max_tt_rank=63)
#sparse_observation_tf3 = t3f.get_variable('sparse_observation_tf', initializer=sparse_observation_tf, trainable=False)


# In[20]:


x_train_tf = t3f.to_tt_tensor(x_init, max_tt_rank=63)


# In[ ]:





# In[21]:


normAOmegavar = tf.get_variable('normAOmega', initializer=norm_sparse_observation, trainable=False)


# In[22]:


X = t3f.get_variable('X', initializer=x_train_tf)


# In[23]:


print X


# In[24]:


def train_it_rel_cost(cost_hist, k):
    res = np.abs(cost_hist[k] - cost_hist[k-1])/np.abs(cost_hist[k])
    return res


# In[ ]:





# In[ ]:





# In[25]:


# Algorithm
#grad_full = (t3f.full(X)*t3f.full(sparse_observation_tf3) - t3f.full(sparse_observation_tf3))
grad_full = sparsity_mask * t3f.full(X) - sparse_observation
grad_t3f = t3f.to_tt_tensor(grad_full, max_tt_rank=63)

loss = 0.5 * t3f.frobenius_norm_squared(grad_t3f)
gradnorm_omega = t3f.frobenius_norm(grad_t3f)/(normAOmegavar)

riemannian_grad = t3f.riemannian.project(grad_t3f, X)
#riemannian_grad_norm = t3f.flat_inner(riemannian_grad, riemannian_grad)



#rel_error1 = relative_error1(t3f.full(X), t3f.full(sparse_observation))




# In[26]:


eps = 1e-3
epsilon_train=1e-5


# In[27]:


print grad_full


# In[28]:


print grad_t3f


# In[29]:


print eps


# In[30]:


alpha = 0.3
train_step = t3f.assign(X, t3f.round(X - alpha * riemannian_grad, max_tt_rank=63))


# In[31]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[32]:


log = []
train_loss_hist = []
#for i in range(1000):
gradnorm_val = sess.run([gradnorm_omega])
print gradnorm_val


# In[33]:




i = 0
while gradnorm_val > eps:   
#for k in range(50):
    i = i + 1
    F_v, gradnorm_val, _ = sess.run([loss, gradnorm_omega, train_step.op])
    
    train_loss_hist.append(gradnorm_val)
    if i > 1:
        diff_train = np.abs(train_loss_hist[i - 1] - train_loss_hist[i-2])/np.abs(train_loss_hist[i-1])
        print (F_v, i, gradnorm_val, diff_train)
        if diff_train <= epsilon_train:
            print "Breaking after " + str(i) + " iterations"
            break
    log.append(F_v)


# In[34]:


print X


# In[35]:


estimated_val = sess.run(t3f.full(X))


# In[36]:


ground_truth_val = sess.run(ground_truth)


# In[37]:


x_reconstr = mt.reconstruct(estimated_val,ground_truth_val, ten_ones, mask_indices)


# In[38]:


def relative_error(x_hat,x_true):
    percent_error = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    return percent_error


# In[39]:


rel_error = relative_error(estimated_val,ground_truth_val)


# In[40]:


print rel_error


# In[41]:


rel_error_rec = relative_error(x_reconstr,ground_truth_val)


# In[42]:


rel_error_rec


# In[43]:


x_miss_img = mt.reconstruct_image_affine(x_true_org, x_train)


# In[44]:


x_miss = image.index_img(x_miss_img,1)


# In[45]:


x_miss_image = plotting.plot_epi(x_miss, bg_img=None,black_bg=True, cmap='jet', cut_coords=None) 


# In[46]:


x_hat_img = mt.reconstruct_image_affine(x_true_org, x_reconstr)
x_hat = image.index_img(x_hat_img,1)
recovered_image = plotting.plot_epi(x_hat, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)


# In[47]:


x_true_org_img = image.index_img(x_true_org,1)
org_image = plotting.plot_epi(x_true_org_img, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)


# In[48]:


images_folder = "/work/scratch/tt1/1"
mrd.draw_original_vs_reconstructed_rim(image.index_img(x_true_org,1),x_hat, image.index_img(x_miss_img,1), "Rim Completion",
                                             observed_ratio, observed_ratio, coord=None, folder=images_folder)


# In[49]:


estimated_val_img = mt.reconstruct_image_affine(x_true_org, estimated_val)


# In[50]:


estimated_val_hat = image.index_img(estimated_val_img,1)


# In[51]:


images_folder = "/work/scratch/tt1/2"
mrd.draw_original_vs_reconstructed_rim(image.index_img(x_true_org,1),estimated_val_hat,image.index_img(x_miss_img,1), "Rim Completion2",
                                             observed_ratio, observed_ratio, coord=None, folder=images_folder)


# In[ ]:




