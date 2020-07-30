
# coding: utf-8

# In[1]:


import numpy as np

# Import PyTorch
import torch
from torch.autograd import Variable

# Import TensorLy
import tensorly as tl
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.random import check_random_state
import nibabel as nib
import copy


# In[2]:


tl.set_backend('pytorch')


# In[3]:


from nilearn.image import math_img
import nibabel as nib
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from nilearn import plotting
from nilearn import image


# In[4]:


random_state = 1234
rng = check_random_state(random_state)
#device = 'cuda:8'


# In[5]:


def get_mask(data, observed_ratio):
    
    if len(data.shape) == 3:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2]) < observed_ratio).astype('int') 
    elif len(data.shape) == 4:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1],data.shape[2], data.shape[3]) < observed_ratio).astype('int') 
    elif len(data.shape) == 2:
        mask_indices = (np.random.rand(data.shape[0],data.shape[1]) < observed_ratio).astype('int') 
    return mask_indices


# In[6]:


def read_image_abs_path(path):
    img = nib.load(path)
    return img

def reconstruct_image_affine(img_ref, x_hat):
    result = nib.Nifti1Image(x_hat, img_ref.affine)
    return result


# In[7]:


subject_scan_path = "/home/ec2-user/analysis/data/subject1/swaAMAYER+cobre01_63001+M87100944+20110309at135133+RSTpre_V01_R01+CM.nii"
print ("Subject Path: " + str(subject_scan_path))


# In[8]:


observed_ratio = 0.95
missing_ratio = 1 - observed_ratio


# In[9]:


x_true_org = read_image_abs_path(subject_scan_path)
x_true_img = np.array(x_true_org.get_data())

mask_img = compute_epi_mask(x_true_org)
mask_img_data = np.array(mask_img.get_data())


# In[10]:


mask_indices = get_mask(x_true_img, observed_ratio)
epi_mask = copy.deepcopy(mask_img_data)
    
mask_indices[epi_mask==0] = 1

norm_ground_truth = np.linalg.norm(x_true_img)
x_true_img = x_true_img * (1./norm_ground_truth)


norm_ground_truth = np.linalg.norm(x_true_img)
x_true_img = x_true_img * (1./norm_ground_truth)
ten_ones = np.ones_like(mask_indices)
x_train = copy.deepcopy(x_true_img)
x_train[mask_indices==0] = 0.0

x_init = copy.deepcopy(x_train)

x_org = reconstruct_image_affine(x_true_org, x_true_img)
x_org_img = image.index_img(x_org,1)
#org_image = plotting.plot_epi(x_org_img, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)


# In[12]:


sparse_observation = x_true_img*mask_indices


# In[13]:


sparse_observation.shape


# In[ ]:


# create random tensor


# In[15]:


shape = [53, 63, 46, 144]
tensor = tl.tensor(rng.random_sample(shape), requires_grad=True)

