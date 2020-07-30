
# coding: utf-8

# In[248]:


from mayavi import mlab
from tvtk.tools import visual
from numpy import array


# In[249]:


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
import mri_draw_utils as mrd
from t3f import initializers
from t3f import approximate
from scipy import optimize 
from scipy import special
from nilearn.image.resampling import coord_transform
from scipy import stats
from mayavi.mlab import *


# In[250]:


subject_scan_path = du.get_full_path_subject1()
print "Subject Path: " + str(subject_scan_path)
x_true_org = mt.read_image_abs_path(subject_scan_path)


# In[251]:


data = np.array(image.index_img(x_true_org, 0).get_data())


# In[252]:


x, y, z = np.mgrid[-78:78+1:3, -111:75+1:3, -51:84+1:3]


# In[253]:


def get_i_j_k(x1, y1, z1):
    result = np.round(coord_transform(x1, y1, z1, np.linalg.inv(x_true_org.affine)))
    result = result.astype(int)
    result = (result[0], result[1], result[2])
    result = np.asarray(result).T
    return result[0], result[1],result [2]


# In[254]:


def spatial_array(a, data, target):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                #print i,j,k
                c1, c2, c3 = get_i_j_k(i,j,k)
                target[i,j,k] = data[c1,c2,c3]
    return target
    
    


# In[255]:


target_array = np.zeros_like(data)


# In[256]:


def get_slice(index, data):
    if index == 'x':
        res = data[0:data.shape[0],:,:]
    elif index == 'y':
        res = data[:,0:data.shape[1],:]
    else:
        res = data[:,:,0:data.shape[2]]
    return res
    


# In[257]:


x1 = np.linspace(0,data.shape[0],data.shape[0], endpoint=True, dtype=int)


# In[258]:


y1 = np.linspace(0,data.shape[1],data.shape[1], endpoint=True, dtype=int)


# In[259]:


z1 = np.linspace(0,data.shape[1],data.shape[2], endpoint=True, dtype=int)


# In[260]:


xv, yv, zv = np.meshgrid(x1, y1, z1, indexing = 'ij', sparse=False)
x2v, y2v, z2v = np.meshgrid(x1, y1, z1, indexing = 'ij', sparse=False)


# In[261]:


dfdx, dfdy, dfdz = np.gradient(data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[262]:


f = mlab.figure(12, fgcolor=(.0, .0, .0), bgcolor=(1.0, 1.0, 1.0))


# In[ ]:





# In[263]:


countour_sf = mlab.contour3d(xv, yv, zv, data, contours=7, opacity=0.5, colormap='hsv')


# In[264]:


vectors = mlab.quiver3d(x2v, y2v, z2v, dfdx, dfdy, dfdz, mode='arrow',scale_mode='vector', mask_points=8, opacity=0.8, colormap='jet')
   


# In[265]:


mlab.outline(countour_sf, color=(0.7, .7, .7))


# In[266]:


#mlab.pipeline.scalar_cut_plane(countour_sf, opacity = 0.8)


# In[267]:


image_plane = mlab.pipeline.image_plane_widget(countour_sf,
                            plane_orientation='z_axes',
                            slice_index=38,transparent=True, opacity = 0.8
                        )


# In[ ]:





# In[ ]:





# In[268]:



image_source = image_plane.mlab_source


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[269]:


@mlab.show
@mlab.animate(delay=250, ui=True)
def anim():
   
    while 1:
         for i in range(data.shape[2]):
            image_plane.ipw.slice_index = i
            yield


# In[270]:


anim()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



    


# In[ ]:





# In[ ]:



    


# In[ ]:



    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[271]:


#mlab.show()


# In[ ]:





# In[ ]:





# In[ ]:




