# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# Copyright (c) 2009, Enthought, Inc.
# License: BSD Style.

import numpy as np

from traits.api import HasTraits, Instance, Array, \
    on_trait_change
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
                                MlabSceneModel
                                
import nilearn

from mayavi.modules.text import Text

from medpy.io import load
from medpy.features.intensity import intensities
from nilearn import image
import nibabel as nib
from medpy.io import header
from medpy.io import load, save
import copy
from nilearn import plotting
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from math import ceil
from nilearn.datasets import MNI152_FILE_PATH
from sklearn.model_selection import train_test_split
from nibabel.affines import apply_affine
from nilearn.image.resampling import coord_transform, get_bounds, get_mask_bounds
from skimage.draw import ellipsoid
from nilearn.image import resample_img
from nilearn.masking import compute_background_mask

import numpy as np
from mayavi import mlab


# Produce some nice data.
n_mer, n_long = 6, 11
pi = np.pi
dphi = pi/1000.0
phi = np.arange(0.0, 2*pi + 0.5*dphi, dphi, 'd')
mu = phi*n_mer
x = np.cos(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
y = np.sin(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
z = np.sin(n_long*mu/n_mer)*0.5

# View it.
mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='jet')
mlab.show()