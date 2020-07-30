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


import numpy

from mayavi import mlab
from mayavi.core.engine import Engine
from mayavi.core.off_screen_engine import OffScreenEngine
from mayavi.tools.figure import savefig, screenshot

x, y, z = numpy.mgrid[1:10, 1:10, 1:10]
u, v, w = numpy.mgrid[1:10, 1:10, 1:10]
s = numpy.sqrt(u**2 + v**2)
mlab.quiver3d(x, y, z, u, v, w, scalars=s)
mlab.show()
