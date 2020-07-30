from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops

import tensorflow as tf
import numpy as np
import t3f
import matplotlib.pyplot as plt
import metric_util as mt
import data_util as du
from t3f import shapes
from nilearn import image
from tensorflow.python.util import nest
import copy
from nilearn import plotting
from t3f import ops as t3fops
from t3f import initializers

tf.set_random_seed(0)
np.random.seed(0)


class RMCGOptimizer(object):
 

    def __init__(self, x=None, x_true=None, mask=None, observed_ratio=None):
        self.x = x
        self.x_true = x_true
        self.mask = mask
        self.observed_ratio = observed_ratio
        self.counter = 0
        
    def init(self):
        self.x_old = self.x
        
        