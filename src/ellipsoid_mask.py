import sys
import numpy as np
import pandas as pd
from collections import OrderedDict
import mri_draw_utils as mrd
import math
from nilearn.masking import compute_epi_mask
import metric_util as mt


class EllipsoidMask(object):
    
    def __init__(self, x0, y0, z0, x_r, y_r, z_r, path):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x_r = x_r
        self.y_r = y_r
        self.z_r = z_r
        self.mask_path = path
         
        
    def volume(self):
        result = (4.0/3.0)*math.pi*(self.x_r *self.y_r*self.z_r)
        return result
    
    def compute_observed_ratio(self, target_img):
        ellipse_img = mt.read_image_abs_path(self.path)
        #target_img = mt.read_image_abs_path(target_img_path)
        target_img_data = np.array(target_img.get_data())
        target_data_count = np.count_nonzero(target_img_data)
        
        ellipse_data = np.array(ellipse_img.get_data())
        ellipse_indices_count = np.count_nonzero(ellipse_data)
        
        result = float(1.0*ellipse_indices_count/target_data_count)
        self.obseved_ratio = result
        print("Ellipsoid :" + "; x_r =" + str(self.x_r) + "; y_r=" + str(self.y_r) + "; z_r = " + str(self.z_r) + "; Volume =" + str(self.volume) + "; Observed Ratio: " + str(result))
        return result
        
    def compute_effective_roi_volume(self, mr, time_point_count):
        self.ts_count = time_point_count
        self.mr = mr
        self.el_volume = self.volume()
        print("Ellipsoid Volume:" + str(self.el_volume))
        self.effective_roi_volume = int(self.volume()*time_point_count)
        return self.effective_roi_volume
        
        
    
