import sys
import numpy
import pandas as pd
from collections import OrderedDict
import mri_draw_utils as mrd
from datetime import datetime
import math
import nibabel as nib
import os
from scipy import ndimage
import matplotlib.image as mpimg

from scipy import misc

class Solution(object):
    
    def __init__(self, pattern, d, observed_ratio, x_true, x_hat, x_miss, rel_err, cost, 
                 test_error,
                 sol_path, 
                 x0=None, y0=None, z0=None, x_r=None, y_r=None, z_r=None, name=None, tsc=None, nrmse=None, 
                 scan_folder=None):
        self.solution_id = self.get_ts()
        self.name = name
        self.pattern = pattern
        self.d = d,
        self.observed_ratio = observed_ratio
        self.x_true = x_true
        self.x_hat = x_hat
        self.x_miss = x_miss
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x_r = x_r
        self.y_r = y_r
        self.z_r = z_r
        self.rel_error = rel_err
        self.cost = cost
        self.test_error = test_error
        self.it_count = len(self.cost)
        self.solution_path = sol_path
        self.scan_folder = scan_folder
        self.missing_ratio = 1.0 - self.observed_ratio
        self.tsc = tsc
        self.nrmse = nrmse
        
        
    def write_summary(self):
        
        output = OrderedDict()
        output['solution_id'] = self.solution_id
        output['d'] = self.d
        output['missing_ratio'] = self.missing_ratio
        output['pattern'] = self.pattern
        output['rel_error'] = self.rel_error
        output['it_count'] = self.it_count
        
        if self.tsc:
            output['tsc_score'] = self.tsc
        
        if self.nrmse:
            output['nrmse'] = self.nrmse    
    
        output_df = pd.DataFrame(output)
        fig_id = self.get_solution_name()
        mrd.save_csv_by_path(output_df,self.solution_path,fig_id)
        
    def write_cost(self):
        
        indices = []
        counter = 0
        cost_arr = []
        test_cost_arr = []
        
        for item in self.cost:
            indices.append(counter)
            cost_arr.append(item[0])
            counter = counter + 1     
            
        for item in self.test_error:
            test_cost_arr.append(item[0])
            
        output = OrderedDict()
        output['solution_id'] = self.solution_id
        output['d'] = self.d
        output['missing_ratio'] = self.missing_ratio
        output['pattern'] = self.pattern
        output['rel_error'] = self.rel_error
        output['it_count'] = self.it_count
        output['n'] = indices
        output['rel_error'] = cost_arr
        
        output['test_error'] = test_cost_arr
        
        output['tsc_score'] = self.tsc
        output['nrmse'] = self.nrmse
    
        output_df = pd.DataFrame(output, index=indices)
        fig_id = self.get_solution_execution_name()
        mrd.save_csv_by_path(output_df, self.solution_path, fig_id)
        
    def get_solution_name(self):
        if self.name:
            name = "D" + str(self.d[0]) + str("_") + str(self.pattern) + "_" + str("_summary_") + str(self.name) + "_" + str(self.formatted_percentage(self.missing_ratio, 2))
        else:
            name = "D" + str(self.d[0]) + str("_") + str(self.pattern) + "_" + str("_summary_") + str(self.formatted_percentage(self.missing_ratio, 2))
        return name
    
    def get_solution_execution_name(self):
        if self.name:
            name = "D" + str(self.d[0]) + str("_") + str(self.pattern) + "_" + str("_cost_") + str(self.name) + "_" + str(self.formatted_percentage(self.missing_ratio, 2))
        else:
            name = "D" + str(self.d[0]) + str("_") + str(self.pattern) + "_" + str("_cost_") + str(self.formatted_percentage(self.missing_ratio, 2))
        return name
    
    def get_ts(self):
        current_date = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        return current_date

    def floored_percentage(self, val, digits):
        val *= 10 ** (digits + 2)
        return '{1:.{0}f}%'.format(digits, math.floor(val) / 10 ** digits)

    def formatted_percentage(self, value, digits):
        format_str = "{:." + str(digits) + "%}"
        return format_str.format(value)
    
    def save_solution_scans(self): 
        
        suffix = int(round((self.missing_ratio)*100.0, 0))
        print "Missing Ratio: " + str(self.missing_ratio)
        x_true_path = os.path.join(self.scan_folder,"x_true_img_" + str(suffix))
        x_hat_path = os.path.join(self.scan_folder,"x_hat_img_" + str(suffix))
        x_miss_path = os.path.join(self.scan_folder,"x_miss_img_" + str(suffix))
        
        print("x_true_path:" + str(x_true_path))
        nib.save(self.x_true, x_true_path)
        nib.save(self.x_hat, x_hat_path)
        nib.save(self.x_miss, x_miss_path)
        
    def save_solution_scans_2D(self): 
        
        suffix = int(round((self.missing_ratio)*100.0, 0))
        print "Missing Ratio: " + str(self.missing_ratio)
        x_true_path = os.path.join(self.scan_folder,"x_true_img_" + str(suffix))
        x_hat_path = os.path.join(self.scan_folder,"x_hat_img_" + str(suffix))
        x_miss_path = os.path.join(self.scan_folder,"x_miss_img_" + str(suffix))
        
        print("x_true_path:" + str(x_true_path))
        mpimg.imsave(str(x_true_path)+".png", self.x_true)
        mpimg.imsave(str(x_hat_path)+".png", self.x_hat)
        mpimg.imsave(str(x_miss_path)+".png", self.x_miss)
        
               
    def save_solution_structural_scans(self, folder_path, missing_frame_count): 
        
        suffix = int(missing_frame_count)
        x_true_path = os.path.join(folder_path,"x_true_img_" + str(suffix))
        x_hat_path = os.path.join(folder_path,"x_hat_img_" + str(suffix))
        x_miss_path = os.path.join(folder_path,"x_miss_img_" + str(suffix))
        
        print("x_true_path:" + str(x_true_path))
        nib.save(self.x_true, x_true_path)
        nib.save(self.x_hat, x_hat_path)
        nib.save(self.x_miss, x_miss_path)
        
        
