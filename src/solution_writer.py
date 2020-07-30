import sys
import numpy
import pandas as pd
from collections import OrderedDict
import mri_draw_utils as mrd
import solution as sol
from datetime import datetime
import math

class SolutionWriter(object):
    
    def __init__(self, folder, name = None):
        self.solutions = []
        self.name = name
        self.folder = folder
        
        
    def write_all(self):
        
        
        indices = []
        cost_indices = []
        counter = 0
        
        output_sum = OrderedDict()
        output_cost = OrderedDict()
        
        solution_id_arr = []
        cost_sol_name_arr = []
        d_arr = []
        miss_ratio_arr = []
        pattern_arr = []
        rel_error_arr = []
        it_count_arr = []
        n_arr = []
        it_rel_error_arr = []   
        tsc_arr = []
        nrmse_arr = [] 
        
        # write summary file
        for item in self.solutions:
            item.write_summary()
            
            solution_id_arr.append(item.solution_id)
            cost_sol_name_arr.append(item.get_solution_name())
            d_arr.append(item.d[0])
            miss_ratio_arr.append(item.missing_ratio)
            pattern_arr.append(item.pattern)
            rel_error_arr.append(item.rel_error)
            it_count_arr.append(item.it_count)
            tsc_arr.append(item.tsc)
            nrmse_arr.append(item.nrmse)
    
            indices.append(counter)
            counter = counter + 1
        
        output_sum['solution_id'] = solution_id_arr
        output_sum['summary_name'] = cost_sol_name_arr
        output_sum['d'] = d_arr
        output_sum['miss_ratio'] = miss_ratio_arr
        output_sum['pattern'] = pattern_arr
        output_sum['rel_error'] = rel_error_arr
        output_sum['it_count'] = it_count_arr
        
        output_sum['tsc_score'] = tsc_arr
        output_sum['nrmse'] = nrmse_arr
        
        
        output_df = pd.DataFrame(output_sum, index=indices)
            
        fig_id = str(self.get_solution_name())    
        mrd.save_csv_by_path(output_df,self.folder,fig_id)   
            
        global_iteration_counter = 0
        
        solution_id_arr = []
        cost_sol_name_arr = []
        d_arr = []
        miss_ratio_arr = []
        pattern_arr = []
        rel_error_arr = []
        it_count_arr = []
        n_arr = []
        it_rel_error_arr = []  
        it_test_error_arr = []  
        tsc_arr_s = []
        nrmse_arr_s = [] 
        
        # write summary file
        for item in self.solutions:
            current_solution = item
            current_solution.write_cost()
            
            sol_it_counter = 0
            for cost_item in current_solution.cost:
                
                solution_id_arr.append(item.solution_id)
                cost_sol_name_arr.append(item.get_solution_execution_name())
                d_arr.append(item.d[0])
                miss_ratio_arr.append(item.missing_ratio)
                pattern_arr.append(item.pattern)
                rel_error_arr.append(current_solution.rel_error)
                it_count_arr.append(item.it_count)
                n_arr.append(sol_it_counter)
                it_rel_error_arr.append(cost_item[0])
                it_test_error_arr.append(current_solution.test_error[sol_it_counter][0])
                
                tsc_arr_s.append(current_solution.tsc)
                nrmse_arr_s.append(current_solution.nrmse)
                
                cost_indices.append(global_iteration_counter)
                global_iteration_counter = global_iteration_counter + 1
                sol_it_counter = sol_it_counter + 1
        
            output_cost['solution_id'] = solution_id_arr
            output_cost['cost_sol_name'] = cost_sol_name_arr
            output_cost['d'] = d_arr
            output_cost['miss_ratio'] = miss_ratio_arr
            output_cost['pattern'] = pattern_arr
            output_cost['rel_error'] = rel_error_arr
            output_cost['it_count'] = it_count_arr
            output_cost['n'] = n_arr
            output_cost['it_rel_error'] = it_rel_error_arr   
            output_cost['it_test_error'] = it_test_error_arr 
            output_cost['tsc_score'] = tsc_arr_s
            output_cost['nrmse'] = nrmse_arr_s
                            
        output_cost_df = pd.DataFrame(output_cost, index=cost_indices)   
        fig_id = str(self.get_solution_execution_name())
        
        mrd.save_csv_by_path(output_cost_df,self.folder,fig_id)    
            
            
    def get_solution_name(self):
        name =  str("agg_summary_")+ str(self.get_ts())
        if self.name:
            name =  str("agg_summary_")+ str(self.get_ts()) + "_" + str(self.name)
        else:
            name =  str("agg_summary_")+ str(self.get_ts())
        return name
    
    def get_solution_execution_name(self):
        if self.name:
            name = str("agg_cost_")+ str(self.get_ts()) + "_" + str(self.name)
        else:
            name = str("agg_cost_")+ str(self.get_ts())
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