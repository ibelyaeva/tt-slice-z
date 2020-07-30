import texfig
import matplotlib.pyplot as plt
import numpy as np
import os
import configparser
from os import path
import pandas as pd
import matplotlib.gridspec as gridspec
import io_util as iot
import locale
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from cycler import cycler
import matplotlib.path as mpath
import mri_draw_utils as mrd
import math_format as mf
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterSciNotation
import io_util
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D

dir = '/work/data/csv'

def save_subset_by_tensor_dim(d,mrs):
    data_path = "/work/pl/sch/analysis/my_work/analysis/results/structural/multirun/results/avg_solution_cost_combined_all.csv"
    data = iot.read_data_structural(data_path)
    subset = data.loc[data['tensor_dim'] ==d]
    subset1 = subset.loc[subset['mr'].isin(mrs)]
    solution_csv_path = dir
    solution_id_csv = 'subset_tensor_dim_' + str(d)
    solution_id_crosstab_csv = 'subset_tensor_dim_crosstab_' + str(d)
    
    col_names_csv = ['tensor_dim', 'ts_count', 'roi_volume_label', 'el_volume', 'roi_volume', 'roi_volume_label', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc', 
                     'mr5_tcs', 'mr10_tcs', 'mr15_tcs', 'mr20_tcs', 'mr25_tcs', 'mr25_tcs', 'mr30_tcs', 'mr5_tcs_z', 'mr10_tcs_z', 'mr15_tcs_z', 'mr20_tcs_z', 'mr25_tcs_z', 'mr25_tcs_z', 'mr30_tcs_z'
                     ]
    
    csv_result = pd.DataFrame(col_names_csv)
    csv_result5 = pd.DataFrame(col_names_csv)
    csv_result10 = pd.DataFrame(col_names_csv)
    csv_result15 = pd.DataFrame(col_names_csv) 
    csv_result20 = pd.DataFrame(col_names_csv)
    
    mr5_tcs_subset = subset1.loc[subset['mr'] == 5]
    mr10_tcs_subset = subset1.loc[subset['mr'] == 10]
    mr15_tcs_subset = subset1.loc[subset['mr'] == 15]
    mr20_tcs_subset = subset1.loc[subset['mr'] == 20]
    #mr25_val = subset1.loc[subset['mr'] == 10]
    
    
    csv_result5['tensor_dim'] = pd.Series(mr5_tcs_subset['tensor_dim'])
    csv_result5['roi_volume_label'] = pd.Series(mr5_tcs_subset['roi_volume_label'])
    
    csv_result5['ts_count'] = pd.Series(mr5_tcs_subset['ts_count'])

    csv_result5['el_volume'] = pd.Series(mr5_tcs_subset['el_volume'])
    
    csv_result5['roi_volume'] = pd.Series(mr5_tcs_subset['roi_volume'])  
    csv_result5['roi_volume_label'] = pd.Series(mr5_tcs_subset['roi_volume_label'])  
    csv_result5['missing_tp_rate'] = pd.Series(mr5_tcs_subset['mr'])  
    csv_result5['spatial_mr_rate'] = pd.Series(mr5_tcs_subset['spatial_mr_rate'])  
    csv_result5['spatial_mr_rate_perc'] = pd.Series(mr5_tcs_subset['spatial_mr_rate_perc'])  
    
    #csv_result10['tensor_dim'] = pd.Series(mr10_tcs_subset['tensor_dim'])
    #csv_result10['roi_volume_label'] = pd.Series(mr10_tcs_subset['roi_volume_label'])
    
    
    csv_result15['tensor_dim'] = pd.Series(mr15_tcs_subset['tensor_dim'])
    csv_result15['roi_volume_label'] = pd.Series(mr15_tcs_subset['roi_volume_label'])
    
    
    csv_result20['tensor_dim'] = pd.Series(mr20_tcs_subset['tensor_dim'])
    csv_result20['roi_volume_label'] = pd.Series(mr20_tcs_subset['roi_volume_label'])
    
    csv_result5['mr5_tcs'] = pd.Series(mr5_tcs_subset['tcs_cost'])
    csv_result10['mr10_tcs'] = pd.Series(mr10_tcs_subset['tcs_cost'])
    csv_result15['mr15_tcs'] = pd.Series(mr15_tcs_subset['tcs_cost'])
    csv_result20['mr20_tcs'] = pd.Series(mr20_tcs_subset['tcs_cost'])
    #csv_result['ts_count'] = mr25_val

    csv_result5['mr5_tcs_z'] = pd.Series(mr5_tcs_subset['tsc_z_cost'])
    csv_result10['mr10_tcs_z'] = pd.Series(mr10_tcs_subset['tsc_z_cost'])
    csv_result15['mr15_tcs_z'] = pd.Series(mr15_tcs_subset['tsc_z_cost'])
    csv_result20['mr20_tcs_z'] = pd.Series(mr20_tcs_subset['tsc_z_cost'])
    
    results = []
    
    results.append(csv_result5)
    results.append(csv_result10)
    results.append(csv_result15)
    results.append(csv_result20)
    
    all_datasets_df = pd.concat(results, join = 'inner', axis = 1)
    
    #csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    
    mrd.save_csv_by_path_adv(subset1, solution_csv_path, solution_id_csv, index = False) 
    mrd.save_csv_by_path_adv(all_datasets_df, solution_csv_path, solution_id_crosstab_csv, index = False) 
    
def save_subset_by_tensor_dim_mr(dims,mr):
    data_path = "/work/pl/sch/analysis/my_work/analysis/results/structural/multirun/results/avg_solution_cost_combined_all.csv"
    data = iot.read_data_structural(data_path)
    subset = data.loc[data['mr'] ==mr]
    solution_csv_path = dir
    solution_id_csv = 'subset_tensor_dim_mr_' + str(mr)
    solution_id_crosstab_csv = 'subset_tensor_dim_crosstab_mr' +"_" + str(mr)
    
    col_names_csv = ['tensor_dim', 'spatial_mr_rate_perc', 'tcs_cost', 'tcs_cost', 'tcs_z_cost']
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    for d in dims:
        subset_dim = subset.loc[subset['tensor_dim'] == d]    
        solution_id_csv = 'subset_tensor_dim_' 
        solution_id_csv = solution_id_csv  + str(d) + '_' + str(mr)
        mrd.save_csv_by_path_adv(subset_dim, solution_csv_path, solution_id_csv, index = False)
        
    
def run():
    d = 4 
    miss_ratio = [5,10,15,20]
    #save_subset_by_tensor_dim(d,miss_ratio)
    
    dims = [2,3, 4]
    mr =10
    d=2
    save_subset_by_tensor_dim_mr(dims,mr)
    #save_subset_by_tensor_dim(d,miss_ratio)
    
if __name__ == "__main__":
    #draw_structural_results()
    run()