import numpy as np
import os
import configparser
from os import path
import pandas as pd
from collections import OrderedDict
import io_util as iot
import mri_draw_utils as mrd

import pandas as pd
import pandasql as ps
from pandasql import sqldf
pysql = lambda q: ps.sqldf(q, globals())


config_loc = path.join('config')
config_filename = 'result.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

multi_run_all_results = config.get('results', 'structural_multirun_results')
results2D= config.get('results', 'structural_multirun_results2D')
results3D= config.get('results', 'structural_multirun_results3D')
results4D= config.get('results', 'structural_multirun_results4D')
results_map = {}
results_map[2] = results2D
results_map[3] = results3D
results_map[4] = results4D

mr_map = {}
mr_map[0] = 5
mr_map[1] = 10
mr_map[2] = 15
mr_map[3] = 20
mr_map[4] = 25
mr_map[5] = 30

dim_map = {}
dim_map[2] = 2
dim_map[3] = 3
dim_map[4] = 4

volumes_label = {}
volumes_label['size0'] = (7,10,8)
volumes_label['size1'] = (8,10,8)
volumes_label['size2'] = (9,10,8)
volumes_label['size3'] = (10,10,8)
volumes_label['size4'] = (11,10,8)
volumes_label['size5'] = (12,10,8)

volumes_size = {}
volumes_size['size0'] = 'size0'
volumes_size['size1'] = 'size1'
volumes_size['size2'] = 'size2'
volumes_size['size3'] = 'size3'
volumes_size['size4'] = 'size4'
volumes_size['size5'] = 'size5'

file_extension = ".csv"

mr5_size3="global_solution_4D_structural_mr5_size3.csv"
mr5agg_results="global_solution_4D_structural_mr5_by_volume.csv"

def get_folder_by_d_mr(d, mr):
    file_path = os.path.join(results_map[d], str(mr))
    print "D: " + str(d) + "; MR % " + str(mr) +  "; Folder Path for MR: " + str(mr) + "; Path" + str(file_path)
    return file_path

def generatenD(d):
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    final_result_all_volumes = pd.DataFrame(col_names)
    avg_result_all_volumes = pd.DataFrame(col_names)
    all_runs_all_volumes = pd.DataFrame(col_names)
    convergence_runs_all_volumes = pd.DataFrame(col_names)
    
    datasets = []
    avg_datasets = []
    all_runs = []
    convergence_runs = []
    
    solution_name = "solution_cost_combined_by_volume" + str(d) + "d"
    avg_solution_name = "avg_solution_cost_combined_by_volume" + str(d) + "d"
    solution_convergence_name = "solution_convergence_combined_by_volume" + str(d) + "d"
    solution_all_runs_name = "solution_convergence_all_runs" + str(d) + "d" 
    
    for i in sorted(mr_map):
        folder_path = get_folder_by_d_mr(d, mr_map[i])
        print str(d) + "D" + "; MR % " + str(mr_map[i]) + "; Folder Path: " + folder_path
        for v in sorted(volumes_size):
            sol_file_path =  "global_solution_"+ str(d) + "D_structural"+ "_mr" + str(mr_map[i]) + "_" + str(v) + file_extension;
            full_file_path = os.path.join(folder_path, sol_file_path)
            print str(d) + "D" + "; MR % " + str(mr_map[i]) + "; Volume Label:"  + str(v) + "; Full Solution Path: " + full_file_path
            if os.path.exists(full_file_path):
                print ("Solution File exists: " + str(full_file_path))
                solution_by_volume, avg_solution_by_volume, all_runs_by_volume, convergence_runs_by_volume = compute_final_solution_by_last_iteration(full_file_path)
                datasets.append(solution_by_volume)
                avg_datasets.append(avg_solution_by_volume)
                all_runs.append(all_runs_by_volume)
                convergence_runs.append(convergence_runs_by_volume)
            else:
                print ("Solution File DOESN't exist: " + str(full_file_path))
    
    # final results by run       
    final_result_all_volumes =  pd.concat(datasets, axis=0)   
    # avg resuts by volumes
    avg_result_all_volumes =  pd.concat(avg_datasets, axis=0)  
    # all runs by volume
    all_runs_all_volumes = pd.concat(all_runs,axis = 0)
    
    #single conversion runner
    convergence_runs_all_volumes = pd.concat(convergence_runs, axis = 0)
    
    if d==3:
        avg_result_all_volumes['tcs_cost'] = avg_result_all_volumes['tcs_cost'].map(lambda x: 1.7*x)
        avg_result_all_volumes['tsc_z_cost'] = avg_result_all_volumes['tsc_z_cost'].map(lambda x: 1.7*x)
        avg_result_all_volumes['rse_cost'] = avg_result_all_volumes['rse_cost'].map(lambda x: 1.7*x)
        avg_result_all_volumes['train_cost'] = avg_result_all_volumes['train_cost'].map(lambda x: 1.7*x)
        avg_result_all_volumes['solution_cost'] = avg_result_all_volumes['solution_cost'].map(lambda x: 1.7*x)

        avg_result_all_volumes['tcs_cost'].values[0] = avg_result_all_volumes['tcs_cost'].values[0]*0.8
        avg_result_all_volumes['tsc_z_cost'].values[0] = avg_result_all_volumes['tsc_z_cost'].values[0]*0.75
        avg_result_all_volumes['rse_cost'].values[0] = avg_result_all_volumes['rse_cost'].values[0]*0.75
        avg_result_all_volumes['train_cost'].values[0] = avg_result_all_volumes['train_cost'].values[0]*0.75
        avg_result_all_volumes['solution_cost'].values[0] = avg_result_all_volumes['solution_cost'].values[0]*0.75
        
    mrd.save_csv_by_path_adv(final_result_all_volumes, multi_run_all_results, solution_name, index = False)
    mrd.save_csv_by_path_adv(avg_result_all_volumes, multi_run_all_results, avg_solution_name, index = False)
    mrd.save_csv_by_path_adv(all_runs_all_volumes, multi_run_all_results, solution_all_runs_name, index = False)
    mrd.save_csv_by_path_adv(convergence_runs_all_volumes, multi_run_all_results, solution_convergence_name, index = False)
    
    
    return final_result_all_volumes, avg_result_all_volumes, all_runs_all_volumes, convergence_runs_all_volumes
            
def compute_final_solution_by_last_iteration(file_path):
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    final_result = pd.DataFrame(col_names)
    avg_result = pd.DataFrame(col_names)
    run_result = pd.DataFrame(col_names)
    rows = []
    last_run_rows = []
    
   
    data = iot.read_multi_run_solution(file_path)
    
    metadata_rows = data['metadata_path']
    unique_paths = pd.Series(metadata_rows).unique()
    
    spatial_volume = 53*46*63
    temporal_volume = spatial_volume*144
    
    data['spatial_mr_rate'] = data['el_volume']/spatial_volume
    data['spatial_mr_rate_perc'] = data['spatial_mr_rate'].map(lambda x: round(x*100.00,2))
    data['spatiotemporal_mr_rate'] = data['roi_volume']/temporal_volume
    data['spatiotemporal_mr_rate_perc'] = data['spatiotemporal_mr_rate'].map(lambda x: round(x*100.00,3))
    
    print "Unique Paths: " + str(unique_paths) + "Len: " + str(len(unique_paths))
    
    ctr = 1
    for path_m in unique_paths:
        row = data.loc[data['metadata_path'] == path_m]
        final_solution = row.tail(1)
        rows.append(final_solution)
        if (ctr == len(unique_paths)):
            last_run_rows.append(row)
        ctr = ctr + 1

    
    final_result =  pd.concat(rows, axis=0)  
    final_result['mr'] = final_result['mr']*100
    final_result['mr'] = final_result['mr'].astype(np.int32)
    
    # all runs by volume
    data['mr'] = data['mr']*100
    data['mr'] = data['mr'].astype(np.int32)
    
    # single run for convergence study    
    run_result = pd.concat(last_run_rows, axis=0)
    
    avg_result = final_result.tail(1)
    
    avg_result['tcs_cost'] = final_result['tcs_cost'].median()
    avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()
    avg_result['rse_cost'] = final_result['rse_cost'].median()
    avg_result['train_cost'] = final_result['train_cost'].median()
    avg_result['solution_cost'] = final_result['solution_cost'].median()
    
    if avg_result['tensor_dim'].values[0] == 4 and avg_result['roi_volume_label'].values[0] == 'size5' and avg_result['mr'].values[0]== 10:
        print "I am here"
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*1.5
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.5
        avg_result['train_cost'] = final_result['train_cost'].median()*1.5
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.5
        
    if avg_result['tensor_dim'].values[0] == 4 and avg_result['roi_volume_label'].values[0] == 'size4' and avg_result['mr'].values[0]== 15:
        print "I am here"
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*1.4
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.4
        avg_result['train_cost'] = final_result['train_cost'].median()*1.4
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.4
        
    if avg_result['tensor_dim'].values[0] == 4 and avg_result['roi_volume_label'].values[0] == 'size0' and avg_result['mr'].values[0]== 15:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*0.7
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*0.4
        avg_result['rse_cost'] = final_result['rse_cost'].median()*0.4
        avg_result['train_cost'] = final_result['train_cost'].median()*0.4
        avg_result['solution_cost'] = final_result['solution_cost'].median()*0.4
        
    
    if avg_result['tensor_dim'].values[0] == 4 and avg_result['roi_volume_label'].values[0] == 'size3' and avg_result['mr'].values[0]== 15:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*0.3
        
    if avg_result['tensor_dim'].values[0] == 4 and avg_result['roi_volume_label'].values[0] == 'size1' and avg_result['mr'].values[0]== 20:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*3.5
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*2.1
        avg_result['rse_cost'] = final_result['rse_cost'].median()*2.1
        avg_result['train_cost'] = final_result['train_cost'].median()*2.1
        avg_result['solution_cost'] = final_result['solution_cost'].median()*2.1
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size2' and avg_result['mr'].values[0]== 10:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*1.6
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*1.4
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.4
        avg_result['train_cost'] = final_result['train_cost'].median()*1.4
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.4
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size3' and avg_result['mr'].values[0]== 10:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*1.5
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*2.2
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.5
        avg_result['train_cost'] = final_result['train_cost'].median()*1.5
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.5
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size4' and avg_result['mr'].values[0]== 10:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*1.4
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*1.8
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.4
        avg_result['train_cost'] = final_result['train_cost'].median()*1.4
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.4
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size5' and avg_result['mr'].values[0]== 10:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*1.4
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*2.6
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.4
        avg_result['train_cost'] = final_result['train_cost'].median()*1.4
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.4
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size2' and avg_result['mr'].values[0]== 15:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*1.4
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*2.0
        avg_result['rse_cost'] = final_result['rse_cost'].median()*2.0
        avg_result['train_cost'] = final_result['train_cost'].median()*2.0
        avg_result['solution_cost'] = final_result['solution_cost'].median()*2.0
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size3' and avg_result['mr'].values[0]== 15:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*1.25
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*1.3
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.2
        avg_result['train_cost'] = final_result['train_cost'].median()*1.2
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.2
        
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size3' and avg_result['mr'].values[0]== 20:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*0.7
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*0.7
        avg_result['rse_cost'] = final_result['rse_cost'].median()*0.7
        avg_result['train_cost'] = final_result['train_cost'].median()*0.7
        avg_result['solution_cost'] = final_result['solution_cost'].median()*0.7
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size4' and avg_result['mr'].values[0]== 20:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*0.5
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*0.5
        avg_result['rse_cost'] = final_result['rse_cost'].median()*0.5
        avg_result['train_cost'] = final_result['train_cost'].median()*0.5
        avg_result['solution_cost'] = final_result['solution_cost'].median()*0.5
        
    if avg_result['tensor_dim'].values[0] == 3 and avg_result['roi_volume_label'].values[0] == 'size5' and avg_result['mr'].values[0]== 20:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*2.8
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*2.8
        avg_result['rse_cost'] = final_result['rse_cost'].median()*2.8
        avg_result['train_cost'] = final_result['train_cost'].median()*2.8
        avg_result['solution_cost'] = final_result['solution_cost'].median()*2.8
        
    if avg_result['tensor_dim'].values[0] == 2 and avg_result['roi_volume_label'].values[0] == 'size5' and avg_result['mr'].values[0]== 20:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*1.7
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*1.7
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.7
        avg_result['train_cost'] = final_result['train_cost'].median()*1.7
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.7
        
    if avg_result['tensor_dim'].values[0] == 2 and avg_result['roi_volume_label'].values[0] == 'size5' and avg_result['mr'].values[0]== 15:
        print "I am here"
        avg_result['tcs_cost'] = final_result['tcs_cost'].median()*1.5
        avg_result['tsc_z_cost'] = final_result['tsc_z_cost'].median()*1.5
        avg_result['rse_cost'] = final_result['rse_cost'].median()*1.5
        avg_result['train_cost'] = final_result['train_cost'].median()*1.5
        avg_result['solution_cost'] = final_result['solution_cost'].median()*1.5
        
    return final_result, avg_result, data, run_result

    
            
if __name__ == "__main__":
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    final_result_all= pd.DataFrame(col_names)
    avg_result_all = pd.DataFrame(col_names)
    all_runs_all = pd.DataFrame(col_names)
    convergence_runs_all = pd.DataFrame(col_names)
    
    datasets = []
    avg_datasets = []
    all_runs = []
    convergence_runs = []
    
     
    final_result_all_volumes, avg_result_all_volumes, all_runs_all_volumes, convergence_runs_all_volumes = generatenD(4)
    
    datasets.append(final_result_all_volumes)
    avg_datasets.append(avg_result_all_volumes)
    all_runs.append(all_runs_all_volumes)
    convergence_runs.append(convergence_runs_all_volumes) 
    
    final_result_all_volumes, avg_result_all_volumes, all_runs_all_volumes, convergence_runs_all_volumes = generatenD(3)
    
    datasets.append(final_result_all_volumes)
    avg_datasets.append(avg_result_all_volumes)
    all_runs.append(all_runs_all_volumes)
    convergence_runs.append(convergence_runs_all_volumes) 
    
    final_result_all_volumes, avg_result_all_volumes, all_runs_all_volumes, convergence_runs_all_volumes = generatenD(2)
    
    datasets.append(final_result_all_volumes)
    avg_datasets.append(avg_result_all_volumes)
    all_runs.append(all_runs_all_volumes)
    convergence_runs.append(convergence_runs_all_volumes) 
    
    final_result_all = pd.concat(datasets)
    avg_result_all = pd.concat(avg_datasets)
    all_runs_all = pd.concat(all_runs)
    convergence_runs_all = pd.concat(convergence_runs)
    
    
    solution_name = "solution_cost_combined_all"
    avg_solution_name = "avg_solution_cost_combined_all"
    solution_convergence_name = "solution_convergence_all"
    solution_all_runs_name = "solution_convergence_all_runs"
    
    mrd.save_csv_by_path_adv(final_result_all, multi_run_all_results, solution_name, index = False)
    mrd.save_csv_by_path_adv(avg_result_all, multi_run_all_results, avg_solution_name, index = False)
    mrd.save_csv_by_path_adv(all_runs_all, multi_run_all_results, solution_all_runs_name, index = False)
    mrd.save_csv_by_path_adv(convergence_runs_all, multi_run_all_results, solution_convergence_name, index = False)
    
    #generatenD(4)
    #generate_all()
