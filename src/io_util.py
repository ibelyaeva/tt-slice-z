import numpy as np
import csv
import pandas as pd

def read_tsc_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tsc_cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'tsc_cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_multi_run_solution(file_path):
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    dtype_names = {'k': np.int32, 'mr': np.float32, 
                   'observed_ratio': np.float32,
                   'ts_count': np.int32,
                   'el_volume': np.float32, 
                   'roi_volume': np.int32, 
                   'tsc_cost': np.float64,
                   'tsc_z_cost': np.float64,
                   'rse_cost': np.float64,
                   'train_cost': np.float64,
                   'solution_cost': np.float64
                    }
    
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_tsc_z_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tsc_z_cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'tsc_z_cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_rse_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'rse_cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'rse_cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_train_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'train_cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'train_cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_solution_cost(file_path):
    col_names = ['k', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'cost']
    dtype_names = {'k': np.int32, 'mr': np.float32, 'ts_count': np.int32, 'el_volume': np.float32, 'roi_volume': np.int32, 'cost': np.float64 }
    result = pd.read_csv(file_path, sep=',', usecols=col_names, dtype=dtype_names)
    return result

def read_data_structural(file_path):
    result = pd.read_csv(file_path, sep=',')
    return result

def read_data_by_path(file_path):
    result = pd.read_csv(file_path, sep=',')
    return result

def write_header(file_path, col_names):
    with open(file_path,"ab") as solution_file:
                writer  = csv.DictWriter(solution_file, fieldnames=col_names)
                writer.writeheader()
                
def append_row(file_path, row, col_names):
    
    with open(file_path,"ab") as solution_file:
            writer  = csv.DictWriter(solution_file, fieldnames=col_names)
            writer.writerow(row)
            
    solution_file.close()
