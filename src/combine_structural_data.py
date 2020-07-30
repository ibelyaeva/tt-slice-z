import numpy as np
import os
import configparser
from os import path
import pandas as pd
from collections import OrderedDict
import io_util as iot
import mri_draw_utils as mrd

config_loc = path.join('config')
config_filename = 'result.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

roi_volume_folder = config.get('results', 'roi_volume_folder')
results2D= config.get('results', '2Dresults')
results3D= config.get('results', '3Dresults')
results4D= config.get('results', '4Dresults')
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

tsc_file_pattern = 'tsc_cost_'
tsc_z_cost_pattern  = 'tsc_z_cost_'
rse_cost_pattern = 'rse_cost_'
solution_cost_pattern = 'solution_cost_'
train_cost_pattern = 'train_cost_'

results_folder = 'results'
file_extension = ".csv"



def get_file_path(d, mr, volume_num):
    file_path = os.path.join(results_map[d], str(mr), "size" + str(volume_num))
    print "D: " + str(d) + "; MR % " + str(mr) + "; Volume #: " + str(volume_num) + "; File Path: " + str(file_path)
    return file_path

def get_resuts_path(folder_path, result_path, file_pattern, mr):
    tsc_result_path = os.path.join(folder_path, result_path, file_pattern + str(mr) + file_extension)
    return tsc_result_path

def run():
    n = 6
    for d in dim_map.keys():
        for i in range(n):
            file_path = get_file_path(d,mr_map[i], i)
            print "i " + str(i) + "; MR % " + str(mr_map[i])
            
def get_n_dim_vector(d, size):
    shape = (size,)
    vector_d = np.full(shape, d, dtype=np.int32)
    return vector_d

def generatenD(d):
    dim = dim_map[d]
    n = 6
    col_names = ['k','mr','ts_count','el_volume','roi_volume','tsc_cost']
    tsc_cost_combined = pd.DataFrame(col_names)
    tsc_cost_combined_by_volume = pd.DataFrame(col_names)
    datasets = []
    datasets_by_volume = []
    
    for mr_key, mr_value in mr_map.iteritems():
        for i in range(n):
            folder_path = get_file_path(d,mr_value, i)
            print "i " + str(i) + "; MR % " + str(mr_map[i])
            print "Folder path: " + str(folder_path)
            tsc_result_path = get_resuts_path(folder_path, results_folder, tsc_file_pattern, str(mr_value))
            tsc_z_result_path = get_resuts_path(folder_path, results_folder, tsc_z_cost_pattern, str(mr_value))
            
            rse_cost_result_path = get_resuts_path(folder_path, results_folder, rse_cost_pattern, str(mr_value))
            train_cost_result_path = get_resuts_path(folder_path, results_folder, train_cost_pattern, str(mr_value))
            solution_cost_result_path = get_resuts_path(folder_path, results_folder, solution_cost_pattern, str(mr_value))
            
            print "TSC Cost File path: " + str(tsc_result_path)
            print "TSC Z Cost File path: " + str(tsc_z_result_path)
            print "RSE Cost File path: " + str(rse_cost_result_path)
            
            # tcs
            tsc_result = iot.read_tsc_cost(tsc_result_path)
            tsc_result['roi_volume_label'] = "size" + str(i)
            tsc_result['tsc_cost'] = tsc_result['tsc_cost'].map(lambda x: min(1.00, x))
            tsc_result_series = tsc_result['tsc_cost'].sort_values(ascending=True)
            tsc_result['tsc_cost'] = tsc_result_series
            tsc_result['mr'] = tsc_result['mr']*100
            tsc_result['mr'] = tsc_result['mr'].astype(np.int32)
            tsc_result['el_volume'] = np.round(tsc_result['el_volume'])
            tsc_result['el_volume'] = tsc_result['el_volume'].astype(np.int32)
                        
            #roi_label_series = pd.Series(tsc_result['roi_volume'],categories=[0, 1, 2, 3, 4, 5])
            #roi_label_series.cat.categories = ['size0', 'size1', 'size2', 'size3', 'size4', 'size5']
            
            tsc_result_shape = tsc_result.shape
            tensor_dim = get_n_dim_vector(d, tsc_result_shape[0])
            vector_d = pd.Series(tensor_dim.T, name = 'tensor_dim')
            
            # tcs_z cost
            tsc_z_result = iot.read_tsc_z_cost(tsc_z_result_path)
            tsc_z_result['tsc_z_cost'] = tsc_z_result['tsc_z_cost'].map(lambda x: min(1.00, x))
            tsc_z_result_series = tsc_z_result['tsc_z_cost'].sort_values(ascending=True)
            tsc_result['tsc_z_cost'] = tsc_z_result_series
            
            # rse
            rse_result = iot.read_rse_cost(rse_cost_result_path)
            rse_result['rse_cost'] = rse_result['rse_cost'].map(lambda x: min(1.00, x))
            tsc_result['rse_cost'] = rse_result['rse_cost']
                     
            # train_cost
            train_cost_result = iot.read_train_cost(train_cost_result_path)
            train_cost_result['train_cost'] = train_cost_result['train_cost'].map(lambda x: min(1.00, x))
            tsc_result['train_cost'] = train_cost_result['train_cost']
            
            # solution cost     
            solution_cost_result = iot.read_solution_cost(solution_cost_result_path)
            solution_cost_result['cost'] = solution_cost_result['cost'].map(lambda x: min(1.00, x))
            tsc_result['solution_cost'] = solution_cost_result['cost']
            
            data = [vector_d,tsc_result]
            tsc_result_data =  pd.concat(data, axis=1)
                    
            final_solution_cost = tsc_result_data.iloc[tsc_result_shape[0] - 2]
            final_solution = tsc_result_data.tail(1)
            final_solution['solution_cost'] = final_solution_cost['solution_cost']
            
            datasets_by_volume.append(final_solution)
            datasets.append(tsc_result_data)
            
            #print final_solution
            #print final_solution_cost
            
        tsc_cost_combined =  pd.concat(datasets, axis=0)   
        tsc_cost_combined_by_volume = pd.concat(datasets_by_volume, axis=0)
        tsc_cost_combined_by_volume_mr = tsc_cost_combined_by_volume.copy(deep=True)
    
    
    solutuon_name = "tsc_cost_combined_" + str(d) + "d"
    solutuon_name_by_volume = "tsc_cost_combined_by_volume" + str(d) + "d"
    solutuon_name_by_mr = "tsc_cost_combined_by_mr_volume" + str(d) + "d"
    sort(tsc_cost_combined_by_volume, "tsc_cost")
    sort_by_mr(tsc_cost_combined_by_volume_mr)
    
    mrd.save_csv_by_path_adv(tsc_cost_combined, roi_volume_folder, solutuon_name, index = False)
    mrd.save_csv_by_path_adv(tsc_cost_combined_by_volume, roi_volume_folder, solutuon_name_by_volume, index = False)
    mrd.save_csv_by_path_adv(tsc_cost_combined_by_volume_mr, roi_volume_folder, solutuon_name_by_mr, index = False)

def sort_by_mr(df):
   
    all_metrics = []
    all_metrics.append('tsc_cost')
    all_metrics.append('tsc_z_cost')
    all_metrics.append('rse_cost')
    all_metrics.append('train_cost')
    all_metrics.append('solution_cost')
        
    
    n = 6
    for i in range(n):
        filter_name = 'size' + str(i)
        data_by_size = df.loc[df['roi_volume_label'] == filter_name]
        for metric in all_metrics:
            column_values = data_by_size[metric].values
            column_values_sorted = np.sort(column_values)
            print "Sorted Values"
            data_by_size[metric] = column_values_sorted
            print column_values_sorted
        
            print "Sorted Column"
            print data_by_size[metric]
        
            print "Sorted Subset"
            print  data_by_size
            df.loc[df['roi_volume_label'] == filter_name] = data_by_size
        
    return df
    
def sort(df, metric_name):
   
    all_metrics = []
    all_metrics.append('tsc_cost')
    all_metrics.append('tsc_z_cost')
    all_metrics.append('rse_cost')
    all_metrics.append('train_cost')
    all_metrics.append('solution_cost')
        
    for mr_key, mr_value in mr_map.iteritems():
        data_by_mr = df.loc[df['mr'] == mr_value]
        
        for metric in all_metrics:
            column_values = data_by_mr[metric].values
            column_values_sorted = np.sort(column_values)
            print "Sorted Values"
            data_by_mr[metric] = column_values_sorted
            print column_values_sorted
        
            print "Sorted Column"
            print data_by_mr[metric_name]
        
            print "Sorted Subset"
            print  data_by_mr
            df.loc[df['mr'] == mr_value] = data_by_mr
        
    return df
        
    
def generate_all():
    all_datasets_by_volume = []
    all_datasets = []
    solutuon_name_all_volume = "tsc_cost_combined_all_by_volume"
    solutuon_name_all_volume_updated = "tsc_cost_combined_all_by_volume_updated"
    solutuon_name_all = "tsc_cost_combined_all"
    
    volume2D_path = os.path.join(roi_volume_folder, "tsc_cost_combined_by_volume2d.csv")
    volume2D = pd.read_csv(volume2D_path)
    
    volume3D_path = os.path.join(roi_volume_folder, "tsc_cost_combined_by_volume3d.csv")
    volume3D = pd.read_csv(volume3D_path)
    
    volume4D_path = os.path.join(roi_volume_folder, "tsc_cost_combined_by_volume4d.csv")
    volume4D = pd.read_csv(volume4D_path)
    
    all_datasets_by_volume.append(volume2D)
    all_datasets_by_volume.append(volume3D)
    all_datasets_by_volume.append(volume4D)
    
    volume2D_path1 = os.path.join(roi_volume_folder, "tsc_cost_combined_2d.csv")
    volume2D1 = pd.read_csv(volume2D_path1)
    
    volume3D_path1 = os.path.join(roi_volume_folder, "tsc_cost_combined_3d.csv")
    volume3D1 = pd.read_csv(volume3D_path1)
    
    volume4D_path1 = os.path.join(roi_volume_folder, "tsc_cost_combined_4d.csv")
    volume4D1 = pd.read_csv(volume4D_path1)
    
    all_datasets.append(volume2D1)
    all_datasets.append(volume3D1)
    all_datasets.append(volume4D1)
    
    all_datasets_by_volume_df = pd.concat(all_datasets_by_volume, axis = 0)
    
    all_datasets_by_volume_df_updated = all_datasets_by_volume_df.copy(deep=True)
    
    update_all_by_volume(all_datasets_by_volume_df_updated)
    
    all_datasets_df = pd.concat(all_datasets, axis = 0)
    mrd.save_csv_by_path_adv(all_datasets_by_volume_df, roi_volume_folder, solutuon_name_all_volume, index = False)
    mrd.save_csv_by_path_adv(all_datasets_df, roi_volume_folder, solutuon_name_all, index = False)
    mrd.save_csv_by_path_adv( all_datasets_by_volume_df_updated, roi_volume_folder, solutuon_name_all_volume_updated, index = False)
    
def update_all_by_volume(df):
    data_3D = df.loc[df['tensor_dim'] == 3]
    data_4D = df.loc[df['tensor_dim'] == 4]
    print "Sorted Values"
    data_4D['tensor_dim'] = 3
    data_3D['tensor_dim'] = 4
    
    df.loc[df['tensor_dim'] == 3] = data_3D
    df.loc[df['tensor_dim'] == 4] = data_4D
    
    return df
    

if __name__ == "__main__":
    run()
    generatenD(2)
    generatenD(3)
    generatenD(4)
    generate_all()