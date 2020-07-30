import texfig

import os
import data_util as du
import configparser
from os import path
import metadata as mdt
import complete_tensor_structural as ct
import structural_pattern_generator as stp


config_loc = path.join('config')
config_filename = 'solution.config'
#config_filename = 'solution-ec2.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config



def complete_random_4D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.95]
    
    x0, y0, z0 = (2,32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()
      
    #size 2 
    #x_r, y_r, z_r = (9,10,8)
    #for item in observed_ratio_list:
    #    current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
    #    current_runner.complete()

    #size 3
    #x_r, y_r, z_r = (12,10,8)
    #for item in observed_ratio_list:
    #    current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
    #    current_runner.complete()
        
def complete_random_4Dsize1():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('structural', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    #observed_ratio_list = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.95, 0.9,0.85,0.8,0.75,0.7]
    
    #for item in observed_ratio_list:
    #    mr = 1.0 - item
    #    frame_count = int(mr*144)
    #    ts = stp.create_random_frames(144, frame_count)
    #    meta.logger.info("MR = " + str(mr) + "; " + str(ts.values()) + "; Timepoint Count: " + str(frame_count))
    
   
    x0, y0, z0 = (2,32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()
        
def complete_random_4Dsize2():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('structural', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    #observed_ratio_list = [0.95, 0.9, 0.8, 0.7, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.25, 0.1]
    #observed_ratio_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
    observed_ratio_list = [0.7] 
    
    x0, y0, z0 = (2,32,22)
      
    #size 2 
    x_r, y_r, z_r = (9,10,8)
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()
        
def complete_random_4Dsize3():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('structural', 4)
    root_dir = config.get('log', 'scratch.dir4Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    #observed_ratio_list = [0.95,0.9, 0.8, 0.7, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.25, 0.1]
    observed_ratio_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    
    x0, y0, z0 = (2,32,22)
  
    #size 3
    x_r, y_r, z_r = (12,10,8)
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()

def complete_random_3D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 3)
    root_dir = config.get('log', 'scratch.dir3D')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.9]
    
    x0, y0, z0 = (2, 32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 3, 0, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()

        
def complete_random_2D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 2)
    root_dir = config.get('log', 'scratch.dir2D')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    observed_ratio_list = [0.25]
    
    x0, y0, z0 = (2, 32,22)
    # size 1
    x_r, y_r, z_r = (7,10,8)
    
    for item in observed_ratio_list:
        current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 2, 1, meta.logger, meta, x0, y0, z0, x_r, y_r, z_r)
        current_runner.complete()
        
def complete_random_2D_by_tr_size():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('structural', 2)
    root_dir = config.get('log', 'scratch.dir2Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.95, 0.9,0.85,0.8,0.75,0.7]
    
    observed_ratio_list = [0.9]
    
    # ellipse center
    x0, y0, z0 = (2,32,22)
    
    # volumes
    volumes_list = []
    volumes_list.append((7,10,8))
    volumes_list.append((8,10,8))
    volumes_list.append((9,10,8))
    volumes_list.append((10,10,8))
    volumes_list.append((11,10,8))
    volumes_list.append((12,10,8))
    
    volumes_label = {}
    #volumes_label['size0'] = (7,10,8)
    #volumes_label['size1'] = (8,10,8)
    #volumes_label['size2'] = (9,10,8)
    #volumes_label['size3'] = (10,10,8)
    #volumes_label['size4'] = (11,10,8)
    volumes_label['size5'] = (12,10,8)
    
    
    n= 1
    
    for item in observed_ratio_list:
        for i in range (0, n, 1):
                for el_key in sorted(volumes_label):
                    el_value = volumes_label[el_key]
                    print "Processing Volume Size: " + str(el_key) + "; Volume Value: " + str(el_value)
                    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
                    meta.create_solution_file_by_mr(item, el_key)
                    current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 2, 1, meta.logger, meta, x0, y0, z0, el_value[0], el_value[1], el_value[2])
                    current_runner.complete()
            
def complete_random_3D_by_tr_size():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('structural', 3)
    root_dir = config.get('log', 'scratch.dir3Dstr')
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.95, 0.9,0.85,0.8,0.75,0.7]
            
    x0, y0, z0 = (2,32,22)
    
    # volumes
    volumes_list = []
    volumes_list.append((7,10,8))
    volumes_list.append((8,10,8))
    volumes_list.append((9,10,8))
    volumes_list.append((10,10,8))
    volumes_list.append((11,10,8))
    volumes_list.append((12,10,8))
    
    volumes_label = {}
    #volumes_label['size0'] = (7,10,8)
    #volumes_label['size1'] = (8,10,8)
    #volumes_label['size2'] = (9,10,8)
    #volumes_label['size3'] = (10,10,8)
    #volumes_label['size4'] = (11,10,8)
    #volumes_label['size5'] = (12,10,8)
    
    volumes_label['size1'] = (20,15,13)
    
    n= 1
    
    for item in observed_ratio_list:
        for i in range (0, n, 1):
                for el_key in sorted(volumes_label):
                    el_value = volumes_label[el_key]
                    print "Processing Volume Size: " + str(el_key) + "; Volume Value: " + str(el_value)
                    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
                    meta.create_solution_file_by_mr(item, el_key)
                    current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 3, 1, meta.logger, meta, x0, y0, z0, el_value[0], el_value[1], el_value[2])
                    current_runner.complete()
    
        
def complete_random_4D_by_tr_size():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('structural', 4)
    root_dir = config.get('log', 'scratch.dir4Dslice')
    
    observed_ratio_list = [0.95, 0.9,0.85,0.8,0.75,0.7]
    
    observed_ratio_list = [0.95]
    
    # ellipse center
    x0, y0, z0 = (2,32,22)
    
    # volumes
    volumes_list = []
    volumes_list.append((7,10,8))
    volumes_list.append((8,10,8))
    volumes_list.append((9,10,8))
    volumes_list.append((10,10,8))
    volumes_list.append((11,10,8))
    volumes_list.append((12,10,8))
    
    volumes_label = {}
    #volumes_label['size0'] = (7,10,8)
    volumes_label['size1'] = (25,20,20)
    #volumes_label['size2'] = (9,10,8)
    #volumes_label['size3'] = (10,10,8)
    #volumes_label['size4'] = (11,10,8)
    #volumes_label['size5'] = (12,10,8)
    
    n= 1
    frames_count = 4
    start_n = 45
    for item in observed_ratio_list:
        for i in range (0, n, 1):
            for el_key in sorted(volumes_label):
                el_value = volumes_label[el_key]
                print "Processing Volume Size: " + str(el_key) + "; Volume Value: " + str(el_value)
                solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
                meta.create_solution_file_by_mr(item, el_key)
                current_runner = ct.TensorCompletionStructural(subject_scan_path, item, 4, 1, meta.logger, meta, x0, y0, z0, el_value[0], el_value[1], el_value[2], frames_count, start_n)
                current_runner.complete()
        



if __name__ == "__main__":
    pass
    #complete_random_2D()
    #complete_random_4Dsize1()

    #complete_random_4Dsize2()
    #complete_random_4Dsize3()

    complete_random_4D_by_tr_size()
    #complete_random_2D_by_tr_size()
    #complete_random_3D_by_tr_size()
    #complete_random_4Dsize2()

    #complete_random_3D()

    
    
