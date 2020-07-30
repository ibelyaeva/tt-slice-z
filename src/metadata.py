import data_util as du
import metric_util as mt
import configparser
from os import path
import logging
from datetime import datetime
import file_service as fs
import os
import csv

class Metadata(object):
    
    
    def __init__(self, pattern, n):
        config_loc = path.join('config')
        #config_filename = 'solution-ec2.config'
        config_filename = 'solution.config'
        config_file = os.path.join(config_loc, config_filename)
        config = configparser.ConfigParser()
        config.read(config_file)
        self.config = config
        self.logger = create_logger(self.config)
        self.pattern = pattern
        self.n = n

    def init_meta_data(self, root_dir):
    
        #root_dir = self.config.get('log', 'scratch.dir')
        self.root_dir = root_dir
        dim_dir = "d" + str(self.n)
        self.solution_dir = path.join(dim_dir, self.pattern)
        
        self.solution_folder = fs.create_batch_directory(root_dir, self.solution_dir)
       
        self.images_folder = fs.create_batch_directory(self.solution_folder, "images", False)
        self.results_folder = fs.create_batch_directory(self.solution_folder, "results", False)
        self.reports_folder = fs.create_batch_directory(self.solution_folder, "reports", False)
        self.scans_folder = fs.create_batch_directory(self.solution_folder, "scans", False)
        self.ellipsoid_folder = fs.create_batch_directory(self.solution_folder, "ellipsoid_folder", False)
        self.scans_folder_final = fs.create_batch_directory(self.scans_folder, "final", False)
        self.scans_folder_iteration = fs.create_batch_directory(self.scans_folder, "iteration", False)
        self.movies_folder = fs.create_batch_directory(self.solution_folder, "movie", False)
        self.google_drive_folder = fs.create_batch_directory(self.solution_folder, "google_drive", False)
            
        self.logger.info("Created solution dir at: [%s]" ,self.solution_folder)
        self.logger.info("Created images dir at: [%s]" ,self.images_folder)
        self.logger.info("Created results dir at: [%s]" ,self.results_folder)
        self.logger.info("Created reports dir at: [%s]" ,self.reports_folder)
        self.logger.info("Created scans_folder_final dir at: [%s]" ,self.scans_folder_final)
        self.logger.info("Created scans_folder_iteration at: [%s]" ,self.scans_folder_iteration)
        self.logger.info("Created google_drive_folder at: [%s]" ,self.google_drive_folder)
        
        return self.solution_dir, self.movies_folder, self.images_folder, self.results_folder, self.reports_folder, self.scans_folder, self.scans_folder_final, self.scans_folder_iteration
    
    def create_scan_mr_folder(self, mr):
        suffix = self.get_suffix(mr)
        self.scan_mr_dir = path.join(self.scans_folder_final, 'mr')
        self.scan_mr_folder = fs.create_batch_directory(self.scan_mr_dir, suffix, False)
        self.logger.info("Created MR [%s] Folder at: [%s]" ,suffix, self.scan_mr_folder)
        return self.scan_mr_folder
        
    def create_scan_mr_folder_iteration(self, mr):
        suffix = self.get_suffix(mr)
        self.scans_folder_iteration_dir = path.join(self.scans_folder_iteration, 'mr')
        self.scan_mr_iteration_folder = fs.create_batch_directory(self.scans_folder_iteration_dir, suffix, False)
        self.logger.info("Created MR Iteration [%s] Folder at: [%s]" ,suffix, self.scan_mr_iteration_folder)
        return self.scan_mr_iteration_folder
        
    def create_images_mr_folder_iteration(self, mr):
        suffix = self.get_suffix(mr)
        self.images_folder_iteration_dir = path.join(self.images_folder, 'mr')
        self.scan_images_iteration_folder = fs.create_batch_directory(self.images_folder_iteration_dir, suffix, False)
        self.images_folder_mr_final_dir = fs.create_batch_directory(self.scan_images_iteration_folder,'final', False)
        self.images_metadata_folder = fs.create_batch_directory(self.scan_images_iteration_folder, 'metadata', False)
        self.logger.info("Created MR Images Iteration [%s] Folder at: [%s]" ,suffix, self.scan_images_iteration_folder)
        self.logger.info("Created MR Images Final Iteration [%s] Folder at: [%s]" ,suffix, self.images_folder_mr_final_dir)
        self.logger.info("Created MR Images Metadata [%s] Folder at: [%s]" ,suffix, self.images_metadata_folder)
        return self.scan_images_iteration_folder      
    
    def get_suffix(self,mr):    
        suffix = str(int(round((mr) * 100.0, 0)))
        return suffix
    
    def set_suffix(self,observed_ratio):
        mr = 1.00 - observed_ratio
        self.suffix = str(int(round((mr) * 100.0, 0)))
        return self.suffix
    
    def create_solution_file(self, solution_label):
       
        self.col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
        
        self.set_solution_label(solution_label)   
        self.global_solution_path = os.path.join(self.root_dir, 'global_solution_'+ str(self.n) + '_' + self.pattern +'.csv')
        if not os.path.exists(self.global_solution_path):
            with open(self.global_solution_path,"ab") as global_solution_file:
                writer  = csv.DictWriter(global_solution_file, fieldnames=self.col_names)
                writer.writeheader()
                
            
            global_solution_file.close()
            
    def create_solution_file_by_mr(self, observed_ratio, solution_label):
       
        self.col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
        
        self.set_suffix(observed_ratio)
        self.set_solution_label(solution_label)
        self.global_solution_path = os.path.join(self.root_dir, 'global_solution_'+ str(self.n) + 'D_' + self.pattern + '_' + 'mr' + str(self.suffix) + '_' +self.solution_label + '.csv')
        if not os.path.exists(self.global_solution_path):
            with open(self.global_solution_path,"ab") as global_solution_file:
                writer  = csv.DictWriter(global_solution_file, fieldnames=self.col_names)
                writer.writeheader()
                
            
            global_solution_file.close()
            
        
    def set_solution_label(self, solution_label):
        self.solution_label = solution_label
    
def create_logger(config):
    
    ##########################################
    # logging setup
    ##########################################
    
    start_date = str(datetime.now())
    r_date = "{}-{}-{}".format(start_date[0:4], start_date[5:7], start_date[8:10])
    log_dir = config.get("log", "log.dir")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    current_date = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    # create a file handler
    app_log_name = 'tensor_completion_' +  str(current_date) + '.log'
    app_log = path.join(log_dir, app_log_name)
    handler = logging.FileHandler(app_log)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.info("Tensor Completion...")

    ##########################################
    # end logging stuff:
    ##########################################
    logger.info('Starting @ {}'.format(str(current_date)))
    
    return logger
