import metadata_cloud as mdc
import metadata as md
import configparser
import os
from os import path
import google_drive_client as gcl
import mri_draw_utils as mrd
import metric_util as mt
from nilearn import plotting
from nilearn import image

config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

def traverse(dirname):
    for dirpath, dirs, files in os.walk(dirname):    
        folder_path = os.path.dirname(dirpath)
        print ("Dir Path: " + str(folder_path)) 
        for filename in files:
            fname = os.path.join(dirpath,filename)
            print ("Current Path: " + str(fname)) 
            
def traverse_dir(dirname):
    for dirpath, dirs, files in os.walk(dirname):    
        folder_path = os.path.dirname(dirpath)
        print ("Dir Path: " + str(folder_path)) 

def run():
    pass
            
def init_drive():
    root_dir = config.get('log', 'scratch.dirgoogledrive')
    meta = md.Metadata('random', 4)
    metadata = meta.init_meta_data(root_dir)
    client = gcl.GoogleClient(meta.logger, meta)
    #client.cd(directory='/UMBC/research')
    #client.pwd
    #client.cd(directory='1DXjM4vRdzUnNtEl-6Hdt80EPIU2svO-j', isID=True)
    #client.pwd
    #folder_id = client.getFolderId('mri_paper_completion')
    #print ("folder id:"  + str(folder_id))
    folder_id1 = client.get_ID_by_name('UMBC/research/mri_paper_completion')
    print ("4D folder id:"  + str(folder_id1))
    
    file_name = '/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/scans/final/mr/10/x_hat_img_10.nii'
    cloud_name = 'x_hat_img_10.nii'
    
    #client.upload_file_to_parent_folder(file_name, '1r7NBbVqiE2yvjvW6M8-6gjbr8UGKM1Sy',  cloud_name)
    #new_dir= '/UMBC/research/mri_paper_completion/test'
    #id = client.mkdir(new_dir)
    #print ("Id ?" + str(id))
    
    #file_list = client.list_folder('1laZh0X_DFlkxcpMWwW48h3OFNyh-NRsr')
    
    #for f in file_list:
    #    print f
    #client.get_parents('1qQw9VDuOWyBYJL9RJyIfqiwMSj5gvZ57')
    #client.print_files_in_folder('1laZh0X_DFlkxcpMWwW48h3OFNyh-NRsr')
    #client.download_to_file('1rlunlVBWJdq2ojTrghMxIGuZoIAfdgkB')
    #x_true_img  = mt.read_image_abs_path('x_hat_img_10.nii')
    
    #plotting.plot_epi(image.index_img(x_true_img,0), annotate=False, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)
    #plotting.show()
    #folderId2 = client.create_folder(folder_id1, "test2")
    #print "test folder: " + folderId2
    
    #folderId3 = client.create_folder(folderId2, "test3")
    #print "test folder: " + folderId3
    
    path = client.get_full_path('1n7EN3lA3Bgy5LHuxlC5rfvRCYOVpvaNa')
    
    print "path: " + str(path)
    
    #client.print_files_in_folder('1laZh0X_DFlkxcpMWwW48h3OFNyh-NRsr')
    
    #print "generating folder content ... wait"
    #content = client.generate_folder_content('1laZh0X_DFlkxcpMWwW48h3OFNyh-NRsr')
    
    parent_dir = '/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/'
    #traverse_dir(parent_dir)
    
    mr10_random = config.get('google_drive_random', 'mr20')
    #content = client.generate_folder_content(mr10_random)
    #local_file_path='/work/scratch/tensor_completion/4D/run_2018-07-20_07_36_13/d4/random/google_drive_folder'
    #client.generate_metadata_file(content, local_file_path)
   
    #full_path = '/work/scratch/tensor_completion/4D/run_2018-07-21_23_58_45/d4/random/google_drive_folder/google_drive_metadata_2018-07-26_01_27_42.csv'
    #file_path = os.path.basename(full_path) 
    #client.upload_file_to_parent_folder(client.google_meta_path, '1TRfyPaiSWklNAGZCo6i0swXETgIc0tCx')
    #folder_content = client.list_folder('1DXjM4vRdzUnNtEl-6Hdt80EPIU2svO-j')
    
    #for item in folder_content:
    #    print item
    
    parent_folder_id = '1-2TeWL9e5ByqOynwl3j55lOhz0YQQoAX'
    
    head, tail = os.path.split(parent_dir)
    tail = os.path.basename(os.path.normpath(parent_dir))
    print "Tail:"  + tail

    client.create_folder_rec(parent_dir, parent_folder_id)

if __name__ == "__main__":
    pass
    init_drive()