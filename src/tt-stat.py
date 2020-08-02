import numpy as np
from scipy import stats
from scipy.special import stdtr
import tensor_util as tu
import metric_util as mt
from nilearn.masking import compute_epi_mask

file_path_tr45 = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45/images"
file_path_tr45low = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45/images/x_low_rank_hat_img.nii"
file_path_tr45noisy = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45/images/x_noisy_img.nii"
file_path_tr45original = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45/images/x_true_img.nii"

def meanTSNR(img):
    mask_img = compute_epi_mask(img)
    epi_mask = np.array(mask_img.get_data())

    data= np.array(img.get_data())
    data, data_norm = tu.normalize_data(data)
    
    mean_tsnr=np.mean(data[epi_mask==1])
    return mean_tsnr

def imgSTDEV(img):
    mask_img = compute_epi_mask(img)
    epi_mask = np.array(mask_img.get_data())

    data= np.array(img.get_data())
    data, data_norm = tu.normalize_data(data)
    
    std_data=np.std(data[epi_mask==1])
    return std_data

def t_test_img(imgt1, imgt2, imgt3):
    img1_t_data = np.array(imgt1.get_data())
    img2_t_data = np.array(imgt1.get_data())
    
    img1_tnsr = meanTSNR(imgt1)
    img2_tnsr = meanTSNR(imgt2)
    img3_tnsr = meanTSNR(imgt3)
    
    print("Image 1")
    img1_std_dev = imgSTDEV(imgt1)
    img2_std_dev = imgSTDEV(imgt2)
    img3_std_dev = imgSTDEV(imgt3)
    
    img1_tnsr_raw=float(img1_tnsr/img1_std_dev)  
    img1_tnsr_db = 10*np.log10(img1_tnsr_raw)
    
    img2_tnsr_raw=float(img2_tnsr/img2_std_dev)  
    img2_tnsr_db = 10*np.log10(img2_tnsr_raw)
    
    img3_tnsr_raw=float(img3_tnsr/img3_std_dev)  
    img3_tnsr_db = 10*np.log10(img3_tnsr_raw) 
    
    print ("img1 = mean = " + str(img1_tnsr_db) + "; stdDEV=" + str(img2_std_dev) + "; img1_tnsr_raw=" + str(img1_tnsr_raw) + "; img1_tnsr_db=" +str(img1_tnsr_db))
    
    print ("img2 = mean = " + str(img2_tnsr_db) + "; stdDEV=" + str(img2_std_dev) + "; img2_tnsr_raw=" + str(img2_tnsr_raw) + "; img2_tnsr_db=" +str(img2_tnsr_db))
    
    print("Image 3")
    print ("img3 = mean = " + str(img3_tnsr_db) + "; stdDEV=" + str(img3_std_dev) + "; img3_tnsr_raw=" + str(img3_tnsr_raw) + "; img3_tnsr_db=" +str(img3_tnsr_db))
    
    n1 = img1_t_data.size
    n1_dof = n1 - 1 
    
    n2 = img2_t_data.size
    n2_dof = n2 - 1 
    
    print("Image Low vs Noisy")
    t2, p2 = stats.ttest_ind_from_stats(img1_tnsr_raw, img1_std_dev, n1,
                              img2_tnsr_raw, img2_std_dev, n2,
                              equal_var=False)
    print("ttest_ind_from_stats: t = %g  p = %g" % (t2, p2))
    
    
def compute_t45_vs_ref(infilet1, infilet2, infiles1, infiles2):
        
        imgt1  = mt.read_image_abs_path(infilet1)
        imgt2 = mt.read_image_abs_path(infilet2)
        imgs1  = mt.read_image_abs_path(infiles1)
        imgs2  = mt.read_image_abs_path(infiles2)
        
        t_test_img(imgt1, imgt2, imgs1, imgs2)

def compute_t45_vs_low(infilet1, infilet2, infiles1):
        
        imgt1  = mt.read_image_abs_path(infilet1)
        imgt2 = mt.read_image_abs_path(infilet2)
        imgt3  = mt.read_image_abs_path(infiles1)
        
        
        t_test_img(imgt1, imgt2, imgt3)
    
if __name__ == "__main__":
   
    compute_t45_vs_low(file_path_tr45low, file_path_tr45noisy, file_path_tr45original)

    