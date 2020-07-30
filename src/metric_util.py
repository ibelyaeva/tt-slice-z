import nilearn

from nilearn import image
import nibabel as nib
import copy
import numpy as np
from nilearn.image.resampling import coord_transform, get_bounds, get_mask_bounds
from nilearn.masking import compute_background_mask

def get_xyz(i, j, k, epi_img):
    M = epi_img.affine[:3, :3]
    abc = epi_img.affine[:3, 3]
    return M.dot([i, j, k]) + abc

def xyz_coord(img):
    affine = img.affine.copy()
    data_coords = list(np.ndindex(img.shape[:3]))
    data_coords = np.asarray(list(zip(*data_coords)))
    data_coords = coord_transform(data_coords[0], data_coords[1],
                                  data_coords[2], affine)
    data_coords = np.asarray(data_coords).T
    return data_coords

def reconstruct_image_affine(img_ref, x_hat):
    result = nib.Nifti1Image(x_hat, img_ref.affine)
    return result

def reconstruct_image_affine_d(img_ref, x_hat, d, target_shape):
    if d >3:
        result = nib.Nifti1Image(x_hat, img_ref.affine)
    else:
        x = reshape_to4D(x_hat, target_shape)
        result = nib.Nifti1Image(x, img_ref.affine)
    return result

def reshape_to4D(x, target_shape):        
    x_org = copy.deepcopy(x)
    x_reshaped = np.reshape(x_org, (target_shape[0], target_shape[1], target_shape[2], target_shape[3]))
    return x_reshaped

def get_target_shape(x, d):
    if d == 2:
        target_shape = x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]
    elif d == 3:
        target_shape = x.shape[0]*x.shape[1],x.shape[2],x.shape[3]
        #target_shape = x.shape
    else:
        target_shape = x.shape
    return target_shape

def reshape_as_nD(x, d,target_shape):
        
    x_org = copy.deepcopy(x)
        
    if d == 2:
        print("Reshape Required. D = " + str(d) + "; Target Shape: " + str(target_shape))
        num_rows = target_shape[0]
        x_reshaped = np.reshape(x_org, (num_rows,target_shape[1]))
        print("Resulted Target Shape: " + str(x_reshaped.shape))
    elif d == 3:
        print("Reshape Required. D = " + str(d) + "; Target Shape: " + str(target_shape))
        num_rows = target_shape[0]
        x_reshaped = np.reshape(x_org, (num_rows,target_shape[1], target_shape[2]))  
        #x_reshaped = x_org
        #print("No Reshape Required. D: "  + str(d))
        print("Resulted Target Shape: " + str(x_reshaped.shape))
    elif d == 4:
        # do nothing
        x_reshaped = x_org
        print("No Reshape Required. D: "  + str(d))
        print("Resulted Target Shape: " + str(x_reshaped.shape))
    else:
        errorMsg = "Unknown Tensor Dimensionality. Cannot Reshape Image"
        raise(errorMsg)
    
    return x_reshaped
            
      

def get_box_coord(img):
    box_coordinates = get_mask_bounds(img)
    x_min = box_coordinates[0]
    x_max = box_coordinates[1]
    y_min = box_coordinates[2]
    y_max = box_coordinates[3]
    z_min = box_coordinates[4]
    z_max = box_coordinates[5]
    return x_min, x_max, y_min, y_max, z_min, z_max

def ellipsoid_masker(x_r, y_r, z_r, x0, y0, z0, img):
    # compute background mask
    brain_mask = compute_background_mask(img)
    # build a mesh grid as per original image
    x_min, x_max, y_min, y_max, z_min, z_max = get_box_coord(img)
    print (x_min, x_max, y_min, y_max, z_min, z_max)
    x_spacing = abs(img.affine[0,0])
    y_spacing = abs(img.affine[1,1])
    z_spacing = abs(img.affine[2,2])
    print ("X-spacing: " +str(x_spacing) + "; Y-spacing: " + str(y_spacing) + "; Z-spacing: " + str(z_spacing))
    
     # build a mesh grid as per original image
    x, y, z = np.mgrid[x_min:x_max+1:x_spacing, y_min:y_max+1:y_spacing, z_min:z_max+1:z_spacing]
    
    # analytical ellipse equation
    xx, yy, zz = np.nonzero((((x - x0) / float(x_r)) ** 2 +
               ((y - y0) / float(y_r)) ** 2 +
               ((z - z0) / float(z_r)) ** 2) <= 1)
    # create array with the same shape as brain mask to mask entries of interest
    activation_mask = np.zeros(img.get_data().shape)
    ##mask = (np.random.rand(xx.shape[0],yy.shape[0],zz.shape[0]) < 0.05).astype('int')
    #print "Mask" + str(mask.shape)
    print ("XX sahpe: " + str(xx.shape))
    xx1 = np.random.choice(xx, size = (xx.shape[0]/60))
    yy1 = np.random.choice(yy, size = (yy.shape[0]/60))
    zz1 = np.random.choice(zz, size = (zz.shape[0]/60))
    
    zz2 = (np.random.choice(zz, size = zz.shape)< 0.05).astype('int')
    print (zz2)
    #x_train[mask==0] = 0
    activation_mask[xx1, yy1, zz1] = 1
    activation_img = nib.Nifti1Image(activation_mask, affine=img.affine)
    return activation_img    

def apply_ellipse_mask(mask_img, img):
    data = copy.deepcopy(img.get_data())
    data[mask_img.get_data() > 0] = 0
    masked_image = reconstruct_image_affine(img, data)
    return masked_image

def create_image_with_ellipse_mask(x_r, y_r, z_r, x0, y0, z0, img):
    mask = ellipsoid_masker(x_r, y_r, z_r, x0, y0, z0, img)
    image_masked = apply_ellipse_mask(mask,img)
    return image_masked  
    

def read_image(folder, path):
    img = nib.load(folder + "/" + path)
    return img

def read_image_abs_path(path):
    img = nib.load(path)
    return img

def read_frame(folder, path, n):
    img4D = read_image(folder, path)
    img3D = image.index_img(img4D,n)
    return img3D

def read_frame_by_full_path(path, n):
    img4D = read_image_abs_path(path)
    img3D = image.index_img(img4D,n)
    return img3D

def get_data(img):
    data = img.get_data()
    return data

def relative_error(x_hat,x_true):
    percent_error = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    return percent_error

def compute_observed_ratio(target_img):
    data = np.array(target_img.get_data())
    nonzero_indices_count = np.count_nonzero(data)
    
    print ("nnz = " + str(nonzero_indices_count))
    print ("size = " + str(data.size))
    result = 1.0 - float(1.0*nonzero_indices_count/data.size)
    return result

def compute_observed_ratio_arr(x):
    nonzero_indices_count = np.count_nonzero(x)
    
    print ("nnz = " + str(nonzero_indices_count))
    print ("size = " + str(x.size))
    result = 1.0 - float(1.0*nonzero_indices_count/x.size)
    return result

def tsc(x_hat,x_true, ten_ones, mask):
    nomin = np.linalg.norm(np.multiply((ten_ones - mask), (x_true - x_hat)))
    denom = np.linalg.norm(np.multiply((ten_ones - mask), x_true))
    score = nomin/denom
    return score  

def reconstruct(x_hat,x_true, ten_ones, mask):
    x_reconstruct = np.multiply(mask,x_true) + np.multiply((ten_ones - mask), x_hat)
    return x_reconstruct

def reconstruct2(x_hat,x_true, mask):
    x_reconstruct = x_true * mask + x_hat * (1 - mask)
    return x_reconstruct
