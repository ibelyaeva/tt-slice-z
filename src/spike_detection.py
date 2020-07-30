import nipy
import tensor_util as tu
import nibabel as nib
from math import sqrt
import metric_util
import numpy as np
from nilearn import image
import os
import os.path as op
import numpy as np
import nibabel as nb
from scipy.ndimage.filters import median_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
from statsmodels.robust.scale import mad

def spectrum_mask(size):
    """Creates a mask to filter the image of size size"""
    import numpy as np
    from scipy.ndimage.morphology import distance_transform_edt as distance

    ftmask = np.ones(size)

    # Set zeros on corners
    # ftmask[0, 0] = 0
    # ftmask[size[0] - 1, size[1] - 1] = 0
    # ftmask[0, size[1] - 1] = 0
    # ftmask[size[0] - 1, 0] = 0
    ftmask[size[0] // 2, size[1] // 2] = 0

    # Distance transform
    ftmask = distance(ftmask)
    ftmask /= ftmask.max()

    # Keep this just in case we want to switch to the opposite filter
    ftmask *= -1.0
    ftmask += 1.0

    ftmask[ftmask >= 0.4] = 1
    ftmask[ftmask < 1] = 0
    return ftmask

#axial spike
#z-score threshold for spike detector
def slice_wise_fft(in_file, folder, ftmask=None, spike_thres=3., out_prefix='subject'):
    """Search for spikes in slices using the 2D FFT"""

    if out_prefix is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, _ = op.splitext(fname)
        out_prefix = op.abspath(fname)

    func_data = nb.load(in_file).get_data()

    if ftmask is None:
        ftmask = spectrum_mask(tuple(func_data.shape[:2]))

    fft_data = []
    for t in range(func_data.shape[-1]):
        func_frame = func_data[..., t]
        fft_slices = []
        for z in range(func_frame.shape[2]):
            sl = func_frame[..., z]
            fftsl = median_filter(np.real(np.fft.fft2(sl)).astype(np.float32),
                                  size=(5, 5), mode='constant') * ftmask
            fft_slices.append(fftsl)
        fft_data.append(np.stack(fft_slices, axis=-1))

    # Recompose the 4D FFT timeseries
    fft_data = np.stack(fft_data, -1)

    # Z-score across t, using robust statistics
    mu = np.median(fft_data, axis=3)
    sigma = np.stack([mad(fft_data, axis=3)] * fft_data.shape[-1], -1)
    idxs = np.where(np.abs(sigma) > 1e-4)
    fft_zscored = fft_data - mu[..., np.newaxis]
    fft_zscored[idxs] /= sigma[idxs]

    # save fft z-scored
    out_fft = op.abspath(out_prefix + '_zsfft.nii.gz')
    nii = nb.Nifti1Image(fft_zscored.astype(np.float32), np.eye(4), None)
    nii.to_filename(out_fft)

    # Find peaks
    spikes_list = []
    for t in range(fft_zscored.shape[-1]):
        fft_frame = fft_zscored[..., t]

        for z in range(fft_frame.shape[-1]):
            sl = fft_frame[..., z]
            if np.all(sl < spike_thres):
                continue

            # Any zscore over spike_thres will be called a spike
            sl[sl <= spike_thres] = 0
            sl[sl > 0] = 1

            # Erode peaks and see how many survive
            struc = generate_binary_structure(2, 2)
            sl = binary_erosion(sl.astype(np.uint8), structure=struc).astype(np.uint8)

            if sl.sum() > 10:
                print ((t, z), sl.sum() )
                spikes_list.append((t, z, sl.sum()))

    if folder is not None:
        file_path = os.path.join(folder,out_prefix)
    else:
        file_path = out_prefix
    out_spikes = op.abspath(file_path + '_spikes.tsv')
    np.savetxt(out_spikes, spikes_list, fmt=b'%d', delimiter=b'\t', header='TR\tZ')

    return len(spikes_list), out_spikes, out_fft, spikes_list

#n_spikes, out_spikes, out_fft, spikes_list = slice_wise_fft(subject_scan_path, spike_thres=4.)

def get_spiked_image(in_fft,tr):
    return image.index_img(in_fft,tr)

def get_spiked_overlay_by_z_score(in_fft,tr, z_score):
    spike_tr_img = get_spiked_image(in_fft,tr)
    spike_zscored_overlay_img = tu.get_z_score_robust_spatial_mask(spike_tr_img,z_score) 
    return spike_zscored_overlay_img

def get_spiked_tr_img_with_overlay(in_fft, tr, z_score):
    spike_tr_img = get_spiked_image(in_fft,tr)
    spike_zscored_overlay_img = get_spiked_overlay_by_z_score(in_fft,tr, z_score)
    return spike_tr_img, spike_zscored_overlay_img

def get_prev_tr_img_with_overlay(in_fft, tr, z_score):
      
    prev_spike_tr_img = None
    prev_spike_zscored_overlay_img = None
    
    if tr > 0:
        prev_spike_tr_img, prev_spike_zscored_overlay_img = get_spiked_tr_img_with_overlay(in_fft, tr - 1, z_score)
        
    return prev_spike_tr_img, prev_spike_zscored_overlay_img

    
def get_post_tr_img_with_overlay(in_fft, tr, z_score):
    data = np.array(in_fft.get_data())
    
    post_spike_tr_img = None
    post_spike_zscored_overlay_img = None
    
    ntpoints = data.shape[-1]
    if tr < (ntpoints - 1):
        post_spike_tr_img, post_spike_zscored_overlay_img = get_spiked_tr_img_with_overlay(in_fft, tr + 1, z_score)
        
    return post_spike_tr_img, post_spike_zscored_overlay_img

