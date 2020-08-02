from nilearn import masking
import nibabel as nb
import matplotlib.pyplot as plt
import nilearn
import numpy as np
from nilearn.datasets import load_mni152_template
from nilearn.image import new_img_like, iter_img
from nilearn.masking import compute_epi_mask, apply_mask
from nilearn.signal import clean

def plot_carpet(img, mask_img=None, detrend=True, output_file=None,
                figure=None, axes=None, vmin=None, vmax=None, title=None):
    """Plot an image representation of voxel intensities across time.
    This figure is also known as a "grayplot" or "Power plot".
    Parameters
    ----------
    img : Niimg-like object
        4D image.
        See http://nilearn.github.io/manipulating_images/input_output.html.
    mask_img : Niimg-like object or None, optional
        Limit plotted voxels to those inside the provided mask (default is
        None). If not specified a new mask will be derived from data.
        See http://nilearn.github.io/manipulating_images/input_output.html.
    detrend : :obj:`bool`, optional
        Detrend and z-score the data prior to plotting (default is `True`).
    output_file : :obj:`str` or None, optional
        The name of an image file to which to export the plot (default is
        None). Valid extensions are .png, .pdf, and .svg.
        If `output_file` is not None, the plot is saved to a file, and the
        display is closed.
    figure : :class:`matplotlib.figure.Figure` or None, optional
        Matplotlib figure used (default is None).
        If None is given, a new figure is created.
    axes : matplotlib axes or None, optional
        The axes used to display the plot (default is None).
        If None, the complete figure is used.
    title : :obj:`str` or None, optional
        The title displayed on the figure (default is None).
    Returns
    -------
    figure : :class:`matplotlib.figure.Figure`
        Figure object with carpet plot.
    Notes
    -----
    This figure was originally developed in [1]_.
    In cases of long acquisitions (>800 volumes), the data will be downsampled
    to have fewer than 800 volumes before being plotted.
    References
    ----------
    .. [1] Power, J. D. (2017). A simple but useful way to assess fMRI scan
            qualities. Neuroimage, 154, 150-158. doi:
            https://doi.org/10.1016/j.neuroimage.2016.08.009
    """
    #img = _utils.check_niimg_4d(img, dtype='auto')

    # Define TR and number of frames
    tr = img.header.get_zooms()[-1]
    n_tsteps = img.shape[-1]

    if mask_img is None:
        mask_img = compute_epi_mask(img)
    else:
        mask_img = _utils.check_niimg_3d(mask_img, dtype='auto')

    data = apply_mask(img, mask_img)
    # Detrend and standardize data
    if detrend:
        data = clean(data, t_r=tr, detrend=True, standardize='zscore')

    if figure is None:
        if not axes:
            figsize = (10, 5)
            figure = plt.figure(figsize=figsize)
        else:
            figure = axes.figure

    if axes is None:
        axes = figure.add_subplot(1, 1, 1)
    else:
        assert axes.figure is figure, ("The axes passed are not "
                                       "in the figure")

    # Determine vmin and vmax based on the full data
    std = np.mean(data.std(axis=0))
    default_vmin = data.mean() - (2 * std)
    default_vmax = data.mean() + (2 * std)

    # Avoid segmentation faults for long acquisitions by decimating the data
    LONG_CUTOFF = 800
    # Get smallest power of 2 greater than the number of volumes divided by the
    # cutoff, to determine how much to decimate (downsample) the data.
    n_decimations = int(np.ceil(np.log2(np.ceil(n_tsteps / LONG_CUTOFF))))
    data = data[::2 ** n_decimations, :]

    axes.imshow(data.T, interpolation='nearest',
                aspect='auto', cmap='gray',
                vmin=vmin or default_vmin,
                vmax=vmax or default_vmax)

    axes.grid(False)
    axes.set_yticks([])
    axes.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = max(
        (int(data.shape[0] + 1) // 10, int(data.shape[0] + 1) // 5, 1))
    xticks = list(range(0, data.shape[0])[::interval])
    axes.set_xticks(xticks)

    axes.set_xlabel('time (s)')
    axes.set_ylabel('voxels')
    if title:
        axes.set_title(title)
    labels = tr * (np.array(xticks))
    labels *= (2 ** n_decimations)
    axes.set_xticklabels(['%.02f' % t for t in labels.tolist()])

    # Remove and redefine spines
    for side in ['top', 'right']:
        # Toggle the spine objects
        axes.spines[side].set_color('none')
        axes.spines[side].set_visible(False)

    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')
    axes.spines['bottom'].set_position(('outward', 20))
    axes.spines['left'].set_position(('outward', 20))

    if output_file is not None:
        figure.savefig(output_file)
        plt.close(figure)
        figure = None

    return figure


