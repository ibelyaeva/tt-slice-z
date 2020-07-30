import texfig
import matplotlib.pyplot as plt
import numpy as np
import os
import configparser
from os import path
import pandas as pd
import matplotlib.gridspec as gridspec
import io_util as iot
import locale
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from cycler import cycler
import matplotlib.path as mpath
import mri_draw_utils as mrd
import math_format as mf
import matplotlib.ticker as ticker
import math_format as mf
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import ScalarFormatter

config_loc = path.join('config')
config_filename = 'alternate_minim_result.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

roi_volume_folder = config.get('results', 'roi_volume_folder')
structural_figures = config.get('results', 'structural_figures')

alternate_minim_and_godec = "/work/project/cmsc655/results/alternate_min/alternate_minimization_solution_and_godec.csv"

data2D_path = 'tsc_cost_combined_2d.csv'
data3D_path = 'tsc_cost_combined_3d.csv'
data3D_path = 'tsc_cost_combined_4d.csv'
data_all_path = 'tsc_cost_combined_all.csv'
data_all_by_volume = 'tsc_cost_combined_all_by_volumev1.csv'
data_all_by_volume2 = 'tsc_cost_combined_all_by_volumev2.csv'
data_by_volume2D = 'tsc_cost_combined_by_volume2d.csv'
data_by_volume3D = 'tsc_cost_combined_by_volume3d.csv'
data_by_volume4D = 'tsc_cost_combined_by_volume4d.csv'
data_by_mr_by_volume3D = 'tsc_cost_combined_by_mr_volume3d.csv'
data_by_mr_by_volume2D = 'tsc_cost_combined_by_mr_volume2d.csv'
data_by_mr_by_volume4D = 'tsc_cost_combined_by_mr_volume4d.csv'

results_path = "/work/project/cmsc655/results/alternate_min/"
richian_noise = "/work/project/cmsc655/results/alternate_min/richian"
guassian_noise = "/work/project/cmsc655/results/alternate_min/guassian/"
raleign_noise = "/work/project/cmsc655/results/alternate_min/raleign//"

top_legend_handles = []
top_labels = []

top_legend_handles3D = []
top_labels3D = []

top_legend_handles2D = []
top_labels2D = []

top_legend_handles3D_mr = []
top_labels3D_mr = []

top_legend_handles4D_mr = []
top_labels4D_mr = []

top_legend_handles2D_mr = []
top_labels2D_mr = []

top_legend_tensor_dim = []
top_labelstensor_dim = []

star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)

tex_width = 5.78853 # in inches
tex_width = 4.78853 # in inches

noise_type = []
noise_type.append('richian')
noise_type.append('raleign')
noise_type.append('guassian')

snr_level = []

snr_level.append(5)
snr_level.append(10)
snr_level.append(15)
snr_level.append(20)
snr_level.append(25)
snr_level.append(35)
snr_level.append(40)

bg_legend_handle = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [0.1,1,10]:
            return LogFormatterSciNotation.__call__(self,x, pos=None)
        else:
            return "{x:g}".format(x=x)
        
def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('k')
        ax.spines[spine].set_linewidth(1)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('k')
        ax.spines[spine].set_linewidth(1)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='in', top = True, left = True, color='k')

    return ax

def legend_outside(ncol, extra_text):
    handles, labels = plt.gca().get_legend_handles_labels()
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(-0.15, -0.3))
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=lgd.get_texts()[0].get_fontsize()))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_top(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='lower left', ncol=ncol, bbox_to_anchor=(0., 1.02, 1., .102))
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_left(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(0.01, 0.98), fontsize="small", edgecolor='k')
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_right(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(0.6, 0.98), fontsize="small", edgecolor='k')
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_right3(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(0.01, 0.9), fontsize="small", edgecolor='k')
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_right2(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(0.15, 0.98), fontsize="small", edgecolor='k')
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_right4(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(0.05, 0.95), fontsize="small", edgecolor='k')
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_lower_right4(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper right', ncol=ncol, bbox_to_anchor=(0.95, 0.2), fontsize="small", edgecolor='k')
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_left_inset(ncol, handles, labels, extra_text, x, y):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(x, y), fontsize="small", edgecolor='k')
    lgd.set_zorder(20)
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_bottom(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(-0.15, -0.17),fontsize="x-small", edgecolor='k')
    return lgd

def create_legend_on_bottom_inset(ncol, handles, labels, extra_text, x, y):
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(x, y),fontsize="x-small", edgecolor='k')
    return lgd

def legend_outside1(ncol, extra_text):
    handles, labels = plt.gca().get_legend_handles_labels()
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.55, -0.15), frameon=False)
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def group(number):
    s = '%d' % number
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    return s + ','.join(reversed(groups))


# FuncFormatter can be used as a decorator
@ticker.FuncFormatter
def major_perc_formatter(x, pos):
    return "%.2f" % x

def plot_volume(data, metric_name, filter_name,  **kwargs):
    data_size = data.loc[data['roi_volume_label'] == filter_name]
    x = data_size['mr']
    y = data_size[metric_name]
    roi_volume = data_size.tail(1)['el_volume']
    lw = 1
    label = ur'ROI Volume: ' + group(roi_volume)
    result = plt.plot(x, y, linewidth=lw, label = "")
    return label, result

def plot_mr(data, metric_name, filter_name,  **kwargs):
    data_size = data.loc[data['mr'] == filter_name]
    y = data_size[metric_name]
    x = data_size['el_volume']
    lw = 1
    label = r'MR ' + str(filter_name) + "%"
    result = plt.plot(x, y, linewidth=lw, label = "")
    return label, result

def plot_scatter_mr(data, metric_name, filter_name,  marker, color, size = 2, alpha = 1, **kwargs):
    data_size = data.loc[data['mr'] == filter_name]
    x = data_size['el_volume']
    y = data_size[metric_name]
    tr_count = data_size.tail(1)['ts_count']
    filter_name_str = str(filter_name)
    label = ur'MR '  + filter_name_str +  ur'%' 
    return label, plt.scatter(x, y, label = "",  c = color, s = size, marker =  marker, alpha = alpha)


def set_axis_style(ax, labels, name):
    ax.get_xaxis().set_tick_params(direction='in')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel(name)
    

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value
    
def plot_violin_mr(data, metric_name, filter_name,  ax, **kwargs):
    violin_data = []
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    labels_x = []
    
    for k in range(5,35, 5):
        data_size = data.loc[data['mr'] == k]
        y = data_size[metric_name]
        violin_data.append(y)
        
    n = 6
    for i in range(n):
        filter_label = 'size' + str(i)
        row = data.loc[data['roi_volume_label'] == filter_label]
        item_label = row['el_volume']
        labels_x.append(item_label)
        
    parts = ax.violinplot(violin_data, vert=True, showmeans=False, showmedians=False,
        showextrema=False)
    
    ctr = 0
    for pc in parts['bodies']:
        pc.set_facecolor(colors[ctr])
        pc.set_edgecolor('black')
        pc.set_lw(1)
        pc.set_alpha(0.8)
        ctr = ctr + 1
    
    violin_data_as_arr = np.array(violin_data)   
    quartile1, medians, quartile3 = np.percentile(violin_data_as_arr, [25, 50, 75], axis=1)
    whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(violin_data_as_arr, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    
    axis_label = 'x'
    #set_axis_style(ax, labels_x, axis_label)
    
    return labels_x, parts

def plot_scatter(data, metric_name, filter_name,  marker, color, size = 2, alpha = 1, **kwargs):
    data_size = data.loc[data['roi_volume_label'] == filter_name]
    x = data_size['mr']
    y = data_size[metric_name]
    roi_volume = data_size.tail(1)['el_volume']
    label_size = 'ROI Volume: ' + group(roi_volume)
    return plt.scatter(x, y, label = "",  c = color, s = size, marker =  marker, alpha = alpha)

def draw_4D_roi_volume():
   
    file_path = os.path.join(roi_volume_folder, data_by_volume4D)
    data = iot.read_data_structural(file_path)
    data_size0 = data.loc[data['roi_volume_label'] == 'size0']
    fig = texfig.figure(frameon = False)
    
    print data_size0
   
    title = "Effect of ROI Volume on 4D fMRI Completion by Missing Values Rate \n  Structural Missing Value Pattern";
    fig.suptitle(title, fontsize=8)
    
    grid_rows = 1
    grid_cols = 2
    
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.05, wspace=0.05)
    
    lw = 2
    tsc_ax = fig.add_subplot(grid[0, 0])
    tsc_ax.set_xlabel("Missing Values %")
    tsc_x = data_size0['mr']
    tsc_y = data_size0['tsc_cost']
    
    roi_volume = data_size0.tail(1)['el_volume']
    label_size0 = 'ROI Volume: ' + group(roi_volume)
    
    print label_size0
    plt.plot(tsc_x, tsc_y, linewidth=lw,
             label= label_size0)
    
    plt.legend(loc='upper right', fontsize=6)
    solution_id = 'structural_by_roi_volume'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig(file_path, pad = 2)
    
    pass

def draw_4D_by_roi_volume():
    file_path = os.path.join(roi_volume_folder, data_by_mr_by_volume4D)
    data = iot.read_data_structural(file_path)

    n = 6
      
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of ROI Size on 4D fMRI Completion by Missing Values Rate  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values (%)')
    ax0.tick_params(direction="in")
    plt.xticks(range(5, 35, 5), fontsize=8)
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.6)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    for i in range(n):
        filter_name = 'size' + str(i)
        metric_name = 'tsc_cost'
        plt.sca(ax0)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plot_scatter(data, metric_name, filter_name, markers[i], colors[i], 25, alpha = 0.5)
        
        top_legend_handles.append(lines[0])
        top_labels.append(label)
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(ur'Missing Values (%)')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.6)
    format_axes(ax1)
    
    for k in range(n):
        filter_name = 'size' + str(k)
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        plot_scatter(data, metric_name, filter_name, markers[k], colors[k], 25, alpha = 0.5)
    
    lgd = create_legend_on_bottom(3, top_legend_handles, top_labels, "")
    
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$4$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
    solution_id = 'structural_effect_roi_volume_by_mr_4D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,)) 
    
def draw_4D_by_roi_volume_and_mr():
    file_path = os.path.join(roi_volume_folder, data_by_volume4D)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)

    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of Missing Values Rate on 4D fMRI Completion by ROI Size  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(r'Tensor Completion Score')
    ax0.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    
    ax0.tick_params(direction="in")
    
    format_axes(ax0)
    
    x_ticks = []
    n = 6
    for i in range(n):
        filter_label = 'size' + str(i)
        row = data.loc[data['roi_volume_label'] == filter_label]
        item_label = row['el_volume']
        x_ticks.append(item_label)
        
    x_ticks = [2346,2681,3016,3351,3686,4021]
        
    plt.xticks(x_ticks, fontsize=8)
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.6)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    ctr = 0
    for i in range(5,35, 5):
        filter_name = i
        metric_name = 'tsc_cost'
        plt.sca(ax0)
    
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        
        top_legend_handles4D_mr.append(scatter)
        top_labels4D_mr.append(label)
        ctr = ctr + 1
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.6)
    format_axes(ax1)
    
    ctr = 0
    for k in range(5,35, 5):
        filter_name = k
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        ctr = ctr + 1
    
    lgd = create_legend_on_bottom(3, top_legend_handles4D_mr, top_labels4D_mr, "")
    
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$4$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
            
    solution_id = 'structural_effect_mr_by_roi_volume4D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    

def draw_3D_by_roi_volume_and_mr():
    file_path = os.path.join(roi_volume_folder, data_by_volume2D)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)

    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of Missing Values Rate on 4D fMRI Completion by ROI Size  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(r'Tensor Completion Score')
    ax0.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    
    ax0.tick_params(direction="in")
    
    format_axes(ax0)
    
    x_ticks = []
    n = 6
    for i in range(n):
        filter_label = 'size' + str(i)
        row = data.loc[data['roi_volume_label'] == filter_label]
        item_label = row['el_volume']
        x_ticks.append(item_label)
        
    x_ticks = [2346,2681,3016,3351,3686,4021]
        
    plt.xticks(x_ticks, fontsize=8)
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.6)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    ctr = 0
    for i in range(5,35, 5):
        filter_name = i
        metric_name = 'tsc_cost'
        plt.sca(ax0)
    
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        
        top_legend_handles3D_mr.append(scatter)
        top_labels3D_mr.append(label)
        ctr = ctr + 1
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.6)
    format_axes(ax1)
    
    ctr = 0
    for k in range(5,35, 5):
        filter_name = k
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        ctr = ctr + 1
    
    lgd = create_legend_on_bottom(3, top_legend_handles4D_mr, top_labels4D_mr, "")
    
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$3$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
            
    solution_id = 'structural_effect_mr_by_roi_volume3D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
    
def draw_3D_by_roi_volume():
    file_path = os.path.join(roi_volume_folder, data_by_mr_by_volume3D)
    data = iot.read_data_structural(file_path)

    n = 6
      
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of ROI Size on 3D fMRI Completion by Missing Values Rate  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values (%)')
    ax0.tick_params(direction="in")
    plt.xticks(range(5, 35, 5), fontsize=8)
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.7)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    for i in range(n):
        filter_name = 'size' + str(i)
        metric_name = 'tsc_cost'
        plt.sca(ax0)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plot_scatter(data, metric_name, filter_name, markers[i], colors[i], 25, alpha = 0.5)
        
        top_legend_handles3D.append(lines[0])
        top_labels3D.append(label)
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(ur'Missing Values (%)')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.7)
    format_axes(ax1)
    
    for k in range(n):
        filter_name = 'size' + str(k)
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        plot_scatter(data, metric_name, filter_name, markers[k], colors[k], 25, alpha = 0.5)
    
    lgd = create_legend_on_bottom(3, top_legend_handles3D, top_labels3D, "")
    
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$3$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
    solution_id = 'structural_effect_roi_volume_by_mr_3D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,)) 
    

def draw_2D_by_roi_volume():
    file_path = os.path.join(roi_volume_folder, data_by_mr_by_volume2D)
    data = iot.read_data_structural(file_path)

    n = 6
      
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of ROI Size on 2D fMRI Completion by Missing Values Rate  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values (%)')
    ax0.tick_params(direction="in")
    plt.xticks(range(5, 35, 5), fontsize=8)
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.7)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    for i in range(n):
        filter_name = 'size' + str(i)
        metric_name = 'tsc_cost'
        plt.sca(ax0)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plot_scatter(data, metric_name, filter_name, markers[i], colors[i], 25, alpha = 0.5)
        
        top_legend_handles2D.append(lines[0])
        top_labels2D.append(label)
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(ur'Missing Values (%)')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.7)
    format_axes(ax1)
    
    for k in range(n):
        filter_name = 'size' + str(k)
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        plot_scatter(data, metric_name, filter_name, markers[k], colors[k], 25, alpha = 0.5)
    
    lgd = create_legend_on_bottom(3, top_legend_handles2D, top_labels2D, "")
    
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$2$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
    solution_id = 'structural_effect_roi_volume_by_mr_2D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))         

def draw_2D_by_roi_volume_and_mr():
    file_path = os.path.join(roi_volume_folder, data_by_volume2D)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)

    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of Missing Values Rate on 2D fMRI Completion by ROI Size  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(r'Tensor Completion Score')
    ax0.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    
    ax0.tick_params(direction="in")
    
    format_axes(ax0)
    
    x_ticks = []
    n = 6
    for i in range(n):
        filter_label = 'size' + str(i)
        row = data.loc[data['roi_volume_label'] == filter_label]
        item_label = row['el_volume']
        x_ticks.append(item_label)
        
    x_ticks = [2346,2681,3016,3351,3686,4021]
        
    plt.xticks(x_ticks, fontsize=8)
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.6)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    ctr = 0
    for i in range(5,35, 5):
        filter_name = i
        metric_name = 'tsc_cost'
        plt.sca(ax0)
    
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.9)
        
        top_legend_handles2D_mr.append(scatter)
        top_labels2D_mr.append(label)
        ctr = ctr + 1
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.6)
    format_axes(ax1)
    
    ctr = 0
    for k in range(5,35, 5):
        filter_name = k
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        ctr = ctr + 1
    
    lgd = create_legend_on_bottom(3, top_legend_handles2D_mr, top_labels2D_mr, "")
    
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$2$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
            
    solution_id = 'structural_effect_mr_by_roi_volume2D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))

@ticker.FuncFormatter
def major_formatter(x, pos):
    return mf.format_number(x, fmt='%1.2e')

@ticker.FuncFormatter
def sci_formatter(x, pos):
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_scientific(True)
    return xfmt


@ticker.FuncFormatter
def sci_formatter_without_offset(x, pos):
    xfmt = ScalarFormatter(useMathText=True, useOffset=False)
    xfmt.set_scientific(True)
    xfmt.set_useOffset(10000000)
    return xfmt

@ticker.FuncFormatter
def major_formatter2(x, pos):
    return mf.format_number(x, fmt='%1.4e')

def tensor_order_by_tsc(tensor_dims, mr, roi_size, log_scale = False):
    file_path = os.path.join(roi_volume_folder,  data_all_by_volume)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)
    
    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    ur'Structural Missing Value Pattern'
    title = ur'Effect of Tensor Dimensionality on fMRI Completion by Effective ROI Size'
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.1, left = 0.2, right = 0.95, bottom = 0.15)
    
    ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Effective ROI Size ' + ur'($voxel^3 \times timepoints$)')
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    ax0.yaxis.set_major_formatter(major_formatter)
    
    if log_scale:     
        #ax0.xscale('log')
        plt.yscale('log')
        solution_id = 'effect_tensor_dimension_roi_size_' + str(mr)+ '_log'
    else:
        solution_id = 'effect_tensor_dimension_roi_size_' + str(mr)
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim1'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['roi_volume']
        y = row['tsc_cost']
        lw = 1
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
    extra_text = ur'for Missing Values Rate = ' + str(mr) + ur'%'
    lgd = create_legend_on_inside_upper_left(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
    
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
def tensor_order_by_tsc_mr(tensor_dims, roi_size, log_scale = False):
    file_path = os.path.join(roi_volume_folder,  data_all_by_volume2)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)
    
    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    ur'Structural Missing Value Pattern'
    title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Values Rate'
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.1, left = 0.2, right = 0.95, bottom = 0.15)
    
    ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values ' + ur'(%)')
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    ax0.yaxis.set_major_formatter(major_formatter)
    plt.xticks(range(5, 35, 5), fontsize=8)
    ax0.set_xlim(5, 35)
    
    x_ticks_map = {}
    
    x_ticks_map['size0'] = 2346
    x_ticks_map['size1'] = 2681
    x_ticks_map['size2'] = 3016
    x_ticks_map['size3'] = 3351
    x_ticks_map['size4'] = 3686
    x_ticks_map['size5'] = 4021
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    ctr = 0
    
    for i in tensor_dims:
        
        
        plt.sca(ax0)
        subset = data.loc[data['tensor_dim1'] == i]
        row = subset.loc[subset['roi_volume_label'] == roi_size]
     
        x = row['mr']
        print x
        y = row['tsc_cost']
        
        roi_volume = row.tail(1)
        volume = roi_volume['el_volume']
        lw = 1
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
        extra_text = ur'for ROI Volume = ' + str(x_ticks_map[roi_size])
    
    lgd = create_legend_on_inside_upper_left(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
    solution_id = 'effect_tensor_dimension_by_mr_roi_size_' + str(roi_size)
    
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
def tensor_order_by_tsc_all(tensor_dims, mr, roi_size, log_scale = False):
    file_path = os.path.join(roi_volume_folder,  data_all_by_volume)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)
    
    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    ur'Structural Missing Value Pattern'
    title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Values Rate'
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.2)
    
    ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values ' + ur'(%)')
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    ax0.yaxis.set_major_formatter(major_formatter)
    
    if log_scale:     
        #ax0.xscale('log')
        plt.yscale('log')
        solution_id = 'effect_tensor_dimension_roi_size_' + str(mr)+ '_log'
    else:
        solution_id = 'effect_tensor_dimension_roi_size_' + str(mr)
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim1'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['roi_volume']
        y = row['tsc_cost']
        lw = 1.5
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
    extra_text = ur'for Missing Values Rate = ' + str(mr) + ur'%'
    lgd = create_legend_on_inside_upper_left_inset(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text,  0.01, 0.89)
    
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))

def formatted_percentage(value, digits):
    format_str = "{:." + str(digits) + "%}"
    
    return format_str.format(value) 
  
def objective_value_by_iteration_by_snr(snr, file_path, results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Objective Value by Iteration  Number and Noise Distribution'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    #ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Objective Value')
    ax0.set_xlabel(ur'Iteration Number')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax0.set_ylim([0.001, 0.004])
    
    if log_scale:     
        #ax0.xscale('log')
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'objective_value_by_noise_distribution_fixed_snr_' + str(snr)+ '_log_scale'
    else:
        solution_id = 'objective_value_by_noise_distribution_fixed_snr_' + str(snr)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    
    ctr = 0
    for i in noise_type:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['noise_type'] == filter_label]
        row = subset.loc[subset['snr'] == snr]
        perc = row['corruption_error']
        #perc_str = formatted_percentage(perc, 2)
        
        x = row['k']
        y = row['solution_grad']

        print ("min: " + str(np.min(y)) + "; max: " + str(np.max(y)))
        lw = 1
        label = str(i).capitalize()
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
    
    perc_values = pd.Series(perc).values[0]

    perc_fmt = formatted_percentage(perc_values, 2)
    extra_text = ur'for SNR = ' + str(snr) + ur', N = ' + str(perc_fmt)
    lgd = create_legend_on_inside_upper_right(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))

def low_rank_rse_by_iteration_by_snr(snr, file_path, results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Low-Rank Relative Error by Iteration  Number and by Noise Distribution'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    #ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'$\frac{\Vert L_{k+1} - L_{k}\Vert_F}{\Vert L_{k} \Vert_F}$')
    ax0.set_xlabel(ur'Iteration Number')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax0.set_ylim([0.001, 0.004])
    
    if log_scale:     
        #ax0.xscale('log')
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'low_rank_rse_by_noise_distribution_fixed_snr_' + str(snr)+ '_log_scale'
    else:
        solution_id = 'low_rank_rse_value_by_noise_distribution_fixed_snr_' + str(snr)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    
    ctr = 0
    for i in noise_type:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['noise_type'] == filter_label]
        row = subset.loc[subset['snr'] == snr]
        perc = row['corruption_error']
        #perc_str = formatted_percentage(perc, 2)
        
        x = row['k']
        y = row['low_rank_rse']

        print ("min: " + str(np.min(y)) + "; max: " + str(np.max(y)))
        lw = 1
        label = str(i).capitalize()
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
    
    perc_values = pd.Series(perc).values[0]

    perc_fmt = formatted_percentage(perc_values, 2)
    extra_text = ur'for SNR = ' + str(snr) + ur', N = ' + str(perc_fmt)
    lgd = create_legend_on_inside_upper_right(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def sparse_rank_rse_by_iteration_by_snr(snr, file_path, results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Sparse Relative Error by Iteration  Number and by Noise Distribution'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    #ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'$\frac{\Vert S_{k+1} - S_{k}\Vert_F}{\Vert S_{k} \Vert_F}$')
    ax0.set_xlabel(ur'Iteration Number')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax0.set_ylim([0.001, 0.004])
    
    if log_scale:     
        #ax0.xscale('log')
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'sparse_rse_by_noise_distribution_fixed_snr_' + str(snr)+ '_log_scale'
    else:
        solution_id = 'sparse_rse_rse_value_by_noise_distribution_fixed_snr_' + str(snr)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    
    ctr = 0
    for i in noise_type:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['noise_type'] == filter_label]
        row = subset.loc[subset['snr'] == snr]
        perc = row['corruption_error']
        #perc_str = formatted_percentage(perc, 2)
        
        x = row['k']
        y = row['sparse_rank_rse']

        print ("min: " + str(np.min(y)) + "; max: " + str(np.max(y)))
        lw = 1
        label = str(i).capitalize()
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
    
    perc_values = pd.Series(perc).values[0]

    perc_fmt = formatted_percentage(perc_values, 2)
    extra_text = ur'for SNR = ' + str(snr) + ur', N = ' + str(perc_fmt)
    lgd = create_legend_on_inside_upper_right(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def objective_value_by_iteration_by_noise_type(noise_type, file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Objective Value by Iteration  Number and Signal-To-Noise Ratio'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    #ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Objective Value')
    ax0.set_xlabel(ur'Iteration Number')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax0.set_ylim([0.001, 0.004])
    
    if log_scale:     
        #ax0.xscale('log')
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'objective_value_by_snr_fixed_noise_distribution_' + str(noise_type)+ '_log_scale'
    else:
        solution_id = 'objective_value_by_snr_fixed_noise_distribution_' + str(noise_type)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    
    ctr = 0
    for i in snr_level:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['snr'] == filter_label]
        row = subset.loc[subset['noise_type'] == noise_type]
        perc = row['corruption_error']
        #perc_str = formatted_percentage(perc, 2)
        perc_values = pd.Series(perc).values[0]
        
        x = row['k']
        y = row['solution_grad']

        perc_fmt = formatted_percentage(perc_values, 2)
        print ("min: " + str(np.min(y)) + "; max: " + str(np.max(y)))
        lw = 1
        label = ur'SNR = ' + str(i)  + ur', N = ' + str(perc_fmt)
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], marker =  markers[ctr], s = 15, alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
    print "CTR = " + str(ctr)

    extra_text = ur'Noise Distribution = ' + str(noise_type.capitalize())
    extra_text1 = ""
    lgd = create_legend_on_inside_upper_right2(2, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))

def low_rank_error_by_iteration_by_noise_type(noise_type, file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Low-Rank Relative Error by Iteration  Number Iteration Number and Signal-To-Noise Ratio'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    #ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'$\frac{\Vert L_{k+1} - L_{k}\Vert_F}{\Vert L_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'Iteration Number')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax0.set_ylim([0.001, 0.004])
    
    if log_scale:     
        #ax0.xscale('log')
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'low_rank_rse_by_snr_fixed_noise_distribution_' + str(noise_type)+ '_log_scale'
    else:
        solution_id = 'low_rank_rse_by_snr_fixed_noise_distribution_' + str(noise_type)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    
    ctr = 0
    for i in snr_level:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['snr'] == filter_label]
        row = subset.loc[subset['noise_type'] == noise_type]
        perc = row['corruption_error']
        #perc_str = formatted_percentage(perc, 2)
        perc_values = pd.Series(perc).values[0]
        
        x = row['k']
        y = row['low_rank_rse']

        perc_fmt = formatted_percentage(perc_values, 2)
        print ("min: " + str(np.min(y)) + "; max: " + str(np.max(y)))
        lw = 1
        label = ur'SNR = ' + str(i)  + ur', N = ' + str(perc_fmt)
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], marker =  markers[ctr], s = 15, alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
    print "CTR = " + str(ctr)

    extra_text = ur'Noise Distribution = ' + str(noise_type.capitalize())
    extra_text1 = ""
    lgd = create_legend_on_inside_upper_right2(2, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def sparse_rank_error_by_iteration_by_noise_type(noise_type, file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Sparse Relative Error by Iteration  Number Iteration Number and Signal-To-Noise Ratio'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    #ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'$\frac{\Vert S_{k+1} - S_{k}\Vert_F}{\Vert S_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'Iteration Number')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax0.set_ylim([0.001, 0.004])
    
    if log_scale:     
        #ax0.xscale('log')
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'sparse_rse_by_snr_fixed_noise_distribution_' + str(noise_type)+ '_log_scale'
    else:
        solution_id = 'sparse_rse_rse_by_snr_fixed_noise_distribution_' + str(noise_type)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    
    ctr = 0
    for i in snr_level:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['snr'] == filter_label]
        row = subset.loc[subset['noise_type'] == noise_type]
        perc = row['corruption_error']
        #perc_str = formatted_percentage(perc, 2)
        perc_values = pd.Series(perc).values[0]
        
        x = row['k']
        y = row['sparse_rank_rse']

        perc_fmt = formatted_percentage(perc_values, 2)
        print ("min: " + str(np.min(y)) + "; max: " + str(np.max(y)))
        lw = 1
        label = ur'SNR = ' + str(i)  + ur', N = ' + str(perc_fmt)
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], marker =  markers[ctr], s = 15, alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
    print "CTR = " + str(ctr)

    extra_text = ur'Noise Distribution = ' + str(noise_type.capitalize())
    extra_text1 = ""
    lgd = create_legend_on_inside_upper_right2(2, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def draw_structural_results():
    
    figures_path1="/work/project/cmsc655/results/out/generated/by_noise_type"
    solution_path="/work/project/cmsc655/results/alternate_min/alternate_minimization_solution.csv"

    for s in snr_level:
        objective_value_by_iteration_by_snr(s, solution_path,figures_path1, title=False, log_scale=False)
        objective_value_by_iteration_by_snr(s, solution_path,figures_path1, title=True, log_scale=False)
    
    figures_path2="/work/project/cmsc655/results/out/generated/by_snr"
    for n in noise_type:
        objective_value_by_iteration_by_noise_type(n, solution_path,figures_path2, title=False, log_scale=False)
        objective_value_by_iteration_by_noise_type(n, solution_path,figures_path2, title=True, log_scale=False)
    
    figures_path3 = "/work/project/cmsc655/results/out/generated/low_rank_rse/by_snr"
    
    for s in snr_level:
        low_rank_rse_by_iteration_by_snr(s, solution_path,figures_path3, title=False, log_scale=False)
        low_rank_rse_by_iteration_by_snr(s, solution_path,figures_path3, title=True, log_scale=False)

    figures_path4 = "/work/project/cmsc655/results/out/generated/low_rank_rse/by_noise_type"
        
    for n in noise_type:
        low_rank_error_by_iteration_by_noise_type(n, solution_path,figures_path4, title=False, log_scale=False)
        low_rank_error_by_iteration_by_noise_type(n, solution_path,figures_path4, title=True, log_scale=False)
        
    figures_path5 = "/work/project/cmsc655/results/out/generated/sparse_rank_rse/by_snr"
    
    for s in snr_level:
        sparse_rank_rse_by_iteration_by_snr(s, solution_path,figures_path5, title=False, log_scale=False)
        sparse_rank_rse_by_iteration_by_snr(s, solution_path,figures_path5, title=True, log_scale=False)

     
    figures_path6 = "/work/project/cmsc655/results/out/generated/sparse_rank_rse/by_noise_type"
    
    for n in noise_type:
        low_rank_error_by_iteration_by_noise_type(n, solution_path,figures_path6, title=False, log_scale=False)
        low_rank_error_by_iteration_by_noise_type(n, solution_path,figures_path6, title=True, log_scale=False)
    
    #low_rank_error_by_iteration_by_noise_type
    
    #draw_4D_by_roi_volume()
    #draw_4D_by_roi_volume_and_mr()
    
    #draw_3D_by_roi_volume()
    #draw_3D_by_roi_volume_and_mr()
    #draw_2D_by_roi_volume()
    #draw_2D_by_roi_volume_and_mr()

def objective_value_llsrt_vs_matrix(snr, file_path, results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Objective Value of Tensor Decomposition vs Matrix Decomposition'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.1, right = 0.95, bottom = 0.15)
    
    #ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Objective Value')
    ax0.set_xlabel(ur'Iteration Number')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax0.set_ylim([0.001, 0.004])
    
    if log_scale:     
        #ax0.xscale('log')
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'objective_value_lrstf_vs_matrix_fixed_snr_' + str(snr)+ '_log_scale'
    else:
        solution_id = 'objective_value_lrstf_vs_matrix_fixed_snr_' + str(snr)
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    
    ctr = 0
    
    nn = []
    nn.append('richian')
    
    for i in nn:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['noise_type'] == filter_label]
        row = subset.loc[subset['snr'] == snr]
        perc = row['corruption_error']
        #perc_str = formatted_percentage(perc, 2)
        
        x = row['k']
        y = row['solution_grad']
              
        lw = 1
        label = "Low-Rank-Sparse Tensor Decomposition"
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        
        y1 = row['sol_error']
        
        label = "Randomized Low-Rank-Sparse Matrix Decomposition"
        lines = plt.plot(x, y1, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y1, label = "",  c = colors[ctr+1], s = 25, marker =  markers[ctr+1], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        
        ctr = ctr + 1
    
    #perc_values = pd.Series(perc).values[0]

    #perc_fmt = formatted_percentage(perc_values, 2)
    extra_text = ur'for SNR = ' + str(snr) + ur', N = ' + str('4.2%')
    lgd = create_legend_on_inside_upper_right3(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))

def low_rank_by_snr(file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Relative Solution Error by Signal-To-Noise Ratio'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    ax0.set_ylabel(ur'$\frac{\Vert X_{k+1} - X_{k}\Vert_F}{\Vert X_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'SNR, $Db$')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    
    if log_scale:     
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'rel_solution_cost_by_snr_' + '_log_scale'
    else:
        solution_id = 'rel_solution_cost_by_snr'
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
 
    ctr = 0
    for i in noise_type:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['noise_type'] == filter_label]
       
        perc = subset['corruption_error']
       
        
        x = subset['snr']
        y = subset['rel_solution_cost']
                   
        noise_type_subset = subset['noise_type']
        noise_type_v = pd.Series(noise_type_subset).values[0]
        lw = 1
        label = ur'$P(N|X)$' + ur' = ' + str(noise_type_v.capitalize())
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], marker =  markers[ctr], s = 15, alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)

        
        ctr = ctr + 1
        
    print "CTR = " + str(ctr)

    extra_text = ""
    lgd = create_legend_on_inside_upper_right2(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def low_rank_by_snr_noise_type(noise_t, file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Relative Solution Error by Signal-To-Noise Ratio'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    ax0.set_ylabel(ur'$\frac{\Vert X_{k+1} - X_{k}\Vert_F}{\Vert X_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'SNR, $Db$')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax0.set_ylim([0.001, 0.004])
    
    if log_scale:     
       
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'rel_solution_cost_by_snr_' + noise_t + '_log_scale'
    else:
        solution_id = 'rel_solution_cost_snr_' + noise_t 
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    tick_labels = []
    
    subset = data.loc[data['noise_type'] == noise_t]
    x = subset['snr']
    y = subset['rel_solution_cost']
    
    perc = subset['corruption_error']
    #perc_str = formatted_percentage(perc, 2)
    perc_values = pd.Series(perc).values
        
    lw = 1
    label = ur'Noise Distribution = ' + str(noise_t.capitalize())
    lines = plt.plot(x, y, linewidth=lw, label="")
   
    scatter = plt.scatter(x, y, label = "",  c = colors[1], marker =  markers[1], s = 15, alpha = 1)
    top_legend_tensor_dim.append(scatter)
    top_labelstensor_dim.append(label)

    extra_text = ""
    
    lgd = create_legend_on_inside_lower_right4(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))

def low_rank_by_corr_level_noise_type(noise_t, file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Relative Solution Error by Noise Corruption'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    ax0.set_ylabel(ur'$\frac{\Vert X_{k+1} - X_{k}\Vert_F}{\Vert X_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'$N$')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    
    if log_scale:     
       
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = ' rel_solution_cost_by_corr_level_' + noise_t + '_log_scale'
    else:
        solution_id = 'rel_solution_cost_by_corr_level_' + noise_t 
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        ax0.xaxis.set_major_formatter(major_perc_formatter)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
          
    subset = data.loc[data['noise_type'] == noise_t]
    x = subset['corruption_error']
    y = subset['rel_solution_cost']
        
    lw = 1
    label = ur'$P(N|X)$' + ur' = ' + str(noise_t.capitalize())
    lines = plt.plot(x, y, linewidth=lw, label="")
   
    scatter = plt.scatter(x, y, label = "",  c = colors[1], marker =  markers[1], s = 15, alpha = 1)
            
    top_legend_tensor_dim.append(scatter)
    top_labelstensor_dim.append(label)

    extra_text = ""
    
    lgd = create_legend_on_inside_lower_right4(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def low_rank_by_corr_level_noise_type2(noise_t, file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Relative Low-Rank Error by Noise Corruption'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    ax0.set_ylabel(ur'$\frac{\Vert L_{k+1} - L_{k}\Vert_F}{\Vert L_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'$N$')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    
    if log_scale:     
       
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'low_rank_solution_cost_by_corr_level_' + noise_t + '_log_scale'
    else:
        solution_id = 'low_rank_solution_cost_by_corr_level_' + noise_t 
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        ax0.xaxis.set_major_formatter(major_perc_formatter)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
          
    subset = data.loc[data['noise_type'] == noise_t]
    x = subset['snr']
    y = subset['low_rank_rse']
        
    lw = 1
    label = ur'Noise Distribution = ' + str(noise_t.capitalize())
    
    lines = plt.plot(x, y, linewidth=lw, label="")
   
    scatter = plt.scatter(x, y, label = "",  c = colors[1], marker =  markers[1], s = 15, alpha = 1)
            
    top_legend_tensor_dim.append(scatter)
    top_labelstensor_dim.append(label)

    extra_text = ""
    
    lgd = create_legend_on_inside_lower_right4(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def sparse_rank_by_corr_level_noise_type2(noise_t, file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Relative Sparse-Rank Error by Noise Corruption'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    ax0.set_ylabel(ur'$\frac{\Vert S_{k+1} - S_{k}\Vert_F}{\Vert S_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'$N$')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    
    if log_scale:     
       
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'sparse_rank_solution_cost_by_corr_level_' + noise_t + '_log_scale'
    else:
        solution_id = 'sparse_solution_cost_by_corr_level_' + noise_t 
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        ax0.xaxis.set_major_formatter(major_perc_formatter)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
          
    subset = data.loc[data['noise_type'] == noise_t]
    x = subset['snr']
    y = subset['sparse_rank_rse']
        
    lw = 1
    label = ur'Noise Distribution = ' + str(noise_t.capitalize())
    
    lines = plt.plot(x, y, linewidth=lw, label="")
   
    scatter = plt.scatter(x, y, label = "",  c = colors[1], marker =  markers[1], s = 15, alpha = 1)
            
    top_legend_tensor_dim.append(scatter)
    top_labelstensor_dim.append(label)

    extra_text = ""
    
    lgd = create_legend_on_inside_lower_right4(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def low_rank_by_corr_perc(file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Relative Solution Error by Noise Corruption Level'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    ax0.set_ylabel(ur'$\frac{\Vert X_{k+1} - X_{k}\Vert_F}{\Vert X_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'$N$')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    
    if log_scale:     
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'rel_solution_cost_by_corr_level' + '_log_scale'
    else:
        solution_id = 'rel_solution_cost_by_corr'
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
 
    ctr = 0
    for i in noise_type:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['noise_type'] == filter_label]
                 
        
        x = subset['corruption_error']
        y = subset['rel_solution_cost']
                   
        noise_type_subset = subset['noise_type']
        noise_type_v = pd.Series(noise_type_subset).values[0]
        lw = 1
        label = ur'$P(N|X)$' + ur' = ' + str(noise_type_v.capitalize())
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], marker =  markers[ctr], s = 15, alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)

        
        ctr = ctr + 1
        
    print "CTR = " + str(ctr)

    extra_text = ""
    lgd = create_legend_on_inside_upper_right2(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
    
def low_rank_by_corr_perc2(file_path,results_folder, title=False, log_scale=False):

    print ("Data Path:" + str(file_path) + "; Results Folder: " + str(results_folder))
    data = iot.read_data_by_path(file_path)
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols)
    
    if title:
        title = ur'Low-rank Solution Error by Noise Corruption Level'
        fig.suptitle(title, fontsize=8, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.15)
    
    ax0.set_ylabel(ur'$\frac{\Vert L_{k+1} - L_{k}\Vert_F}{\Vert L_{k} \Vert_F}$', fontsize=4)
    ax0.set_xlabel(ur'$N$')
        
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(7))
    
    if log_scale:     
        ax0.set_yscale('log')
        ax0.yaxis.set_major_formatter(CustomTicker())
        solution_id = 'low_rank_solution_cost_by_corr_level' + '_log_scale'
    else:
        solution_id = 'low_rank_solution_cost_by_corr_level'
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=5)
        
    if title == False:
        solution_id = solution_id + "_" + "no_title"
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append("v")
    markers.append("*")
    markers.append("v")
    markers.append("d")    
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
 
    ctr = 0
    for i in noise_type:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['noise_type'] == filter_label]
                 
        
        x = subset['corruption_error']
        y = subset['low_rank_rse']
                   
        noise_type_subset = subset['noise_type']
        noise_type_v = pd.Series(noise_type_subset).values[0]
        lw = 1
        label = ur'Noise Distribution ' + str(noise_type_v.capitalize())
        lines = plt.plot(x, y, linewidth=lw, label="")
        #lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], marker =  markers[ctr], s = 15, alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)

        
        ctr = ctr + 1
        
    print "CTR = " + str(ctr)

    extra_text = ""
    lgd = create_legend_on_inside_upper_right2(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
   
    
    solution_file_path = os.path.join(results_folder, solution_id)
    texfig.savefig_pub( solution_file_path, additional_artists=(lgd,))
            
def draw_sparse():
    
    solution_path="/work/project/cmsc655/results/alternate_min/alternate_minimization_solution.csv"
    
    figures_path5 = "/work/project/cmsc655/results/out/generated/sparse_rank_rse/by_snr"
    
    for s in snr_level:
        sparse_rank_rse_by_iteration_by_snr(s, solution_path,figures_path5, title=False, log_scale=False)
        sparse_rank_rse_by_iteration_by_snr(s, solution_path,figures_path5, title=True, log_scale=False)

     
    figures_path6 = "/work/project/cmsc655/results/out/generated/sparse_rank_rse/by_noise_type"
    
    for n in noise_type:
        sparse_rank_error_by_iteration_by_noise_type(n, solution_path,figures_path6, title=False, log_scale=False)
        sparse_rank_error_by_iteration_by_noise_type(n, solution_path,figures_path6, title=True, log_scale=False)

def draw_lsrtf_vs_godec():
    
    solution_path="/work/project/cmsc655/results/alternate_min/alternate_minimization_solution_and_godec.csv"
    figures_path7 = "/work/project/cmsc655/results/out/generated/godec"

    objective_value_llsrt_vs_matrix(25, solution_path, figures_path7, title=False, log_scale=False) 
    objective_value_llsrt_vs_matrix(25, solution_path, figures_path7, title=True, log_scale=False) 
    
def draw_low_rank_by_snr():
    
    solution_path="/work/pl/sch/analysis/cmsc655/completed-project/project/project-distribution/csv-solution-cost/alternate_minimization_solution_agg.csv"
    figures_path8 = "/work/pl/sch/analysis/cmsc655/completed-project/project/project-distribution/figures/gen_rep1/rse"
    
    #low_rank_by_snr(solution_path,figures_path8, title=False, log_scale=False)
    #low_rank_by_snr(solution_path,figures_path8, title=True, log_scale=False)
    
    #low_rank_by_snr_noise_type("richian", solution_path,figures_path8, title=False, log_scale=False)
    #low_rank_by_snr_noise_type("richian", solution_path,figures_path8, title=True, log_scale=False)
    
    #low_rank_by_corr_level_noise_type("richian", solution_path,figures_path8, title=True, log_scale=False)
    
    #low_rank_by_corr_perc(solution_path,figures_path8, title=False, log_scale=False)
    #low_rank_by_corr_perc(solution_path,figures_path8, title=True, log_scale=False)
    
    low_rank_by_corr_level_noise_type2("richian", solution_path,figures_path8, title=False, log_scale=False)
    low_rank_by_corr_level_noise_type2("richian", solution_path,figures_path8, title=True, log_scale=False)
    
    sparse_rank_by_corr_level_noise_type2("richian", solution_path,figures_path8, title=False, log_scale=False)
    sparse_rank_by_corr_level_noise_type2("richian", solution_path,figures_path8, title=True, log_scale=False)
    

if __name__ == "__main__":
    #draw_structural_results()
    #draw_sparse()
    #draw_lsrtf_vs_godec()
    draw_low_rank_by_snr()