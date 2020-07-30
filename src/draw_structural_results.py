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
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterSciNotation
import io_util
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D


config_loc = path.join('config')
config_filename = 'result.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

roi_volume_folder = config.get('results', 'roi_volume_folder')
structural_figures = config.get('results', 'structural_figures')

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

top_legend_tensor_mspv_rate = []
top_labelstensor_mspv_rate = []

star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)

tex_width = 5.78853 # in inches
tex_width = 4.78853 # in inches
tex_width = 3.5 # in inches

mr_map = {}
mr_map[0] = 5
mr_map[1] = 10
mr_map[2] = 15
mr_map[3] = 20
mr_map[4] = 25
mr_map[5] = 30

bg_legend_handle = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        
        return LogFormatterSciNotation.__call__(self,x, pos=None)
        #if x not in [1,10]:
        #    return LogFormatterSciNotation.__call__(self,x, pos=None)
        #else:
         #   return "{x:g}".format(x=x)

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

def create_legend_on_bottom_lower_left(ncol, handles, labels, extra_text=None):
    lgd = plt.legend(handles, labels, loc='lower left', ncol=ncol, bbox_to_anchor=(0.01, 0.02), fontsize="small", edgecolor='k')
    if extra_text:
        import matplotlib.offsetbox as offsetbox 
        txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
        box = lgd._legend_box
        box.get_children().append(txt) 
        box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_left(ncol, handles, labels, extra_text=None):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(0.01, 0.99), fontsize="small", edgecolor='k')
    if extra_text:
        import matplotlib.offsetbox as offsetbox 
        txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
        box = lgd._legend_box
        box.get_children().append(txt) 
        box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_best(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='best', ncol=ncol, bbox_to_anchor=(0.01, 0.98), fontsize="small", edgecolor='k')
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

def formatted_percentage(value, digits):
    format_str = "{:." + str(digits) + "%}"
    
    return format_str.format(value)


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

def plot_mr_spatial_volume(data, metric_name, filter_name,  log_scale=False, **kwargs):
    row = data.loc[data['mr'] == filter_name]
    y = row[metric_name]
    x = row['spatial_mr_rate_perc']
    lw = 1
    label = r'MR ' + str(filter_name) + "%"
    
    if log_scale:
        result = plt.semilogy(x, y, linewidth=lw, label = "")
    else:
        result = plt.plot(x, y, linewidth=lw, label = "")
    return label, result, row

def plot_scatter_mr_spatial_volume(data, metric_name, filter_name,  marker, color, size = 2, alpha = 1, **kwargs):
    data_size = data.loc[data['mr'] == filter_name]
    x = data_size['spatial_mr_rate_perc']
    y = data_size[metric_name]
    filter_name_str = str(filter_name)
    label = ur'MR '  + filter_name_str +  ur'%' 
    return label, plt.scatter(x, y, label = "",  c = color, s = size, marker =  marker, alpha = alpha)

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
    
def tensor_order_by_tsc_by_mr_spatial_volume(data_path, solution_path, tensor_dims, mr, title=False, log_scale = False):
    
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['solid']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    
    if title:
        title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Spatial Volume (%)'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.05, left = 0.1, right = 0.95, bottom = 0.15)
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 6,
        }
    
    ax0.set_title(ur'Missing Timepoints Rate ' + ur' = ' + str(mr) + ur'%', fontdict=font_label)
    ax0.set_ylabel(ur'Tensor Completion Score', fontdict=font_label)
    ax0.set_xlabel(ur'Missing Spatial Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_cost_' + str(mr) + str("_no_title")
    if title:
        solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_cost_' + str(mr)
        
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_scientific(True)
    yfmt.set_powerlimits((-1,1))
    ax0.yaxis.set_major_formatter(yfmt)
    ax0.yaxis.set_tick_params(labelsize=6)
    ax0.xaxis.set_tick_params(labelsize=6)
    
    ymin = 2e-3
    ymax = 1e-1
    
    if mr == 5:
        ymin = 0
        ymax = 2.5e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.004))
        
        
    if mr == 10:
        ymin = 0
        ymax = 2.5e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.004))
        
    if mr == 15:
        ymin = 0
        ymax = 5.1e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.008))
        
    
    if mr == 20 or mr == 25:
        ymin = 0
        ymax = 1.3e-1
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.02))
        
        
    
    xmin = 1.4
    xmax = 2.7
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['spatial_mr_rate_perc']
        y = row['tcs_cost']
        
        csv_rows.append(row)
     
        
        lw = 1
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
    
        
        extra_text = ur'for Missing Timepoints Rate = ' + str(mr) + ur'%'
        lgd = create_legend_on_inside_upper_left(1, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
    
    cur_ylim = ax0.get_ylim(); 
    #ax0.set_ylim([ymin,cur_ylim[1]])
    
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_cost'] = pd.Series(run_result['tcs_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
    solution_id_csv = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_cost_' + str(mr)
    solution_csv_path = os.path.join(solution_path, "csv")
    
    mrd.save_csv_by_path_adv(csv_result, solution_csv_path, solution_id_csv, index = False)

def tensor_order_by_tsc_by_mr_spatial_volume_log(data_path, solution_path, tensor_dims, mr, title=False, log_scale = False):
    
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['dotted']
    NUM_STYLES = len(LINE_STYLES)
    

    plt.clf()
    
    if title:
        layout = {'w_pad': 5, 'h_pad': 5
        }
        tex_width = 3.5
    else: 
        layout = {'w_pad': 0, 'h_pad': 0
        }
        tex_width = 3.5
        
    fig = texfig.figure(width=tex_width)
    fig.set_tight_layout(layout)
    
    ax0 = plt.gca()
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 8,
        }
    
    labelsize = 8
    
    if title:
        title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Ellipsoid Volume (%)'
        fig.suptitle(title, fontsize=6, fontweight='normal')

    
    ylabel = ur' Tensor Completion Score'
    if log_scale:
        ylabel = ur'Tensor Completion Score'
    
    ax0.set_ylabel(ylabel, fontdict=font_label)
    ax0.set_xlabel(ur'Missing Ellipsoid Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#ED0000FF", "#00468BFF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    format_axes(ax0)
    
    solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_cost_' + str(mr) + str("_no_title")
    if title:
        solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_cost_' + str(mr)
    
    if log_scale:     
        ax0.set_yscale('symlog', nonposy='clip', linthreshy=0.005)
        yfmt = CustomTicker()
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=labelsize)
        ax0.xaxis.set_tick_params(labelsize=labelsize)
        solution_id = solution_id  + '_log_scale'
        ymin = 10**-3
        ymax = 10**0
        ax0.set_ylim(bottom = ymin, top = ymax)   
    else:    
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=labelsize)
        ax0.xaxis.set_tick_params(labelsize=labelsize)
    
    
    ax0.tick_params(direction="in")
    
    xmin = 1.4
    xmax = 2.7
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    plt.sca(ax0)
    #plt.yscale('symlog', nonposy='clip', linthreshy=0.005)

    extra_text = ur'for Missing Timepoints Rate = ' + str(mr) + ur'%'
        
    all_dim = False
    num_legend_col = 1
    if 2 in tensor_dims:
        all_dim = True
        num_legend_col = 3
        extra_text = ""

    label_text = ur'for Missing Timepoints Rate = ' + str(mr) + ur'%'
    
    ctr = 0
    for i in tensor_dims:
        filter_label = i
        subset = data.loc[data['tensor_dim'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['spatial_mr_rate_perc']
        y = row['tcs_cost']
        
        csv_rows.append(row)
     
        
        lw = 1
        
        if all_dim:
            label = ur'$D$' + ur' = ' + str(i)
        else:
            label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        #lines = ax0.semilogy(x, y, linewidth=lw, label = "", c = colors[ctr], markersize = 5, marker =  markers[ctr], alpha = 1)
        lines = ax0.semilogy(x, y, linewidth=lw, label = "", alpha = 1)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = ax0.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
    
    xy_coord = (0.04, 0.5)
    if mr >= 20:
        lgd = create_legend_on_bottom_lower_left(num_legend_col, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
        xy_coord = (0.04, 0.23)
    else:
        lgd = create_legend_on_inside_upper_left(num_legend_col, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
        
    if all_dim:    
        ax0.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=7,
                bbox={'boxstyle': 'round', 'fc': 'w', 'ec':'k', 'color': 'black',
              'lw': 1, 'alpha': 0.8, 'zorder':-1})
    
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_cost'] = pd.Series(run_result['tcs_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
    solution_id_csv = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_cost_' + str(mr)
    solution_csv_path = os.path.join(solution_path, "csv")
    
    mrd.save_csv_by_path_adv(csv_result, solution_csv_path, solution_id_csv, index = False)
    
def tensor_order_by_tsc_z_by_mr_spatial_volume(data_path, solution_path, tensor_dims, mr, title=False, log_scale = False):
    
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_z_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['solid']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    
    if title:
        title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Spatial Volume (%)'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.05, left = 0.1, right = 0.95, bottom = 0.15)
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 6,
        }
    
    ax0.set_title(ur'Missing Timepoints Rate ' + ur' = ' + str(mr) + ur'%', fontdict=font_label)
    ax0.set_ylabel(ur'Tensor Completion Score $\vert Z_{score} \vert > 2$', fontdict=font_label)
    ax0.set_xlabel(ur'Missing Spatial Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_' + str(mr) + str("_no_title")
    if title:
        solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_' + str(mr)
        
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_scientific(True)
    yfmt.set_powerlimits((-1,1))
    ax0.yaxis.set_major_formatter(yfmt)
    ax0.yaxis.set_tick_params(labelsize=6)
    ax0.xaxis.set_tick_params(labelsize=6)
    
    ymin = -3e-3
    ymax = 1e-1
    
    if mr == 5:
        ymin = 0
        ymax = 2.5e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.004))
        
        
    if mr == 10:
        ymin = 0
        ymax = 2.5e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.004))
        
    if mr == 15:
        ymin = 0
        ymax = 5.1e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.008))
        
    
    if mr == 20 or mr == 25:
        ymin = 0
        ymax = 1.3e-1
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.02))
        
    
    xmin = 1.4
    xmax = 2.7
    
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['spatial_mr_rate_perc']
        y = row['tsc_z_cost']
     
        csv_rows.append(row)
        
        lw = 1
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
        extra_text = ur'for Missing Timepoints Rate = ' + str(mr) + ur'%'
        lgd = create_legend_on_inside_upper_left(1, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
    
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_z_cost'] = pd.Series(run_result['tsc_z_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
    solution_id_csv = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_' + str(mr)
    solution_csv_path = os.path.join(solution_path, "csv")
    
    mrd.save_csv_by_path_adv(csv_result, solution_csv_path, solution_id_csv, index = False)
    
def tensor_order_by_tsc_z_by_mr_spatial_volume_log(data_path, solution_path, tensor_dims, mr, title=False, log_scale = False):
    
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_z_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    if title:
        layout = {'w_pad': 5, 'h_pad': 5
        }
        tex_width = 3.5
    else: 
        layout = {'w_pad': 0, 'h_pad': 0
        }
        tex_width = 3.5
        
    fig = texfig.figure(width=tex_width)
    fig.set_tight_layout(layout)
    
    ax0 = plt.gca()
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 8,
        }
    
    labelsize = 8
    
    if title:
        title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Ellipsoid Volume (%)'
        fig.suptitle(title, fontsize=6, fontweight='normal')
        #ax0.set_title(ur'Missing Timepoints Rate ' + ur' = ' + str(mr) + ur'%', fontdict=font_label)
        
    ylabel = ur'Tensor Completion Score $\vert Z_{score} \vert > 2$'
    if log_scale:
        ylabel = ur'$\mathrm{TCS}_{\vert \mathrm{Z-Score} \vert > 2}$'
    
    ax0.set_ylabel(ylabel, fontdict=font_label)
    ax0.set_xlabel(ur'Missing Ellipsoid Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#ED0000FF", "#00468BFF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_' + str(mr) + str("_no_title")
    if title:
        solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_' + str(mr)
        
    if log_scale:     
        ax0.set_yscale('symlog', nonposy='clip', linthreshy=0.005)
        yfmt = CustomTicker()
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=labelsize)
        ax0.xaxis.set_tick_params(labelsize=labelsize)
        solution_id = solution_id  + '_log_scale'
        ymin = 10**-3
        ymax = 10**0
        ax0.set_ylim(bottom = ymin, top = ymax)   
    else:    
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=labelsize)
        ax0.xaxis.set_tick_params(labelsize=labelsize)
    
    xmin = 1.4
    xmax = 2.7
    
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    plt.sca(ax0)
    #plt.yscale('symlog', nonposy='clip', linthreshy=0.005)
    
    extra_text = ur'for Missing Timepoints Rate = ' + str(mr) + ur'%'
        
    all_dim = False
    num_legend_col = 1
    if 2 in tensor_dims:
        all_dim = True
        num_legend_col = 3
        extra_text = ""

    label_text = ur'for Missing Timepoints Rate = ' + str(mr) + ur'%'
    
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['spatial_mr_rate_perc']
        y = row['tsc_z_cost']
     
        csv_rows.append(row)
        
        lw = 1
        
        if all_dim:
            label = ur'$D$' + ur' = ' + str(i)
        else:
            label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        #lines = ax0.semilogy(x, y, linewidth=lw, label = "", c = colors[ctr], markersize = 5, marker =  markers[ctr], alpha = 1)
        lines = ax0.semilogy(x, y, linewidth=lw, label = "", alpha = 1)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = ax0.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
    xy_coord = (0.04, 0.5)
    if mr >= 20:
        lgd = create_legend_on_bottom_lower_left(num_legend_col, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
        xy_coord = (0.04, 0.23)
    else:
        lgd = create_legend_on_inside_upper_left(num_legend_col, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
        
    if all_dim:    
        ax0.annotate(
                label_text, xy=xy_coord, xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                va='center', ha='left', color='black', size=6,fontsize=7,
                bbox={'boxstyle': 'round', 'fc': 'w', 'ec':'k', 'color': 'black',
              'lw': 1, 'alpha': 0.8, 'zorder':-1})

    
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_z_cost'] = pd.Series(run_result['tsc_z_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
    solution_id_csv = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_' + str(mr)
    solution_csv_path = os.path.join(solution_path, "csv")
    
    mrd.save_csv_by_path_adv(csv_result, solution_csv_path, solution_id_csv, index = False)
    
def tensor_order_by_tsc_z_by_mr_spatial_volume_range(data_path, solution_path, tensor_dims, mr, title=False, log_scale = False):
    
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_z_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['solid']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    
    if title:
        title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Spatial Volume (%)'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.05, left = 0.1, right = 0.95, bottom = 0.15)
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 6,
        }
    
    ax0.set_title(ur'Missing Timepoints Rate ' + ur' = ' + str(mr) + ur'%', fontdict=font_label)
    ax0.set_ylabel(ur'Tensor Completion Score $\vert Z_{score} \vert > 2$', fontdict=font_label)
    ax0.set_xlabel(ur'Missing Spatial Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_range' + str(mr) + str("_no_title")
    if title:
        solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_range' + str(mr)
        
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_scientific(True)
    yfmt.set_powerlimits((-1,1))
    ax0.yaxis.set_major_formatter(yfmt)
    ax0.yaxis.set_tick_params(labelsize=6)
    ax0.xaxis.set_tick_params(labelsize=6)
    
    ymin = -3e-3
    ymax = 1e-1
    
    if mr == 5:
        ymin = 0
        ymax = 2.5e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.004))
        
        
    if mr == 10:
        ymin = 0
        ymax = 2.5e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.004))
        
    if mr == 15:
        ymin = 0
        ymax = 5.1e-2
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.008))
        
    
    if mr == 20 or mr == 25:
        ymin = 0
        ymax = 1.3e-1
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(np.arange(ymin, ymax, step=0.02))
        
    
    xmin = 1.4
    xmax = 2.7
    
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    t = np.arange(1.5, 2.6, step=0.2)
    lower_bound = 0
    upper_bound = np.empty(len(t))
    upper_bound.fill(1e-2)
     
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['spatial_mr_rate_perc']
        y = row['tsc_z_cost']
     
        cost = y
        csv_rows.append(row)
        
        lw = 1
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        try:
            label_convergence ='In Convergence Range'
            ax0.fill_between(t, lower_bound, cost, where=cost <= upper_bound, facecolor='lightgreen', edgecolor='springgreen', linewidth=1, alpha=0.2, interpolate=True,
                label='In convergence range')
        except:
            print('An error occured.')
            
        try:
            label_convergence ='Out Convergence Range'
            ax0.fill_between(t, lower_bound, cost, where=cost > upper_bound, facecolor='lightcoral', edgecolor='salmon', linewidth=1, alpha=0.2, interpolate=True,
                label='In convergence range')
        except:
            print('An error occured.')
            
        # Draw a thick red hline at y=0 that spans the xrange
        ax0.axhline(y=1e-2, linewidth=2, color='#d62728')
        ax0.text(2.1,1.05e-2,ur'Convergence Threshold = ' + ur'$1 \times 10^{-2}$', fontdict=font_label)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
        extra_text = ur'for Missing Timepoints Rate = ' + str(mr) + ur'%'
        lgd = create_legend_on_inside_upper_left(1, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
    
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_z_cost'] = pd.Series(run_result['tsc_z_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
    solution_id_csv = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_' + str(mr)
    solution_csv_path = os.path.join(solution_path, "csv")
    
    mrd.save_csv_by_path_adv(csv_result, solution_csv_path, solution_id_csv, index = False)

    
def tensor_order_by_tsc_all(tensor_dims, mr, roi_size, log_scale = False):
    file_path = os.path.join(roi_volume_folder,  data_all_by_volume)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)
    
    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
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
 
def draw_structural_results():
   
    dims = [3, 4]
    for k in range(5, 35, 5):   
        #tensor_order_by_tsc(dims, k, 'size0')
        pass
    
    dims = [3, 4, 2] 
    for k in range(5, 35, 5): 
        #tensor_order_by_tsc_all(dims, k, 'size0', log_scale = True)
        pass
        
    x_ticks = ['size0', 'size1', 'size2', 'size3', 'size4']
        
    dims = [3, 4]
    
    for j in x_ticks:
        #tensor_order_by_tsc_mr(dims,  j)
        pass
    
    draw_4D_by_roi_volume()
    #draw_4D_by_roi_volume_and_mr()
    
    draw_3D_by_roi_volume()
    #draw_3D_by_roi_volume_and_mr()
    draw_2D_by_roi_volume()
    #draw_2D_by_roi_volume_and_mr()

def draw_3D_4D_by_mspv_rate():
    solution_folder_tcs = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D-4D/tsc/mspv_rate"
    solution_folder_tcs_z = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D-4D/tsc_z/mspv_rate"
    data_path = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/results/avg_solution_cost_combined_all.csv"
    
    mr_map = {}
    mr_map[0] = 5
    mr_map[1] = 10
    mr_map[2] = 15
    mr_map[3] = 20
    mr_map[4] = 25
    mr_map[5] = 30

    dims = [3, 4]
    for k in range(5, 25, 5):   
    
        tensor_order_by_tsc_by_mr_spatial_volume(data_path, solution_folder_tcs, dims, k, title=False)
        tensor_order_by_tsc_by_mr_spatial_volume(data_path, solution_folder_tcs, dims, k, title=True)
             
        tensor_order_by_tsc_z_by_mr_spatial_volume(data_path, solution_folder_tcs_z, dims, k, title=False)
        tensor_order_by_tsc_z_by_mr_spatial_volume(data_path, solution_folder_tcs_z, dims, k, title=True)
        
        tensor_order_by_tsc_z_by_mr_spatial_volume_range(data_path, solution_folder_tcs_z, dims, k, title=False)
        tensor_order_by_tsc_z_by_mr_spatial_volume_range(data_path, solution_folder_tcs_z, dims, k, title=True)
        
    dims = [3, 4, 2]
    for k in range(5, 15, 5):
        tensor_order_by_tsc_z_by_mr_spatial_volume_range_all(data_path, solution_folder_tcs_z, dims, k, title=True)
        
def draw_3D_4D_by_mspv_rate_log_scale():
    solution_folder_tcs = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D-4D/log/tsc/mspv_rate"
    solution_folder_tcs_z = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D-4D/log/tsc_z/mspv_rate"
    data_path = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/results/avg_solution_cost_combined_all.csv"
    
    solution_folder_tcs_all_d = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D-4D/log/tsc/mspv_rate/all"
    solution_folder_tcs_z_all_d = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D-4D/log/tsc_z/mspv_rate/all"
    
    mr_map = {}
    mr_map[0] = 5
    mr_map[1] = 10
    mr_map[2] = 15
    mr_map[3] = 20
    mr_map[4] = 25
    mr_map[5] = 30

    dims = [4,3]
    for k in range(5, 30, 5):   
    
        tensor_order_by_tsc_by_mr_spatial_volume_log(data_path, solution_folder_tcs, dims, k, title=False, log_scale = True)
        #tensor_order_by_tsc_by_mr_spatial_volume_log(data_path, solution_folder_tcs, dims, k, title=True, log_scale = True)
             
        tensor_order_by_tsc_z_by_mr_spatial_volume_log(data_path, solution_folder_tcs_z, dims, k, title=False, log_scale = True)
        #tensor_order_by_tsc_z_by_mr_spatial_volume_log(data_path, solution_folder_tcs_z, dims, k, title=True, log_scale = True)
        
        #tensor_order_by_tsc_z_by_mr_spatial_volume_range(data_path, solution_folder_tcs_z, dims, k, title=False)
        #tensor_order_by_tsc_z_by_mr_spatial_volume_range(data_path, solution_folder_tcs_z, dims, k, title=True)
    
    dims = [4,3,2]
    for k in range(5, 30, 5):   
    
        tensor_order_by_tsc_by_mr_spatial_volume_log(data_path, solution_folder_tcs_all_d, dims, k, title=False, log_scale = True)
        #tensor_order_by_tsc_by_mr_spatial_volume_log(data_path, solution_folder_tcs, dims, k, title=True, log_scale = True)
             
        tensor_order_by_tsc_z_by_mr_spatial_volume_log(data_path, solution_folder_tcs_z_all_d, dims, k, title=False, log_scale = True)
            
    dims = [3, 4, 2]
    #for k in range(5, 15, 5):
    #    tensor_order_by_tsc_z_by_mr_spatial_volume_range_all(data_path, solution_folder_tcs_z, dims, k, title=True)
        

def draw_mtp_rate_by_msvp_rate_tcs_z(d, data_path, solution_path, title=False):  
    #structural_effect_mr_mspv_rate_tcs
    #structural_effect_mr_mspv_rate_tcs_z      
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_z_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['solid']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    
    fig = texfig.figure(width=tex_width, pad = 0)
    ax0 = plt.gca()
    
    if title:
        title = ur'Effect of Missing Timepoints Rate on fMRI Completion by Spatial Volume (%)'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
        
    #fig.subplots_adjust(wspace = 0.05, left = 0.1, right = 0.95, bottom = 0.15)
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 6,
        }
    
    subtitle = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(d)
    ax0.set_title(subtitle, fontdict=font_label)
    ax0.set_ylabel(ur'Tensor Completion Score $\vert Z_{score} \vert > 2$', fontdict=font_label)
    ax0.set_xlabel(ur'Spatial Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
        
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_scientific(True)
    yfmt.set_powerlimits((-1,1))
    ax0.yaxis.set_major_formatter(yfmt)
    ax0.yaxis.set_tick_params(labelsize=6)
    ax0.xaxis.set_tick_params(labelsize=6)
    
    ymin = 0
    ymax = 0.5e-1
    ax0.set_ylim([ymin,ymax])
    ax0.set_yticks(np.arange(ymin, ymax, step=0.005))
    
    xmin = 1.4
    xmax = 2.7
    
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    ctr = 0
    for k in range(5,20, 5):
        filter_name = k
        metric_name = 'tsc_z_cost'
        #plt.sca(ax0)
        
        subset = data.loc[data['tensor_dim'] == d]
        label, lines, row = plot_mr_spatial_volume(subset, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr_spatial_volume(subset, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
        extra_text =  ur'for Tensor Dimension ' + ur'$D$' + ur' = ' + str(d)
        lgd = create_legend_on_inside_upper_left(2, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
        
        csv_rows.append(row)
        
    solution_id = 'structural_effect_by_mr_mspv_rate_tcs_z_'  + str(d) + str("D") + str("_no_title")
    if title:
        solution_id = 'structural_effect_by_mr_mspv_rate_tcs_z_' + str(d) + str("D")
        
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
        
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_z_cost'] = pd.Series(run_result['tcs_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
   
    solution_csv_path = os.path.join(solution_path, "csv")
    mrd.save_csv_by_path_adv(run_result, solution_csv_path, solution_id, index = False)
    
def draw_mtp_rate_by_msvp_rate_tcs_z_log(d, data_path, solution_path, title=False, log_scale=True):  
    #structural_effect_mr_mspv_rate_tcs
    #structural_effect_mr_mspv_rate_tcs_z      
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_z_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    plt.clf()
    
    if title:
        layout = {'w_pad': 5, 'h_pad': 5
        }
        tex_width = 3.5
    else: 
        layout = {'w_pad': 0, 'h_pad': 0
        }
        tex_width = 3.5

    fig = texfig.figure(width=tex_width, pad = 0)
    fig.set_tight_layout(layout)
    
    ax0 = plt.gca()
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 8,
        }
    
    labelsize = 8
    
    if title:
        title = ur'Effect of Missing Timepoints Rate on fMRI Completion by Missing Ellipsoid Volume (%)'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
    
    ylabel = ur'Tensor Completion Score $\vert Z_{score} \vert > 2$'
    if log_scale:
        ylabel = ur'$\mathrm{TCS}_{\vert \mathrm{Z-Score} \vert > 2}$'
    
    ax0.set_ylabel(ylabel, fontdict=font_label)
    ax0.set_xlabel(ur'Missing Ellipsoid Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#ED0000FF", "#00468BFF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]

    ax0.set_prop_cycle(cycler('color', colors))
    format_axes(ax0)
    
    ax0.tick_params(direction="in")
    
    solution_id = 'structural_effect_by_mr_mspv_rate_tcs_z_'  + str(d) + str("D") + str("_no_title")
    if title:
        solution_id = 'structural_effect_by_mr_mspv_rate_tcs_z_' + str(d) + str("D")

    if log_scale:     
        ax0.set_yscale('symlog', nonposy='clip', linthreshy=0.005)
        yfmt = CustomTicker()
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=labelsize)
        ax0.xaxis.set_tick_params(labelsize=labelsize)
        solution_id = solution_id  + '_log_scale'
        ymin = 10**-3
        ymax = 10**0
        ax0.set_ylim(bottom = ymin, top = ymax)   
    else:    
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=labelsize)
        ax0.xaxis.set_tick_params(labelsize=labelsize)
    
    ax0.tick_params(direction="in")
    
    xmin = 1.4
    xmax = 2.7
    
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    plt.sca(ax0)
    extra_text =  ur'for Tensor Dimension ' + ur'$D$' + ur' = ' + str(d)
    
    ctr = 0
    for k in range(5,25, 5):
        filter_name = k
        metric_name = 'tsc_z_cost'
        #plt.sca(ax0)
        
        subset = data.loc[data['tensor_dim'] == d]
        label, lines, row = plot_mr_spatial_volume(subset, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr_spatial_volume(subset, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 1)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
        csv_rows.append(row)

    
    num_legend_col = 2
    if d == 2:
        lgd = create_legend_on_bottom_lower_left(num_legend_col, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
    else:
        lgd = create_legend_on_inside_upper_left(num_legend_col, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
            
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
        
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_z_cost'] = pd.Series(run_result['tcs_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
   
    solution_csv_path = os.path.join(solution_path, "csv")
    mrd.save_csv_by_path_adv(run_result, solution_csv_path, solution_id, index = False)
        
def draw_mtp_rate_by_msvp_rate_tcs(d, data_path, solution_path, title=False):  
    #structural_effect_mr_mspv_rate_tcs
    #structural_effect_mr_mspv_rate_tcs_z      
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['solid']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig = texfig.figure(width=tex_width, pad = 0)
    
    ax0 = plt.gca()
    
    if title:
        title = ur'Effect of Missing Timepoints Rate on fMRI Completion by Spatial Volume (%)'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
        
    #fig.subplots_adjust(wspace = 0.05, left = 0.1, right = 0.95, bottom = 0.15)
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 6,
        }
    
    subtitle = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(d)
    ax0.set_title(subtitle, fontdict=font_label)
    ax0.set_ylabel(ur'Tensor Completion Score', fontdict=font_label)
    ax0.set_xlabel(ur'Spatial Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
        
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_scientific(True)
    yfmt.set_powerlimits((-1,1))
    ax0.yaxis.set_major_formatter(yfmt)
    ax0.yaxis.set_tick_params(labelsize=6)
    ax0.xaxis.set_tick_params(labelsize=6)
    
    #ymin = -5e-3
    ymin = 0
    ymax = 0.4e-1
    ax0.set_ylim([ymin,ymax])
    ax0.set_yticks(np.arange(ymin, ymax, step=0.005))
    
    xmin = 1.4
    xmax = 2.7
    
    t = np.arange(1.5, 2.6, step=0.2)
    
    ax0.set_xlim([xmin,xmax])

    
    lower_bound = 0
    upper_bound = np.empty(len(t))
    upper_bound.fill(1e-2)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    ctr = 0
    for k in range(5,20, 5):
        filter_name = k
        metric_name = 'tcs_cost'
        #plt.sca(ax0)
        
        subset = data.loc[data['tensor_dim'] == d]
        label, lines, row = plot_mr_spatial_volume(subset, metric_name, filter_name)
        
        cost = row['tcs_cost']
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr_spatial_volume(subset, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        
        print "D= " +str(d)
        print "uppper bound" + str(upper_bound)
        print "cost " + str(cost)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
        extra_text =  ur'for Tensor Dimension ' + ur'$D$' + ur' = ' + str(d)
        lgd = create_legend_on_inside_upper_left(2, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
        
        csv_rows.append(row)
        
    solution_id = 'structural_effect_by_mr_mspv_rate_tcs_'  + str(d) + str("D") + str("_no_title")
    if title:
        solution_id = 'structural_effect_by_mr_mspv_rate_tcs_' + str(d) + str("D")
        
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
        
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_cost'] = pd.Series(run_result['tcs_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
   
    solution_csv_path = os.path.join(solution_path, "csv")
    mrd.save_csv_by_path_adv(run_result, solution_csv_path, solution_id, index = False)
    
def draw_mtp_rate_by_msvp_rate_tcs_log(d, data_path, solution_path, title=False, log_scale=False):  
       
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    plt.clf()
    
    plt.clf()
    
    if title:
        layout = {'w_pad': 5, 'h_pad': 5
        }
        tex_width = 3.5
    else: 
        layout = {'w_pad': 0, 'h_pad': 0
        }
        tex_width = 3.5

    fig = texfig.figure(width=tex_width, pad = 0)
    
    ax0 = plt.gca()
    fig.set_tight_layout(layout)
    
    ax0 = plt.gca()
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 8,
        }
    
    labelsize = 8
    
    if title:
        title = ur'Effect of Missing Timepoints Rate on fMRI Completion by Missing Ellipsoid Volume (%)'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
    
    ylabel = ur' Tensor Completion Score'
    if log_scale:
        ylabel = ur'Tensor Completion Score'
    
    ax0.set_ylabel(ylabel, fontdict=font_label)
    ax0.set_xlabel(ur'Missing Ellipsoid Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#ED0000FF", "#00468BFF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]

    ax0.set_prop_cycle(cycler('color', colors))
    format_axes(ax0)
    
    ax0.tick_params(direction="in")
    
    solution_id = 'structural_effect_by_mr_mspv_rate_tcs_'  + str(d) + str("D") + str("_no_title")
    if title:
        solution_id = 'structural_effect_by_mr_mspv_rate_tcs_' + str(d) + str("D")
        
    if log_scale:     
        ax0.set_yscale('symlog', nonposy='clip', linthreshy=0.005)
        yfmt = CustomTicker()
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=labelsize)
        ax0.xaxis.set_tick_params(labelsize=labelsize)
        solution_id = solution_id  + '_log_scale'
        ymin = 10**-3
        ymax = 10**0
        ax0.set_ylim(bottom = ymin, top = ymax)   
    else:    
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((-1,1))
        ax0.yaxis.set_major_formatter(yfmt)
        ax0.yaxis.set_tick_params(labelsize=labelsize)
        ax0.xaxis.set_tick_params(labelsize=labelsize)
    
    ax0.tick_params(direction="in")
    
    xmin = 1.4
    xmax = 2.7
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    plt.sca(ax0)
    extra_text =  ur'for Tensor Dimension ' + ur'$D$' + ur' = ' + str(d)

    ctr = 0
    for k in range(5,25, 5):
        filter_name = k
        metric_name = 'tcs_cost'
        
        subset = data.loc[data['tensor_dim'] == d]
        label, lines, row = plot_mr_spatial_volume(subset, metric_name, filter_name, log_scale)
        
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr_spatial_volume(subset, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 1)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
        csv_rows.append(row)
     
    num_legend_col = 2
    if d == 2:
        lgd = create_legend_on_bottom_lower_left(num_legend_col, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
    else:
        lgd = create_legend_on_inside_upper_left(num_legend_col, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
           
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
        
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_cost'] = pd.Series(run_result['tcs_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
   
    solution_csv_path = os.path.join(solution_path, "csv")
    mrd.save_csv_by_path_adv(run_result, solution_csv_path, solution_id, index = False)
    
def tensor_order_by_tsc_z_by_mr_spatial_volume_range_all(data_path, solution_path, tensor_dims, mr, title=False, log_scale = False):
    
    print ("Data Path:" + str(data_path))
    data = iot.read_data_structural(data_path)
    
    col_names = ['tensor_dim', 'k', 'observed_ratio', 'mr', 'ts_count', 'el_volume', 'roi_volume', 'tcs_cost', 'roi_volume_label', 'tsc_z_cost', 'rse_cost', 
                     'train_cost', 'solution_cost', 'image_final_path', 'scan_final_path', 'scan_iteration_path', 'metadata_path']
    
    col_names_csv = ['tensor_dim', 'missing_tp_rate', 'spatial_mr_rate', 'spatial_mr_rate_perc',  'tcs_z_cost']
    
    run_result = pd.DataFrame(col_names)
    csv_result = pd.DataFrame(col_names_csv)
    
    csv_rows = []
    
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    LINE_STYLES = ['solid']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    
    if title:
        title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Spatial Volume (%)'
        fig.suptitle(title, fontsize=7, fontweight='semibold')
        
    fig.subplots_adjust(wspace = 0.05, left = 0.1, right = 0.95, bottom = 0.15)
    
    font_label = {'family': 'serif',
        'weight': 'normal',
        'size': 6,
        }
    
    ax0.set_title(ur'Missing Timepoints Rate ' + ur' = ' + str(mr) + ur'%', fontdict=font_label)
    ax0.set_ylabel(ur'Tensor Completion Score $\vert Z_{score} \vert > 2$', fontdict=font_label)
    ax0.set_xlabel(ur'Missing Spatial Volume ' + ur'(%)', fontdict=font_label)
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_range_all' + str(mr) + str("_no_title")
    if title:
        solution_id = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_range_all' + str(mr)
        
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_scientific(True)
    yfmt.set_powerlimits((-1,1))
    ax0.yaxis.set_major_formatter(yfmt)
    ax0.yaxis.set_tick_params(labelsize=6)
    ax0.xaxis.set_tick_params(labelsize=6)
    
    
    ymin = 1e-3
    ymax = 0.35
    ax0.set_ylim([ymin,ymax])
    ax0.set_yticks(np.arange(ymin, ymax, step=0.05))    
    
    xmin = 1.4
    xmax = 2.7
    
    ax0.set_xlim([xmin,xmax])
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_mspv_rate = []
    top_labelstensor_mspv_rate = []
    
    t = np.arange(1.5, 2.6, step=0.2)
    lower_bound = 0
    upper_bound = np.empty(len(t))
    upper_bound.fill(1e-2)
     
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['spatial_mr_rate_perc']
        y = row['tsc_z_cost']
     
        cost = y
        csv_rows.append(row)
        
        lw = 1
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        try:
            label_convergence ='In Convergence Range'
            ax0.fill_between(t, lower_bound, cost, where=cost <= upper_bound, facecolor='lightgreen', edgecolor='springgreen', linewidth=1, alpha=0.2, interpolate=True,
                label='In convergence range')
        except:
            print('An error occured.')
            
        try:
            label_convergence ='Out Convergence Range'
            ax0.fill_between(t, lower_bound, cost, where=cost > upper_bound, facecolor='lightcoral', edgecolor='salmon', linewidth=1, alpha=0.2, interpolate=True,
                label='In convergence range')
        except:
            print('An error occured.')
            
        # Draw a thick red hline at y=0 that spans the xrange
        ax0.axhline(y=1e-2, linewidth=1, color='#d62728')
        ax0.text(2.1,1.9e-2,ur'Convergence Threshold = ' + ur'$1 \times 10^{-2}$', fontdict=font_label)
        
        top_legend_tensor_mspv_rate.append(scatter)
        top_labelstensor_mspv_rate.append(label)
        ctr = ctr + 1
        
        extra_text = ur'for Missing Timepoints Rate = ' + str(mr) + ur'%'
        lgd = create_legend_on_inside_upper_left(1, top_legend_tensor_mspv_rate, top_labelstensor_mspv_rate, extra_text)
    
    file_path = os.path.join(solution_path, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
    run_result = pd.concat(csv_rows, axis=0)
    
    csv_result['tensor_dim'] = pd.Series(run_result['tensor_dim'])
    csv_result['missing_tp_rate'] = pd.Series(run_result['mr'])
    csv_result['spatial_mr_rate'] = pd.Series(run_result['spatial_mr_rate'])
    csv_result['spatial_mr_rate_perc'] = pd.Series(run_result['spatial_mr_rate_perc'])
    csv_result['tcs_z_cost'] = pd.Series(run_result['tsc_z_cost'])
    
    csv_result.drop(axis=1, columns=[0], inplace=True)
    # save csv as well
    solution_id_csv = 'tensor_dimensionality_by_missing_spatial_volume_rate_and_fixed_timepoints_rate_tcs_z_cost_' + str(mr)
    solution_csv_path = os.path.join(solution_path, "csv")
    
    mrd.save_csv_by_path_adv(csv_result, solution_csv_path, solution_id_csv, index = False)

    
    
def draw_4D_by_mspv_rate_mtp_rate():
    solution_folder_tcs = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/4D/tsc/mspv_rate_mtp_rate"
    solution_folder_tcs_z = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/4D/tsc_z/mspv_rate_mtp_rate"
    data_path = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/results/avg_solution_cost_combined_all.csv"
    
    #3D
    solution_folder_3Dtcs = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D/tsc/mspv_rate_mtp_rate"
    solution_folder_3Dtcs_z = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D/tsc_z/mspv_rate_mtp_rate"
    
    #2D
    solution_folder_2Dtcs = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/2D/tsc/mspv_rate_mtp_rate"
    solution_folder_2Dtcs_z = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/2D/tsc_z/mspv_rate_mtp_rate"
    
    draw_mtp_rate_by_msvp_rate_tcs(4, data_path, solution_folder_tcs, title=False)
    draw_mtp_rate_by_msvp_rate_tcs(4, data_path, solution_folder_tcs, title=True)
 
    draw_mtp_rate_by_msvp_rate_tcs_z(4, data_path, solution_folder_tcs_z, title=False)
    draw_mtp_rate_by_msvp_rate_tcs_z(4, data_path, solution_folder_tcs_z, title=True)
    
    draw_mtp_rate_by_msvp_rate_tcs(3, data_path, solution_folder_3Dtcs, title=False)
    draw_mtp_rate_by_msvp_rate_tcs(3, data_path, solution_folder_3Dtcs, title=True)
    
    draw_mtp_rate_by_msvp_rate_tcs_z(3, data_path, solution_folder_3Dtcs_z, title=False)
    draw_mtp_rate_by_msvp_rate_tcs_z(3, data_path, solution_folder_3Dtcs_z, title=True)
    
    draw_mtp_rate_by_msvp_rate_tcs(2, data_path, solution_folder_2Dtcs, title=False)
    draw_mtp_rate_by_msvp_rate_tcs(2, data_path, solution_folder_2Dtcs, title=True)
    
    draw_mtp_rate_by_msvp_rate_tcs_z(2, data_path, solution_folder_2Dtcs_z, title=False)
    draw_mtp_rate_by_msvp_rate_tcs_z(2, data_path, solution_folder_2Dtcs_z, title=True)
    
def draw_4D_by_mspv_rate_mtp_rate_log_scale():
    solution_folder_tcs = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/4D/log/tsc/mspv_rate_mtp_rate"
    solution_folder_tcs_z = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/4D/log/tsc_z/mspv_rate_mtp_rate"
    data_path = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/results/avg_solution_cost_combined_all.csv"
    
    #3D
    solution_folder_3Dtcs = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D/log/tsc/mspv_rate_mtp_rate"
    solution_folder_3Dtcs_z = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/3D/log/tsc_z/mspv_rate_mtp_rate"
    
    #2D
    solution_folder_2Dtcs = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/2D/log/tsc/mspv_rate_mtp_rate"
    solution_folder_2Dtcs_z = "/work/pl/sch/analysis/my_paper/analysis/results/structural/multirun/figures/2_2/2D/log/tsc_z/mspv_rate_mtp_rate"
    
    draw_mtp_rate_by_msvp_rate_tcs_log(4, data_path, solution_folder_tcs, title=False, log_scale=True)
    #draw_mtp_rate_by_msvp_rate_tcs_log(4, data_path, solution_folder_tcs, title=True, log_scale=True)
 
    draw_mtp_rate_by_msvp_rate_tcs_z_log(4, data_path, solution_folder_tcs_z, title=False, log_scale=True)
    #draw_mtp_rate_by_msvp_rate_tcs_z_log(4, data_path, solution_folder_tcs_z, title=True, log_scale=True)
    
    draw_mtp_rate_by_msvp_rate_tcs_log(3, data_path, solution_folder_3Dtcs, title=False, log_scale=True)
    #draw_mtp_rate_by_msvp_rate_tcs_log(3, data_path, solution_folder_3Dtcs, title=True, log_scale=True)
    
    draw_mtp_rate_by_msvp_rate_tcs_z_log(3, data_path, solution_folder_3Dtcs_z, title=False, log_scale=True)
    #draw_mtp_rate_by_msvp_rate_tcs_z_log(3, data_path, solution_folder_3Dtcs_z, title=True, log_scale=True)
    
    draw_mtp_rate_by_msvp_rate_tcs_log(2, data_path, solution_folder_2Dtcs, title=False, log_scale=True)
    #draw_mtp_rate_by_msvp_rate_tcs_log(2, data_path, solution_folder_2Dtcs, title=True, log_scale=True)
    
    draw_mtp_rate_by_msvp_rate_tcs_z_log(2, data_path, solution_folder_2Dtcs_z, title=False, log_scale=True)
    #draw_mtp_rate_by_msvp_rate_tcs_z_log(2, data_path, solution_folder_2Dtcs_z, title=True, log_scale=True)

def draw_structural_resultsv2():
    #draw_3D_4D_by_mspv_rate()
    draw_3D_4D_by_mspv_rate_log_scale()
    draw_4D_by_mspv_rate_mtp_rate_log_scale()
    #draw_4D_by_mspv_rate_mtp_rate()
    
    
if __name__ == "__main__":
    #draw_structural_results()
    draw_structural_resultsv2()