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
import math
from nilearn import plotting
from nilearn import image
from nilearn.plotting import find_xyz_cut_coords
import file_service as fs

config_loc = path.join('config')
config_filename = 'solution'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

tex_width = 3.5
default_ratio = (math.sqrt(5.0) - 1.0) / 2.0

processed_results = "/apps/git/python/tensor-completion-reports/tensor-completion-datanalysis/results/random/single-run/results"

def save_fig_pdf_white(fig_id, tight_layout=True):
        path_pdf = os.path.join(fig_id + ".pdf")
        print("Saving figure", path_pdf)
        print("Called from draw utils")
        plt.savefig(path_pdf, format='pdf', dpi=1000, bbox_inches="tight", pad_inches=-0.1)
        plt.close()
        
def save_subplots(fig_id, fig=None):
    path_pdf = os.path.join(fig_id + ".pdf")
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 0.02, 0.03, 0.02, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_frame_on(False)
    fig.savefig(path_pdf, bbox_inches='tight', facecolor='k', edgecolor='k')
    plt.close()

def save_subplots_wbg(fig_id, fig=None):
    path_pdf = os.path.join(fig_id + ".pdf")
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 0.02, 0.03, 0.02, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_frame_on(False)
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close()
    
def save_subplots_wbg_png(fig_id, fig=None):
    path_pdf = os.path.join(fig_id + ".png")
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 0.02, 0.03, 0.02, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_frame_on(False)
    fig.savefig(path_pdf, bbox_inches='tight')
    plt.close()


def draw_images_xyz_projection(images, folder, fig_id, timepoint=0, title=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
    # fig = plt.figure(frameon = False, figsize=(3.5,2.183))
    
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig, ax0 = plt.subplots(ncols=1, nrows=1, sharex=True, figsize=[3.5, 2.183], frameon=False)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.tight_layout()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    row0 = images[0]  
    img0 = row0[0]
    
    plotting.plot_epi(image.index_img(img0, timepoint), annotate=False, draw_cross=False, bg_img=None, black_bg=False,
                                   figure=fig, display_mode='ortho', cmap='jet') 
    
    # plotting.plot_epi(image.index_img(img0,timepoint), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
    #                               figure= fig, display_mode='x', axes =ax1, cmap='jet', cut_coords = [32]) 
     
    # plotting.plot_epi(image.index_img(img0,timepoint), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
     #                              figure= fig, display_mode='z', axes =ax2, cmap='jet', cut_coords = [32]) 
        
      
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots(fig_path)
    
def draw_images_xyz_projection_as_grid_by_mr_column(images, mrs, rows, cols, folder, fig_id, timepoint, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3.5, 2.5))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0, wspace=0.03)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=-0.8, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    ctr = 0
    for mr in mrs:
        main_ax = fig.add_subplot(grid[0, ctr])
        main_ax.set_facecolor("blue")    
        main_ax.set_aspect('equal')
    
        plotting.plot_epi(image.index_img(images[mr], timepoint), annotate=annotate, bg_img=None, black_bg=False,
                                   figure=fig, axes=main_ax, cmap='jet', display_mode='z', cut_coords=[coord]) 
        
        ctr = ctr + 1
     
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots_wbg(fig_path)
    
def draw_images_xyz_projection_as_grid_by_all_column(images, mrs, rows, cols, folder, fig_id, timepoint, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3.5, 2))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0, wspace=0.03)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=-0.8, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
   
    for i in range(rows):   
        img = images[i]
        for mr in mrs:
            ctr = 0
            main_ax = fig.add_subplot(grid[i, ctr])
            main_ax.set_facecolor("blue")    
            main_ax.set_aspect('equal')
            plotting.plot_epi(image.index_img(img[mr], timepoint), annotate=annotate, bg_img=None, black_bg=False,
                                   figure=fig, axes=main_ax, cmap='jet', display_mode='z', cut_coords=[coord]) 
            ctr = ctr + 1
     
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots_wbg(fig_path)
    
def draw_images_xyz_projection_as_grid_by_mr_row(images, rows, cols, folder, fig_id, timepoint, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3, 1.5))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0, wspace=0.03)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=-0.8, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    ctr = 0
    for i in range(cols):
        main_ax = fig.add_subplot(grid[0, i])
        main_ax.set_facecolor("blue")    
        main_ax.set_aspect('equal')
        
        plotting.plot_epi(image.index_img(images[0][i], timepoint), annotate=annotate, bg_img=None, black_bg=False,
                                   figure=fig, axes=main_ax, cmap='jet', display_mode='y', cut_coords=[coord]) 
    

     
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots_wbg(fig_path)

def draw_images_xyz_projection_as_grid_by_mr_row2(images, rows, cols, folder, fig_id, timepoint, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3, 1.5))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0, wspace=0.03)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=-0.8, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    ctr = 0
    for i in range(cols):
        main_ax = fig.add_subplot(grid[0, i])
        main_ax.set_facecolor("blue")    
        main_ax.set_aspect('equal')
        
        plotting.plot_epi(image.index_img(images[i], timepoint), annotate=annotate, bg_img=None, black_bg=False,
                                   figure=fig, axes=main_ax, cmap='jet', display_mode='x', cut_coords=[coord]) 
    

     
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots_wbg(fig_path)
    
def draw_images_xyz_projection_as_grid_by_mr_row_timepoint(images, rows, cols, mr, folder, fig_id, timepoints, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3.5, 2.5))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0, wspace=0.03)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=0, bottom=-0.6, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
    display  = 'x'      
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)        
    
    for j in range(rows):   
        for i in range(cols):
            
            main_ax = fig.add_subplot(grid[j, i])
            main_ax.set_facecolor("blue")    
            main_ax.set_aspect('equal')
            
            img = images[mr]
            
            if j >0:
                display = 'y'
            
            print "timepoints[i] = "  + str(timepoints[i])
            plotting.plot_epi(image.index_img(img, timepoints[i]), annotate=annotate, bg_img=None, black_bg=False,
                                   figure=fig, axes=main_ax, cmap='jet', display_mode=display, cut_coords=[coord]) 
    

     
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots_wbg(fig_path)
    
def draw_images_xyz_projection_as_grid_by_mr_row_timepoint1(images, rows, cols, mr, folder, fig_id, timepoints, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3.5, 2.5))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0, wspace=0.03)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=-0.6, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
    display  = 'y'      
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)        
    
    for j in range(rows):   
        for i in range(cols):
            
            main_ax = fig.add_subplot(grid[j, i])
            main_ax.set_facecolor("blue")    
            main_ax.set_aspect('equal')
            
            img = images[mr]
            
            if j >0:
                display = 'y'
            
            print "timepoints[i] = "  + str(timepoints[i])
            plotting.plot_epi(image.index_img(img, timepoints[i]), annotate=annotate, bg_img=None, black_bg=False,
                                   figure=fig, axes=main_ax, cmap='jet', display_mode=display, cut_coords=[coord]) 
    

     
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots_wbg(fig_path)
    save_subplots_wbg_png(fig_path)
    
def draw_images_xyz_projection_as_grid_by_mr_row_timepoint1_str(images, rows, cols, mr, folder, fig_id, timepoint, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3.5, 2.5))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0, wspace=0.03)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=-0.6, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
    display  = 'x'      
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)        
    
    for j in range(rows):   
        for i in range(cols):
            
            main_ax = fig.add_subplot(grid[j, i])
            main_ax.set_facecolor("blue")    
            main_ax.set_aspect('equal')
            
            img = images[i]
            
            print "timepoint = "  + str(timepoint)
            plotting.plot_epi(image.index_img(img, timepoint), annotate=annotate, bg_img=None, black_bg=False,
                                   figure=fig, axes=main_ax, cmap='jet', display_mode=display, cut_coords=[coord]) 
    

     
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots_wbg(fig_path)
    save_subplots_wbg_png(fig_path)
    
def draw_images_xyz_projection_as_grid_all_rows(images, rows, cols, folder, fig_id, timepoint, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3.5,2.4))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0, wspace=0.03)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=-0.8, right=1, left=0,
            hspace=0, wspace=0.03)

    plt.margins(0, 0) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    ctr = 0
    
    for j in range(rows):   
        for i in range(cols):
            
            main_ax = fig.add_subplot(grid[j, i])
            main_ax.set_facecolor("blue")    
            main_ax.set_aspect('equal')
            
            img = images[j]
            plotting.plot_epi(image.index_img(img[0][i], timepoint), annotate=annotate, bg_img=None, black_bg=False,
                                   figure=fig, axes=main_ax, cmap='jet', display_mode='z', cut_coords=[coord]) 
    

     
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots_wbg(fig_path)

def draw_images_xyz_projection_as_grid_all_rows_ortho(x_true, x_miss, images,  mr, rows, cols, folder, fig_id, timepoint, annotate=True, title=None, coord=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3.5,2.5))
    
    grid_rows = rows
    grid_cols = cols
    
    grid = gridspec.GridSpec(grid_rows+2, grid_cols, hspace=0, wspace=0.03)  
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=0.1, right=1, left=0,
            hspace=0.03, wspace=0.0)

    plt.margins(0, 0) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    
    # x_true
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")       
    main_ax.set_aspect('equal')
    plotting.plot_epi(image.index_img(x_true[0][0], timepoint), draw_cross=False, annotate=False, bg_img=None, black_bg=True,
                                   figure=fig, axes=main_ax, cmap='jet', cut_coords=coord) 
    
    # x_miss
    main_ax = fig.add_subplot(grid[1, 0])
    main_ax.set_facecolor("blue")       
    main_ax.set_aspect('equal')
    plotting.plot_epi(image.index_img(x_miss[0][1], timepoint), draw_cross=False, annotate=False, bg_img=None, black_bg=True,
                                   figure=fig, axes=main_ax, cmap='jet', cut_coords=coord) 
    
    k = [2,3,4]
    for j in k:           
            
            img = images[j-2]
            
            main_ax = fig.add_subplot(grid[j, 0])
            main_ax.set_facecolor("blue")    
            main_ax.set_aspect('equal')
            
            plotting.plot_epi(image.index_img(img[0][2], timepoint), draw_cross=False, annotate=False, bg_img=None, black_bg=True,
                                   figure=fig, axes=main_ax, cmap='jet', cut_coords=coord) 
            
         
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots(fig_path)
    
    
def draw_images_xyz_projection_as_grid(images, folder, fig_id, timepoint, annotate=True, title=None, coord=None, draw_cross=True):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    plt.axis("off")
        
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig = plt.figure(frameon=False, figsize=(3.5, 2.183))
    
    grid_rows = 1
    grid_cols = 1
    
    grid = gridspec.GridSpec(grid_rows, grid_cols, hspace=0.2, wspace=0.2)
    
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.subplots_adjust(top=1, bottom=-0.5, right=1, left=0,
            hspace=0, wspace=0)

    plt.margins(0, 0) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    img0 = images
    
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")    
   
    main_ax.set_aspect('equal')
    
    plotting.plot_epi(image.index_img(img0, timepoint), annotate=annotate, bg_img=None, black_bg=True,
                                   figure=fig, axes=main_ax, cmap='jet', cut_coords=coord, draw_cross=draw_cross) 
        
      
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_subplots(fig_path)
    
def draw_images_single_row(images, folder, fig_id, title=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    fig = plt.figure(frameon=False, figsize=(3.5, 2.183))
    
    plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,
    right=False
    )
    
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, nrows=3, sharey=True, sharex=True)
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, wspace=0.05, hspace=0.05, left=0.05, right=0.95) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    ax_list0 = []
    ax_list0.append((ax0, ax1, ax2)) 
    
    axes = []
    
    axes.append(ax_list0)
    row0 = images[0]
    
    img0 = row0[0]
    img1 = row0[1]
    img2 = row0[2]
    
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    
    ax0.xaxis.set_major_locator(plt.NullLocator())
    ax0.yaxis.set_major_locator(plt.NullLocator())
    
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    
    plotting.plot_epi(image.index_img(img0, 1), annotate=False, draw_cross=False, bg_img=None, black_bg=True,
                                   figure=fig, display_mode='z', axes=ax0, cmap='jet', cut_coords=[32]) 
    
    plotting.plot_epi(image.index_img(img0, 1), annotate=False, draw_cross=False, bg_img=None, black_bg=True,
                                   figure=fig, display_mode='z', axes=ax1, cmap='jet', cut_coords=[32]) 
     
    plotting.plot_epi(image.index_img(img0, 1), annotate=False, draw_cross=False, bg_img=None, black_bg=True,
                                   figure=fig, display_mode='z', axes=ax2, cmap='jet', cut_coords=[32]) 
        
      
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_fig_pdf_white(fig_path)

def run():
    pass

def get_images_single_per_mr(folder, mr):
    
    x_true = "x_true_img_"
    x_miss = "x_miss_img_"
    x_hat = "x_hat_img_"
    
    x_true_path = os.path.join(folder, str(mr), "images", x_true + str(mr) + ".nii")
    x_miss_path = os.path.join(folder, str(mr), "images", x_miss + str(mr) + ".nii")
    x_hat_path = os.path.join(folder, str(mr), "images", x_hat + str(mr) + ".nii")

    row0 = []
    row0.append(x_true_path)
    row0.append(x_miss_path)
    row0.append(x_hat_path)
    
    images = {0:row0}
    
    print "x_true_path: " + str(x_true_path)
    
    return images

def get_original_images_single_per_mr_set(folder, mrs):
    
    x_true = "x_true_img_"
    x_miss = "x_miss_img_"
    x_hat = "x_hat_img_"
    
    images= {}
    for mr in mrs:
        x_true_path = os.path.join(folder, str(mr), "images", x_true + str(mr) + ".nii")
        row = {mr:x_true_path}
        images.update(row)
        print "x_true_path: " + str(x_true_path)
    
    return images


def get_original_images_single_per_mr(folder, mrs):
    
    x_true = "x_true_img_"
    
    images= {}
    x_miss_path = os.path.join(folder, str(mrs), "images", x_true + str(mrs) + ".nii")
    row = {mrs:x_miss_path}
    images.update(row)
    print "x_true_path: " + str(x_miss_path)
    
    return images

def get_corrupted_images_single_per_mr_set(folder, mrs):
    
    x_true = "x_true_img_"
    x_miss = "x_miss_img_"
    x_hat = "x_hat_img_"
    
    images= {}
    for mr in mrs:
        x_miss_path = os.path.join(folder, str(mr), "images", x_miss + str(mr) + ".nii")
        row = {mr:x_miss_path}
        images.update(row)
        print "x_miss_path: " + str(x_miss_path)
    
    return images

def get_corrupted_images_single_per_mr(folder, mrs):
    
    x_miss = "x_miss_img_"
        
    images= {}
    x_miss_path = os.path.join(folder, str(mrs), "images", x_miss + str(mrs) + ".nii")
    row = {mrs:x_miss_path}
    images.update(row)
    print "x_miss_path: " + str(x_miss_path)
    
    
    return images

def get_x_hat_images_single_per_mr_set(folder, mrs):
    
    x_true = "x_true_img_"
    x_miss = "x_miss_img_"
    x_hat = "x_hat_img_"
    
    images= {}
    for mr in mrs:
        x_hat_path = os.path.join(folder, str(mr), "images", x_hat + str(mr) + ".nii")
        row = {mr:x_hat_path}
        images.update(row)
        print "x_hat_path: " + str(x_hat_path)
    
    return images

def get_x_hat_images_single_per_mr(folder, mrs):
    
    x_hat = "x_hat_img_"
    
    images= {}
    x_miss_path = os.path.join(folder, str(mrs), "images", x_hat + str(mrs) + ".nii")
    row = {mrs:x_miss_path}
    images.update(row)
    print "x_hat_path: " + str(x_miss_path)
    
    return images

def get_x_hat_images_single_per_mr1(folder, mrs):
    
    x_hat = "x_hat_img_"
    
    images= {}
    x_miss_path = os.path.join(folder, str(mrs), x_hat + str(mrs) + ".nii")
    row = {mrs:x_miss_path}
    images.update(row)
    print "x_hat_path: " + str(x_miss_path)
    
    return images
    
def draw_examples():
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    mrs = [10]
    ts = [0, 69, 119, 143]
    
    img_folder = os.path.join(img_folder, "examples")
    for t in ts:
        for mr in mrs:
            images = get_images_single_per_mr(path4D, mr)
            fig_id = "example_timepoint_" + str(t)
            draw_images_xyz_projection_as_grid(images[0][0], img_folder, fig_id, timepoint=t, title=None)

def draw4D():
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    mrs = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    ts = [0, 69, 119, 143]
    
    solution_cost_4D_path = os.path.join(processed_results, "final_cost_by_mr_4d.csv")
    solution_cost4D = pd.read_csv(solution_cost_4D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "by_mr")
    for t in ts:
        for mr in mrs:
                mr_dir = os.path.join(img_folder_by_mr, str(mr))
                fs.ensure_dir(mr_dir)
                images = get_images_single_per_mr(path4D, mr)
                
                coord = find_xyz_cut_coords(image.index_img(images[0][0], t))
        
                fig_id = "x_true_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][0], mr_dir, fig_id, timepoint=t, title=None, coord=coord)
                
                fig_id = "x_miss_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][1], mr_dir, fig_id, timepoint=t, annotate=False, title=None, coord=coord)
                
                fig_id = "x_hat_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][2], mr_dir, fig_id, timepoint=t, annotate=False, title=None, coord=coord)
                
                filtered4D = solution_cost4D.loc[solution_cost4D['mr'] == mr]
                
                mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(images[0][0], t), image.index_img(images[0][2], t), image.index_img(images[0][1], t), "4D fMRI Tensor Completion",
                    float(filtered4D['tcs_cost']), 1.0 - mr * 0.01, float(filtered4D['tcs_cost']), float(filtered4D['tsc_z_cost']), 2, coord=None, folder=mr_dir, iteration=-1, time=t)

def draw3D():
    path3D = "/work/pl/sch/analysis/my_work/analysis/results/random/3D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/3D"
    
    # compute 70 mr for 3D 
    mrs = [10, 20, 25, 30, 40, 50, 60, 75, 80, 90]
    ts = [0, 69, 119, 143]
    
    solution_cost_path = os.path.join(processed_results, "final_cost_by_mr_3d.csv")
    solution_cost = pd.read_csv(solution_cost_path)
    
    img_folder_by_mr = os.path.join(img_folder, "by_mr")
    for t in ts:
        for mr in mrs:
                mr_dir = os.path.join(img_folder_by_mr, str(mr))
                fs.ensure_dir(mr_dir)
                images = get_images_single_per_mr(path3D, mr)
                
                coord = find_xyz_cut_coords(image.index_img(images[0][0], t))
        
                fig_id = "x_true_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][0], mr_dir, fig_id, timepoint=t, title=None, coord=coord)
                
                fig_id = "x_miss_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][1], mr_dir, fig_id, timepoint=t, annotate=False, title=None, coord=coord)
                
                fig_id = "x_hat_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][2], mr_dir, fig_id, timepoint=t, annotate=False, title=None, coord=coord)
                
                filtered_solution = solution_cost.loc[solution_cost['mr'] == mr]
                
                mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(images[0][0], t), image.index_img(images[0][2], t), image.index_img(images[0][1], t), "3D fMRI Tensor Completion",
                    float(filtered_solution['tcs_cost']), 1.0 - mr * 0.01, float(filtered_solution['tcs_cost']), float(filtered_solution['tsc_z_cost']), 2, coord=None, folder=mr_dir, iteration=-1, time=t)

def draw2D():
    path3D = "/work/pl/sch/analysis/my_work/analysis/results/random/2D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/2D"
    
    
    mrs = [10, 20, 25, 30, 40, 50, 60, 75, 80, 90]
    ts = [0, 69, 119, 143]
    
    solution_cost_path = os.path.join(processed_results, "final_cost_by_mr_2d.csv")
    solution_cost = pd.read_csv(solution_cost_path)
    
    img_folder_by_mr = os.path.join(img_folder, "by_mr")
    for t in ts:
        for mr in mrs:
                mr_dir = os.path.join(img_folder_by_mr, str(mr))
                fs.ensure_dir(mr_dir)
                images = get_images_single_per_mr(path3D, mr)
                
                coord = find_xyz_cut_coords(image.index_img(images[0][0], t))
        
                fig_id = "x_true_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][0], mr_dir, fig_id, timepoint=t, title=None, coord=coord)
                
                fig_id = "x_miss_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][1], mr_dir, fig_id, timepoint=t, annotate=False, title=None, coord=coord)
                
                fig_id = "x_hat_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
                draw_images_xyz_projection_as_grid(images[0][2], mr_dir, fig_id, timepoint=t, annotate=False, title=None, coord=coord)
                
                filtered_solution = solution_cost.loc[solution_cost['mr'] == mr]
                
                mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(images[0][0], t), image.index_img(images[0][2], t), image.index_img(images[0][1], t), "3D fMRI Tensor Completion",
                    float(filtered_solution['tcs_cost']), 1.0 - mr * 0.01, float(filtered_solution['tcs_cost']), float(filtered_solution['tsc_z_cost']), 2, coord=None, folder=mr_dir, iteration=-1, time=t)

def draw4Dbymr_column():
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    mrs = [25, 50, 75, 90]
       
    solution_cost_4D_path = os.path.join(processed_results, "final_cost_by_mr_4d.csv")
    solution_cost4D = pd.read_csv(solution_cost_4D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_column")
    fs.ensure_dir(img_folder_by_mr)
    
    x_true_images = get_original_images_single_per_mr_set(path4D, mrs)
    x_miss_images = get_corrupted_images_single_per_mr_set(path4D, mrs)
    x_hat_images = get_x_hat_images_single_per_mr_set(path4D, mrs)
    
    all_images1 = []
    
    all_images1.append(x_true_images)
    all_images1.append(x_miss_images)
    all_images1.append(x_hat_images)
    
    coord = 32
    fig_id = "tensor_completion_4D_by_mr_column_true_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_true_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_4D_by_mr_column_miss_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_miss_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_4D_by_mr_column_hat_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_hat_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_4D_by_mr_all_columns_" + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_all_column(all_images1, mrs, len(all_images1), len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    
def draw4Dbymr_row():
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    mrs = [25, 50, 75, 90]
       
    solution_cost_4D_path = os.path.join(processed_results, "final_cost_by_mr_4d.csv")
    solution_cost4D = pd.read_csv(solution_cost_4D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_row")
    fs.ensure_dir(img_folder_by_mr)
    
    images = get_original_images_single_per_mr_set(path4D, mrs)
    x_miss_images = get_corrupted_images_single_per_mr_set(path4D, mrs)
    x_hat_images = get_x_hat_images_single_per_mr_set(path4D, mrs)
    
    coord = 32
    
    
    all_images = []
    for mr in mrs:
        images = get_images_single_per_mr(path4D, mr)       
        all_images.append(images)      
        coord = 32
        
        fig_id = "tensor_completion_4D_by_mr_" + str(mr) + "_" + str("timepoint") + "_" + str(0)
        draw_images_xyz_projection_as_grid_by_mr_row(images, 1, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
        
    fig_id = "tensor_completion_4D_by_mr_" + str('all_') + "_" + str("timepoint") + "_" + str(0)    
    draw_images_xyz_projection_as_grid_all_rows(all_images, 4, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)

def draw4Dbymr_row1():
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D/eaurasip"
    
    mrs = [25, 50, 75, 90]
       
    solution_cost_4D_path = os.path.join(processed_results, "final_cost_by_mr_4d.csv")
    solution_cost4D = pd.read_csv(solution_cost_4D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_row")
    fs.ensure_dir(img_folder_by_mr)
    
    images = get_original_images_single_per_mr_set(path4D, mrs)
    x_miss_images = get_corrupted_images_single_per_mr_set(path4D, mrs)
    x_hat_images = get_x_hat_images_single_per_mr_set(path4D, mrs)
    
    coord = 32
    
    random4D15_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/15"
    random4D20_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/20/images/"
    random4D30_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/30/images/"
    random4D40_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/40/images/"
    random4D50_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/50/images/x_miss_img_50.nii"
    random4D70_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/70/images/"
    random4D75_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/75/images/"
    random4D90_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/90/images/"
    random4D90_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/90/images/"
    
    all_images = []
    all_images.append(random4D30_+"x_true_img_30.nii")
    all_images.append(random4D50_)
    all_images.append(random4D20_+"x_hat_img_20.nii") #geman
    #all_images.append(random4D30_+"x_hat_img_30.nii") # log
    all_images.append(random4D90_+"x_hat_img_90.nii")#lsp
    all_images.append(random4D90_+"x_hat_img_90.nii") #schatten
    
    #for mr in mrs:
        #images = get_images_single_per_mr(path4D, mr)       
        #all_images.append(images)      
    coord = 32
        
    fig_id = "tensor_completion_4D_by_mr_"  + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_row2(all_images, 1, 5, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
        
    fig_id = "tensor_completion_4D_by_mr_" + str('all_') + "_" + str("timepoint") + "_" + str(0)    
    #draw_images_xyz_projection_as_grid_all_rows(all_images, 4, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
def draw4Dbymr_row2():
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work2/pl/cmsc/cmsc2/cmsc/portfolio/figures/general/img"
    
    mrs = [25, 50, 75, 90]
       
    solution_cost_4D_path = os.path.join(processed_results, "final_cost_by_mr_4d.csv")
    solution_cost4D = pd.read_csv(solution_cost_4D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_row")
    fs.ensure_dir(img_folder_by_mr)
    
    images = get_original_images_single_per_mr_set(path4D, mrs)
    x_miss_images = get_corrupted_images_single_per_mr_set(path4D, mrs)
    x_hat_images = get_x_hat_images_single_per_mr_set(path4D, mrs)
    
    coord = 32
    
    random4D15_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/15"
    random4D20_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/20/images/"
    random4D30_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/30/images/"
    random4D40_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/40/images/"
    random4D50_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/50/images/x_miss_img_50.nii"
    random4D70_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/70/images/"
    random4D75_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/75/images/"
    random4D90_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/90/images/"
    random4D90_="/work/pl/sch/analysis/my_work/analysis/results/random/4D/90/images/"
    
    x_true = "/work/pl/sch/analysis/data/COBRE001/cobre.nii"
    x_recon = "/work/pl/sch/analysis/matlab_toolboxes/prob-tt/x_recon.nii"
   
    all_images = []
    all_images.append(random4D30_+"x_true_img_30.nii")
    all_images.append(random4D50_)
    all_images.append(random4D20_+"x_hat_img_20.nii") #geman
    all_images.append(random4D30_+"x_hat_img_30.nii") # log
    #all_images.append(random4D90_+"x_hat_img_90.nii")#lsp
    all_images.append(random4D90_+"x_hat_img_90.nii") #schatten
    
    #for mr in mrs:
        #images = get_images_single_per_mr(path4D, mr)       
        #all_images.append(images)      
    coord = 32
        
    fig_id = "tensor_completion_4D_by_mr_"  + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_row2(all_images, 1, 4, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
        
    fig_id = "tensor_completion_4D_by_mr_" + str('all_') + "_" + str("timepoint") + "_" + str(0)    
    #draw_images_xyz_projection_as_grid_all_rows(all_images, 4, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)    

def draw4Dby_subject_by_row():
    path4Ds1 = "/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/subject1"
    path4Ds2 = "/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/subject2"
    path4Ds3 = "/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/subject3"
    img_folder = "/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/images"
    
    mrs = [50]
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_row")
    fs.ensure_dir(img_folder_by_mr)
    
    s1_images = get_images_single_per_mr(path4Ds1, 50)
    s2_images = get_images_single_per_mr(path4Ds2, 50)
    s3_images = get_images_single_per_mr(path4Ds3, 50)
    coord = 32
        
    fig_id = "tensor_completion_4D_by_subject_" + str("s1") + "_" + str("timepoint") + "_" + str(75)
    draw_images_xyz_projection_as_grid_by_mr_row(s1_images, 1, 3, img_folder_by_mr, fig_id, 75, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_4D_by_subject_" + str("s2") + "_" + str("timepoint") + "_" + str(75)
    draw_images_xyz_projection_as_grid_by_mr_row(s2_images, 1, 3, img_folder_by_mr, fig_id, 75, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_4D_by_subject_" + str("s3") + "_" + str("timepoint") + "_" + str(75)
    draw_images_xyz_projection_as_grid_by_mr_row(s3_images, 1, 3, img_folder_by_mr, fig_id, 75, annotate=False, title=None, coord=coord)
    
    all_images = [] 
    all_images.append(s1_images)
    all_images.append(s2_images)
    all_images.append(s3_images)
       
    fig_id = "tensor_completion_4D_by_mr_" + str('all_subjects') + "_" + str("timepoint") + "_" + str(75)    
    draw_images_xyz_projection_as_grid_all_rows(all_images, 4, 3, img_folder_by_mr, fig_id, 75, annotate=False, title=None, coord=coord)
    
def draw4Dby_penalty_by_row():
    path4Ds1 = "/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/subject1"
    path4Ds2 = "/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/subject2"
    path4Ds3 = "/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/subject3"
    img_folder = "/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/images"
    
    mrs = [50]
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_row")
    fs.ensure_dir(img_folder_by_mr)
    
    s1_images = get_images_single_per_mr(path4Ds1, 50)
    s2_images = get_images_single_per_mr(path4Ds2, 50)
    s3_images = get_images_single_per_mr(path4Ds3, 50)
    coord = 32
        
    fig_id = "tensor_completion_4D_by_subject_" + str("s1") + "_" + str("timepoint") + "_" + str(75)
    draw_images_xyz_projection_as_grid_by_mr_row(s1_images, 1, 3, img_folder_by_mr, fig_id, 75, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_4D_by_subject_" + str("s2") + "_" + str("timepoint") + "_" + str(75)
    draw_images_xyz_projection_as_grid_by_mr_row(s2_images, 1, 3, img_folder_by_mr, fig_id, 75, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_4D_by_subject_" + str("s3") + "_" + str("timepoint") + "_" + str(75)
    draw_images_xyz_projection_as_grid_by_mr_row(s3_images, 1, 3, img_folder_by_mr, fig_id, 75, annotate=False, title=None, coord=coord)
    
    all_images = [] 
    all_images.append(s1_images)
    all_images.append(s2_images)
    all_images.append(s3_images)
       
    fig_id = "tensor_completion_4D_by_mr_" + str('all_subjects') + "_" + str("timepoint") + "_" + str(75)    
    draw_images_xyz_projection_as_grid_all_rows(all_images, 4, 3, img_folder_by_mr, fig_id, 75, annotate=False, title=None, coord=coord)
    
def draworiginal_randomwith_ts(mr1):
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    timepoints = [0, 69, 119, 143]
    mrs = [25]
    mrs.append(mr1)
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    
    for mr in mrs:
        x_true_images = get_original_images_single_per_mr(path4D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_true_4D_by_mr_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint1(x_true_images, 1, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)
        

def drawcorrupted_randomwith_ts(mr1):
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    timepoints = [0, 69, 119, 143]
    mrs = [25, 50, 75]
    mrs.append(mr1)
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    
    for mr in mrs:
        x_true_images = get_corrupted_images_single_per_mr(path4D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_miss_4D_by_mr_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint(x_true_images, 2, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)

def drawcorrupted_str_with_ts(mr1):
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/structural/figures/4D"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/structural/figures/4D/images"
    
    timepoints = [0, 69, 119, 143]
    mrs = [25, 50, 75]
    mrs.append(mr1)
    img_folder_by_mr = os.path.join(img_folder, "size_by_mr")
    fs.ensure_dir(img_folder_by_mr)
    
    #/work/str/4D/mr/5/size1/scans/final/mr/5/5/x_miss_img_5.nii
    #/work/str/4D/mr/5/size2/scans/final/mr/5/5/x_miss_img_5.nii
    #/work/str/4D/mr/5/size3/scans/final/mr/5/5/x_miss_img_5.nii
    #/work/str/4D/mr/5/size4/scans/final/mr/5/5/x_miss_img_5.nii
    #/work/str/4D/mr/5/size5/scans/final/mr/5/5/x_miss_img_5.nii
    #/work/str/4D/mr/5/size6/scans/final/mr/5/5/x_miss_img_5.nii
    
    x_true_images = {}
    x_true_images[0] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_true_img_5.nii"
    x_true_images[1] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_miss_img_5.nii"
    x_true_images[2] = "/work/str/4D/mr/5/size1/scans/final/mr/5/5/x_miss_img_5.nii"
    x_true_images[3] = "/work/str/4D/mr/5/size2/scans/final/mr/5/5/x_miss_img_5.nii"
    x_true_images[4] = "/work/str/4D/mr/5/size3/scans/final/mr/5/5/x_miss_img_5.nii"
    x_true_images[5] = "/work/str/4D/mr/5/size4/scans/final/mr/5/5/x_miss_img_5.nii"
    x_true_images[6] = "/work/str/4D/mr/5/size5/scans/final/mr/5/5/x_miss_img_5.nii"
    
    x_hat_images_5tr = {}
    x_hat_images_5tr[0] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_true_img_5.nii"
    x_hat_images_5tr[1] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_hat_img_5.nii"
    x_hat_images_5tr[2] = "/work/str/4D/mr/5/size1/scans/final/mr/5/5/x_hat_img_5.nii"
    x_hat_images_5tr[3] = "/work/str/4D/mr/5/size2/scans/final/mr/5/5/x_hat_img_5.nii"
    x_hat_images_5tr[4] = "/work/str/4D/mr/5/size3/scans/final/mr/5/5/x_hat_img_5.nii"
    x_hat_images_5tr[5] = "/work/str/4D/mr/5/size4/scans/final/mr/5/5/x_hat_img_5.nii"
    x_hat_images_5tr[6] = "/work/str/4D/mr/5/size5/scans/final/mr/5/5/x_hat_img_5.nii"
    
    x_hat_images_10tr = {}
    x_hat_images_10tr[0] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_true_img_5.nii"
    x_hat_images_10tr[1] = "/work/str/4D/mr/10/size0/scans/final/mr/10/10/x_hat_img_10.nii"
    x_hat_images_10tr[2] = "/work/str/4D/mr/10/size1/scans/final/mr/10/10/x_hat_img_10.nii"
    x_hat_images_10tr[3] = "/work/str/4D/mr/10/size2/scans/final/mr/10/10/x_hat_img_10.nii"
    x_hat_images_10tr[4] = "/work/str/4D/mr/10/size3/scans/final/mr/10/10/x_hat_img_10.nii"
    x_hat_images_10tr[5] = "/work/str/4D/mr/10/size4/scans/final/mr/10/10/x_hat_img_10.nii"
    x_hat_images_10tr[6] = "/work/str/4D/mr/10/size5/scans/final/mr/10/10/x_hat_img_10.nii"
    
    x_hat_images_15tr = {}
    x_hat_images_15tr[0] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_true_img_5.nii"
    x_hat_images_15tr[1] = "/work/str/4D/mr/15/size0/scans/final/mr/15/15/x_hat_img_15.nii"
    x_hat_images_15tr[2] = "/work/str/4D/mr/15/size1/scans/final/mr/15/15/x_hat_img_15.nii"
    x_hat_images_15tr[3] = "/work/str/4D/mr/15/size2/scans/final/mr/15/15/x_hat_img_15.nii"
    x_hat_images_15tr[4] = "/work/str/4D/mr/15/size3/scans/final/mr/15/15/x_hat_img_15.nii"
    x_hat_images_15tr[5] = "/work/str/4D/mr/15/size4/scans/final/mr/15/15/x_hat_img_15.nii"
    x_hat_images_15tr[6] = "/work/str/4D/mr/15/size5/scans/final/mr/15/15/x_hat_img_15.nii"
    x_miss_images_15tr = "/work/str/4D/mr/15/size1/scans/final/mr/15/15/x_miss_img_15.nii"
    
    x_hat_images_20tr = {}
    x_hat_images_20tr[0] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_true_img_5.nii"
    x_hat_images_20tr[1] = "/work/str/4D/mr/20/size0/scans/final/mr/20/20/x_hat_img_20.nii"
    x_hat_images_20tr[2] = "/work/str/4D/mr/20/size1/scans/final/mr/20/20/x_hat_img_20.nii"
    x_hat_images_20tr[3] = "/work/str/4D/mr/20/size2/scans/final/mr/20/20/x_hat_img_20.nii"
    x_hat_images_20tr[4] = "/work/str/4D/mr/20/size3/scans/final/mr/20/20/x_hat_img_20.nii"
    x_hat_images_20tr[5] = "/work/str/4D/mr/20/size4/scans/final/mr/20/20/x_hat_img_20.nii"
    x_hat_images_20tr[6] = "/work/str/4D/mr/20/size5/scans/final/mr/20/20/x_hat_img_20.nii"
    x_miss_images_20tr = "/work/str/4D/mr/20/size3/scans/final/mr/20/20/x_miss_img_20.nii"
    
    x_hat_images_25tr = {}
    x_hat_images_25tr[0] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_true_img_5.nii"
    x_hat_images_25tr[1] = "/work/str/4D/mr/25/size0/scans/final/mr/25/25/x_hat_img_25.nii"
    x_hat_images_25tr[2] = "/work/str/4D/mr/25/size1/scans/final/mr/25/25/x_hat_img_25.nii"
    x_hat_images_25tr[3] = "/work/str/4D/mr/25/size2/scans/final/mr/25/25/x_hat_img_25.nii"
    x_hat_images_25tr[4] = "/work/str/4D/mr/25/size3/scans/final/mr/25/25/x_hat_img_25.nii"
    x_hat_images_25tr[5] = "/work/str/4D/mr/25/size4/scans/final/mr/25/25/x_hat_img_25.nii"
    x_hat_images_25tr[6] = "/work/str/4D/mr/25/size5/scans/final/mr/25/25/x_hat_img_25.nii"
    
    x_hat_images_30tr = {}
    x_hat_images_30tr[0] = "/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_true_img_5.nii"
    x_hat_images_30tr[1] = "/work/str/4D/mr/30/size0/scans/final/mr/30/30/x_hat_img_30.nii"
    x_hat_images_30tr[2] = "/work/str/4D/mr/30/size1/scans/final/mr/30/30/x_hat_img_30.nii"
    x_hat_images_30tr[3] = "/work/str/4D/mr/30/size2/scans/final/mr/30/30/x_hat_img_30.nii"
    x_hat_images_30tr[4] = "/work/str/4D/mr/30/size3/scans/final/mr/30/30/x_hat_img_30.nii"
    x_hat_images_30tr[5] = "/work/str/4D/mr/30/size4/scans/final/mr/30/30/x_hat_img_30.nii"
    x_hat_images_30tr[6] = "/work/str/4D/mr/30/size5/scans/final/mr/30/30/x_hat_img_30.nii"


    coord = find_xyz_cut_coords(image.index_img(x_true_images[0], 0))
    print "coord: " + str(coord)
    
    
    print x_true_images
    coord = 2
    
    fig_id = "x_miss_4D_by_size_" + "timepoint_19"
    draw_images_xyz_projection_as_grid_by_mr_row_timepoint1_str(x_true_images, 1, 7, 5, img_folder_by_mr, fig_id, 19, annotate=False, title=None, coord=coord)
    
    fig_id = "x_hat_4D_by_size_" + "timepoints_19_mtr5"
    draw_images_xyz_projection_as_grid_by_mr_row_timepoint1_str(x_hat_images_5tr, 1, 7, 5, img_folder_by_mr, fig_id, 19, annotate=False, title=None, coord=coord)
    
    fig_id = "x_hat_4D_by_size_" + "timepoints_19_mtr10"
    draw_images_xyz_projection_as_grid_by_mr_row_timepoint1_str(x_hat_images_10tr, 1, 7, 5, img_folder_by_mr, fig_id, 19, annotate=False, title=None, coord=coord)
    
    fig_id = "x_hat_4D_by_size_" + "timepoints_19_mtr15"
    draw_images_xyz_projection_as_grid_by_mr_row_timepoint1_str(x_hat_images_20tr, 1, 7, 5, img_folder_by_mr, fig_id, 19, annotate=False, title=None, coord=coord)
    
    fig_id = "x_hat_4D_by_size_" + "timepoints_19_mtr20"
    draw_images_xyz_projection_as_grid_by_mr_row_timepoint1_str(x_hat_images_20tr, 1, 7, 5, img_folder_by_mr, fig_id, 19, annotate=False, title=None, coord=coord)
  
    fig_id = "x_hat_4D_by_size_" + "timepoints_19_mtr25"
    draw_images_xyz_projection_as_grid_by_mr_row_timepoint1_str(x_hat_images_25tr, 1, 7, 5, img_folder_by_mr, fig_id, 19, annotate=False, title=None, coord=coord)
  
  
    fig_id = "x_hat_4D_by_size_" + "timepoints_19_mtr30"
    draw_images_xyz_projection_as_grid_by_mr_row_timepoint1_str(x_hat_images_30tr, 1, 7, 5, img_folder_by_mr, fig_id, 19, annotate=False, title=None, coord=coord)
    
    
    fig_id = "x_true_4D_" + "str_mtr_15_size1"

    draw_images_xyz_projection_as_grid(x_hat_images_20tr[0], img_folder_by_mr, fig_id, timepoint=19, title=None, coord=[2,32, 22])
    
    fig_id = "x_miss_4D_" + "str_mtr_15_size1"
    draw_images_xyz_projection_as_grid(x_miss_images_15tr, img_folder_by_mr, fig_id, timepoint=19, annotate=False, title=None, coord=[2,32, 22], draw_cross=False)
    
    fig_id = "x_hat_4D_" + "str_mtr_15_size1"
    draw_images_xyz_projection_as_grid(x_hat_images_15tr[2], img_folder_by_mr, fig_id, timepoint=19, annotate=False, title=None, coord=[2,32, 22], draw_cross=False)
              
def drawcorrupted_randomwith_ts_2mr(mr1):
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    timepoints = [0, 69, 119, 143]
    mrs = [75]
    mrs.append(mr1)
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    
    for mr in mrs:
        x_true_images = get_corrupted_images_single_per_mr(path4D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_miss_4D_by_mr_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint1(x_true_images, 1, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)
        
def drawcorrupted_randomwith_ts_2mr2(mr1):
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    timepoints = [0, 69, 119, 143]
    mrs = [50, 75]
    mrs.append(mr1)
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    
    for mr in mrs:
        x_true_images = get_corrupted_images_single_per_mr(path4D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_miss_4D_by_mr_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint(x_true_images, 1, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)
        
def drawcompleted_randomwith_ts(mr1):
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    timepoints = [0, 69, 119, 143]
    mrs = [25, 50, 75]
    mrs.append(mr1)
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    
    for mr in mrs:
        x_true_images = get_x_hat_images_single_per_mr(path4D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_hat_4D_by_mr_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint(x_true_images, 2, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)
        
    path3D = "/work/pl/sch/analysis/my_work/analysis/results/random/3D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/3D"
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    for mr in mrs:
        x_true_images = get_x_hat_images_single_per_mr(path3D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_hat_3D_by_mr_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint(x_true_images, 2, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)
        
    path2D = "/work/pl/sch/analysis/my_work/analysis/results/random/2D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/2D"
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    for mr in mrs:
        x_true_images = get_x_hat_images_single_per_mr(path2D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_hat_2D_by_mr_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint(x_true_images, 2, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)

def drawcompleted_randomwith_ts2(mr1):
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    timepoints = [0, 69, 119, 143]
    mrs = [25, 50, 75]
    mrs.append(mr1)
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    
    for mr in mrs:
        x_true_images = get_x_hat_images_single_per_mr(path4D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_hat_4D_by_mr_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint1(x_true_images, 1, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)
        
def drawcompleted_randomwith_ts_cp(mr1):
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/cp/4D/scans/run_2019-01-18_07_15_37/mr/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/cp/4D/scans/run_2019-01-18_07_15_37/mr"
    
    timepoints = [0, 69, 119, 143]
    mrs = [75]
   
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    
    for mr in mrs:
        x_true_images = get_x_hat_images_single_per_mr1(path4D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_hat_4D_by_mr_cp_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint1(x_true_images,1, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)
        
def drawcompleted_randomwith_ts_tucker(mr1):
    path4D = "/apps/git/python/tensor-completion-reports/tensor-completion-datanalysis/results/random/single-run/tucker-random/mr"
    img_folder = "/apps/git/python/tensor-completion-reports/tensor-completion-datanalysis/results/random/single-run/tucker-random/mr"
    
    timepoints = [0, 69, 119, 143]
    mrs = [50,80]
   
    img_folder_by_mr = os.path.join(img_folder, "mr_by_ts")
    fs.ensure_dir(img_folder_by_mr)
    
    
    for mr in mrs:
        x_true_images = get_x_hat_images_single_per_mr1(path4D, mr)
        print x_true_images
        coord = 32
        fig_id = "x_hat_4D_by_mr_tucker_" + str(mr) + "_" + "timepoints_all"
        draw_images_xyz_projection_as_grid_by_mr_row_timepoint1(x_true_images, 1, 4, mr, img_folder_by_mr, fig_id, timepoints, annotate=False, title=None, coord=coord)

# comppund images 3D

def draw3Dbymr_column():
    path3D = "/work/pl/sch/analysis/my_work/analysis/results/random/3D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/3D"
    
    mrs = [25, 50, 75, 90]
       
    solution_cost_3D_path = os.path.join(processed_results, "final_cost_by_mr_3d.csv")
    solution_cost3D = pd.read_csv(solution_cost_3D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_column")
    fs.ensure_dir(img_folder_by_mr)
    
    x_true_images = get_original_images_single_per_mr_set(path3D, mrs)
    x_miss_images = get_corrupted_images_single_per_mr_set(path3D, mrs)
    x_hat_images = get_x_hat_images_single_per_mr_set(path3D, mrs)
    
    all_images1 = []
    
    all_images1.append(x_true_images)
    all_images1.append(x_miss_images)
    all_images1.append(x_hat_images)
    
    coord = 32
    fig_id = "tensor_completion_3D_by_mr_column_true_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_true_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_3D_by_mr_column_miss_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_miss_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_3D_by_mr_column_hat_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_hat_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_3D_by_mr_all_columns_" + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_all_column(all_images1, mrs, len(all_images1), len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord) 
    
def draw3Dbymr_row():
    path3D = "/work/pl/sch/analysis/my_work/analysis/results/random/3D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/3D"
    
    mrs = [25, 50, 75, 90]
       
    solution_cost_3D_path = os.path.join(processed_results, "final_cost_by_mr_3d.csv")
    solution_cost3D = pd.read_csv(solution_cost_3D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_row")
    fs.ensure_dir(img_folder_by_mr)
    
    images = get_original_images_single_per_mr_set(path3D, mrs)
    x_miss_images = get_corrupted_images_single_per_mr_set(path3D, mrs)
    x_hat_images = get_x_hat_images_single_per_mr_set(path3D, mrs)
    
    coord = 32
    
    
    all_images = []
    for mr in mrs:
        images = get_images_single_per_mr(path3D, mr)       
        all_images.append(images)      
        coord = 32
        
        fig_id = "tensor_completion_3D_by_mr_" + str(mr) + "_" + str("timepoint") + "_" + str(0)
        draw_images_xyz_projection_as_grid_by_mr_row(images, 1, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
        
    fig_id = "tensor_completion_3D_by_mr_" + str('all_') + "_" + str("timepoint") + "_" + str(0)    
    draw_images_xyz_projection_as_grid_all_rows(all_images, 4, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
# 2D COMPOUND IMAGES
def draw2Dbymr_column():
    path2D = "/work/pl/sch/analysis/my_work/analysis/results/random/2D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/2D"
    
    mrs = [25, 50, 75, 90]
       
    solution_cost_2D_path = os.path.join(processed_results, "final_cost_by_mr_2d.csv")
    solution_cost2D = pd.read_csv(solution_cost_2D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_column")
    fs.ensure_dir(img_folder_by_mr)
    
    x_true_images = get_original_images_single_per_mr_set(path2D, mrs)
    x_miss_images = get_corrupted_images_single_per_mr_set(path2D, mrs)
    x_hat_images = get_x_hat_images_single_per_mr_set(path2D, mrs)
    
    all_images1 = []
    
    all_images1.append(x_true_images)
    all_images1.append(x_miss_images)
    all_images1.append(x_hat_images)
    
    coord = 32
    fig_id = "tensor_completion_2D_by_mr_column_true_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_true_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_2D_by_mr_column_miss_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_miss_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_2D_by_mr_column_hat_" + str('25_50_75_90') + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_mr_column(x_hat_images, mrs, 1, len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    fig_id = "tensor_completion_2D_by_mr_all_columns_" + "_" + str("timepoint") + "_" + str(0)
    draw_images_xyz_projection_as_grid_by_all_column(all_images1, mrs, len(all_images1), len(mrs), img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
def draw2Dbymr_row():
    path2D = "/work/pl/sch/analysis/my_work/analysis/results/random/2D/"
    img_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/2D"
    
    mrs = [25, 50, 75, 90]
       
    solution_cost_2D_path = os.path.join(processed_results, "final_cost_by_mr_2d.csv")
    solution_cost2D = pd.read_csv(solution_cost_2D_path)
    
    img_folder_by_mr = os.path.join(img_folder, "mr_by_row")
    fs.ensure_dir(img_folder_by_mr)
    
    images = get_original_images_single_per_mr_set(path2D, mrs)
    x_miss_images = get_corrupted_images_single_per_mr_set(path2D, mrs)
    x_hat_images = get_x_hat_images_single_per_mr_set(path2D, mrs)
    
    coord = 32
    
    all_images = []
    for mr in mrs:
        images = get_images_single_per_mr(path2D, mr)       
        all_images.append(images)      
        coord = 32
        
        fig_id = "tensor_completion_2D_by_mr_" + str(mr) + "_" + str("timepoint") + "_" + str(0)
        draw_images_xyz_projection_as_grid_by_mr_row(images, 1, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
        
    draw_images_xyz_projection_as_grid_all_rows(all_images, 4, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)
    
    
def combined_dimension_by_row():
    path2D = "/work/pl/sch/analysis/my_work/analysis/results/random/2D/"
    img_folder2D = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/2D"
     
    path3D = "/work/pl/sch/analysis/my_work/analysis/results/random/3D/"
    img_folder3D = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/3D"
    
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder4D = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    mrs = [25, 50, 75, 90]
    
    all_dim_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images"
    img_folder_by_mr = os.path.join(all_dim_folder, "all_by_dim")
    fs.ensure_dir(img_folder_by_mr)
    
    all_images = []
    coord = 32
    for mr in mrs:
        all_images = []
        images2D = get_images_single_per_mr(path2D, mr)       
        images3D = get_images_single_per_mr(path3D, mr)
        images4D = get_images_single_per_mr(path4D, mr)
        
        all_images.append(images2D)  
        all_images.append(images3D)  
        all_images.append(images4D)  
        fig_id = "tensor_completion_all_dim" + str(mr) + "_" + str("timepoint") + "_" + str(0)  
        draw_images_xyz_projection_as_grid_all_rows(all_images, 3, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)

def combined_dimension_by_row_ortho():        
    path2D = "/work/pl/sch/analysis/my_work/analysis/results/random/2D/"
    img_folder2D = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/2D"
     
    path3D = "/work/pl/sch/analysis/my_work/analysis/results/random/3D/"
    img_folder3D = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/3D"
    
    path4D = "/work/pl/sch/analysis/my_work/analysis/results/random/4D/"
    img_folder4D = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D"
    
    mrs = [25, 50, 75, 90]
    
    mr1 = 75
    mrs = [75]
    all_dim_folder = "/work/pl/sch/analysis/my_work/analysis/results/random/figures/images"
    img_folder_by_mr = os.path.join(all_dim_folder, "all_by_dim_ortho")
    fs.ensure_dir(img_folder_by_mr)
    
    x_true_images = get_images_single_per_mr(path4D, mr1)
    x_miss_images = get_images_single_per_mr(path4D, mr1)
    x_hat_images = get_images_single_per_mr(path4D, mr1)
    
    t = 0
    coord = find_xyz_cut_coords(image.index_img(x_true_images[0][0], t))
    print "coord: " + coord
    #
    for mr in mrs:
        x_true = x_true_images
        x_miss = x_miss_images
        
        all_images = []
        images2D = get_images_single_per_mr(path2D, mr)      
        images3D = get_images_single_per_mr(path3D, mr)
        images4D = get_images_single_per_mr(path4D, mr)
        
        all_images.append(images2D)  
        all_images.append(images3D)  
        all_images.append(images4D)  
        
        fig_id = "x_true_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
        draw_images_xyz_projection_as_grid(x_true[0][0], img_folder_by_mr, fig_id, timepoint=t, title=None, coord=coord)
                
        fig_id = "x_miss_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
        
        draw_images_xyz_projection_as_grid(x_miss[0][1], img_folder_by_mr, fig_id, timepoint=t, title=None, coord=coord)
        
        #
        fig_id = "x_2D_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
        draw_images_xyz_projection_as_grid(images2D[0][2], img_folder_by_mr, fig_id, timepoint=t, title=None, coord=coord)
        
        fig_id = "x_3D_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
        draw_images_xyz_projection_as_grid(images3D[0][2], img_folder_by_mr, fig_id, timepoint=t, title=None, coord=coord)
        
        fig_id = "x_4D_" + "missing_ratio_" + str(mr) + "_" + str("timepoint") + "_" + str(t)
        draw_images_xyz_projection_as_grid(images4D[0][2], img_folder_by_mr, fig_id, timepoint=t, title=None, coord=coord)
    
        fig_id = 'tensor_completion_all_dim_ortho_75_timepoint_0'
        draw_images_xyz_projection_as_grid_all_rows_ortho(x_true, x_miss, all_images, mr, 3, 3, img_folder_by_mr, fig_id, 0, annotate=False, title=None, coord=coord)


# /work/pl/sch/analysis/my_work/analysis/results/random/figures/images/4D/by-ts
#cp.random.images=/work/pl/sch/analysis/my_work/analysis/results/random/cp/4D/scans/run_2019-01-18_07_15_37/mr

#/work/str/4D/mr/5/size0/scans/final/mr/5/5
#/work/str/4D/mr/5/size0/scans/final/mr/10/10
#/work/str/4D/mr/5/size0/scans/final/mr/15/15
#/work/str/4D/mr/5/size0/scans/final/mr/15/15
#size0
#x_true_img_5.nii
#/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_true_img_5.nii
#/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_miss_img_5.nii
#/work/str/4D/mr/5/size0/scans/final/mr/5/5/x_miss_img_5.nii
#/work/str/4D/mr/5/size1/scans/final/mr/5/5/x_miss_img_5.nii
#/work/str/4D/mr/5/size2/scans/final/mr/5/5/x_miss_img_5.nii
#/work/str/4D/mr/5/size3/scans/final/mr/5/5/x_miss_img_5.nii
#/work/str/4D/mr/5/size4/scans/final/mr/5/5/x_miss_img_5.nii
#/work/str/4D/mr/5/size5/scans/final/mr/5/5/x_miss_img_5.nii
#/work/str/4D/mr/5/size6/scans/final/mr/5/5/x_miss_img_5.nii


if __name__ == "__main__":
    run()
    #draw_examples()
    #draw4D()
    #draw3D()
    #draw4Dbymr_column()
    #draw4Dbymr_row()
    #draw4Dby_subject_by_row()
    draw4Dbymr_row1() #eupsipco
    #draw4Dbymr_row2()
    #draw3Dbymr_column()
    #draw3Dbymr_row()
    #draw2Dbymr_column()
    #draw2Dbymr_row()
    #combined_dimension_by_row()
    #combined_dimension_by_row_ortho()
    #drawcorrupted_randomwith_ts(25)
    #drawcompleted_randomwith_ts(25)
    #draworiginal_randomwith_ts(25)
    #drawcompleted_randomwith_ts(25)
    #drawcompleted_randomwith_ts2(25)
    #drawcorrupted_randomwith_ts_2mr(25)
    #drawcompleted_randomwith_ts_cp(25)
    #drawcompleted_randomwith_ts_tucker(25)
    #drawcorrupted_str_with_ts(5)
#/work/pl/sch/analysis/my_work/med-ima-mac2/mult-subject/subject1