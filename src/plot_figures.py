import pandas
import data_util as du
import mri_draw_utils as mrd
import matplotlib.pyplot as plt
from sympy.codegen.ffunctions import kind
from collections import OrderedDict
import pandas as pd
import numpy as np

def plot_rse_random(file_path_random):
    
    report_df = pandas.read_excel(open(file_path_random,'rb'), sheet_name='all_summary')
    
    output_df = pandas.read_csv("/work/test.csv")
   
    fig_file_name = "solution/reports/rse_by_d_random_updated"
    fig_file_name_output = "solution/reports/rse_by_d_random_output1"
    
    output = OrderedDict()
    d_arr = []
    rel_error_arr = []
    miss_ratio_arr = []
    
    fig, ax = plt.subplots()
    
    for i in (2,3,4):
        row = report_df[report_df['d']==i]
        ratio = row['miss_ratio']
        rse = row['rel_error']
        ax.plot(ratio, rse, label = str("d=") + str(i), linewidth = 2.0, marker=".", ms=7, mfc='r', mec='r')
             
    
    plt.xlabel("mr")
    plt.ylabel("RSE")
    ax.legend()
    sup_title = "Random Missing Value Pattern"
    title = "RSE vs Missing Ratio by Tensor Mode"
    plt.title(title)
    plt.suptitle(sup_title, size=10, fontweight='bold')

    mrd.save_report_fig(fig_file_name)
    mrd.save_csv(report_df, fig_file_name)
        
    output_df.drop(columns=['old_rel_error', 'pattern'], inplace=True)
    
   
    pandas.set_option('display.float_format', '{:.8E}'.format)
    
    print(output_df)
    
    mrd.save_csv(output_df, fig_file_name_output)
    
    file_path = "/work/pl/sch/analysis/scripts/csv_data/solution/reports/rse_by_d_random_output1.csv"
    output_df.to_csv(file_path, index=False,float_format='%1.8E')
    
      
    output_df2D = output_df.loc[output_df['d'] == 2]
    output_df3D = output_df.loc[output_df['d'] == 3]
    output_df4D = output_df.loc[output_df['d'] == 4]
    
    output_df2D.set_index('miss_ratio', inplace=True)
    output_df3D.set_index('miss_ratio', inplace=True)
    output_df4D.set_index('miss_ratio', inplace=True)

    
    print(output_df2D)
    print(output_df3D)
    print(output_df4D)
    
    columns = ['miss_ratio','2D', '3D', '4D']
    
    #index = output_df2D['miss_ratio']
    
    df = pandas.DataFrame(columns=columns)
    #df['miss_ratio'] = output_df2D['miss_ratio']
    #df['2D'] = output_df2D['rel_error']
    #df['3D'] = output_df3D['rel_error']
    #df['4D'] = output_df4D['rel_error']

    
    print(df)
    
    #print(output_df3D['rel_error'])
    
    header = ['RSE, tensor mode = 2', 'RSE, tensor mode = 3', 'RSE, tensor mode = 4']
    cross_tab_df = pandas.concat([output_df2D['rel_error'], output_df3D['rel_error'], output_df4D['rel_error']],
                                  ignore_index=True, axis = 1)
    
    file_path1 = "/work/pl/sch/analysis/scripts/csv_data/solution/reports/rse_by_d_random_crosstab.csv"
    cross_tab_df.to_csv(file_path1, index_label = 'Missing Values, (%)', float_format='%1.8E', header=header)
    print cross_tab_df
    
    #output_df.r
    
    #np.savetxt('test.out', x, fmt='%1.4e')

    
def plot_rse_structural(file_strcutural):
    
    report_df = pandas.read_excel(open(file_strcutural,'rb'), sheetname='volume')
   
    fig_file_name = "solution/reports/rse_by_ellipse_volume"
    
    fig, ax = plt.subplots()
    
    for i in (3,4):
        row = report_df[report_df['d']==i]
        ratio = row['miss_ratio']
        rse = row['rel_error']
        volume = row['volume']
        ax.plot(volume, rse, linewidth = 2.0, marker=".", ms=7, mfc='r', mec='r')
    
    plt.xlabel("Ellipse Volume (voxels)")
    plt.ylabel("RSE")
    sup_title = "Structural Missing Value Pattern"
    title = "RSE by 3D-ellipsoid volume"
    plt.title(title)
    plt.suptitle(sup_title, size=10, fontweight='bold')

    mrd.save_report_fig(fig_file_name)
    mrd.save_csv(report_df, fig_file_name)
    
def plot_rse_by_frame(file_frame):
    
    report_df = pandas.read_excel(open(file_frame,'rb'), sheetname='by_number_of_frame')
   
    fig_file_name = "solution/reports/rse_by_frame_count"
    
    fig, ax = plt.subplots()
    
    for i in (3,4):
        row = report_df[report_df['d']==i]
        ratio = row['missing_ratio']
        rse = row['rel_error']
        frame_count = row['frame_miss']
        frame_perc = row['frame_perc']
        ax.plot(frame_perc, rse, linewidth = 2.0, marker=".", ms=7, mfc='r', mec='r')
    
    plt.xlabel("% Timepoints Corrupted")
    plt.ylabel("RSE")
    sup_title = "Structural Missing Value Pattern"
    title = "RSE by Number of Corrupted Timepoints"
    plt.title(title)
    plt.suptitle(sup_title, size=10, fontweight='bold')

    mrd.save_report_fig(fig_file_name)
    mrd.save_csv(report_df, fig_file_name)
    
def plot_rse_error_bar(file_path_random):
    
    report_df = pandas.read_excel(open(file_path_random,'rb'), sheetname='summary')
    solution_df = []
   
    fig_file_name = "solution/reports/rse_by_d_random_error_bar"
    
    fig, ax = plt.subplots()
    
    for i in (3,4):
        row = report_df[report_df['d']==i]
        ratio = row['miss_ratio']
        rse = row['rel_error']
        ax.errorbar(ratio, rse, label = str("d=") + str(i), linewidth = 2.0, marker=".", ms=7, mfc='r', mec='r')
    
    plt.xlabel("mr")
    plt.ylabel("RSE")
    ax.legend()
    sup_title = "Random Missing Value Pattern"
    title = "RSE vs Missing Ratio by Tensor Mode"
    plt.title(title)
    plt.suptitle(sup_title, size=10, fontweight='bold')

    mrd.save_report_fig(fig_file_name)
    mrd.save_csv(report_df, fig_file_name)
    

if __name__ == "__main__":

    file_path_random = "/work/pl/sch/analysis/scripts/csv_data/solution/summary/summary_by_d1.xlsx"
    file_path_structural = "/work/pl/sch/analysis/scripts/csv_data/solution/summary/summary_by_d.xlsx"
    file_frame = "/work/pl/sch/analysis/scripts/csv_data/solution/summary/summary_by_d.xlsx"
    plot_rse_random(file_path_random)
    #plot_rse_structural(file_path_structural)
    #plot_rse_by_frame(file_frame)
    #plot_rse_error_bar(file_path_random)