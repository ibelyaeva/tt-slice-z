"""
Utility to generate PGF vector files from Python's Matplotlib plots to use in LaTeX documents.

Read more at https://github.com/knly/texfig
"""

import matplotlib as mpl
mpl.use('pgf')

from math import sqrt
default_width = 3.5 # in inches
default_width_2_col = 6.5 # in inches
default_width_pt = 522
default_ratio = (sqrt(5.0) - 1.0) / 2.0 # golden mean

FONT_SIZE = 8

mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "xelatex",
    'text.latex.unicode': True,
    "pgf.rcfonts": False,
    "font.family":"serif",
    "font.serif": ["Times New Roman"],
    "font.sans-serif": ["Helvetica"],
    "font.monospace": ["Courier", "Computer Modern Typewriter"],
    "figure.figsize": [default_width, default_width * default_ratio],
    "pgf.preamble": [
        # put LaTeX preamble declarations here
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        # macros defined here will be available in plots, e.g.:
        r"\newcommand{\vect}[1]{#1}",
        # You can use dummy implementations, since you LaTeX document
        # will render these properly, anyway.
    ],
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'savefig.pad_inches':0,
    'xtick.major.size' : 2,
    'ytick.major.size' : 2,
    'axes.linewidth' : 0.5,
    'image.cmap' : 'gray',
    'image.interpolation' : 'none'
})

import matplotlib.pyplot as plt


"""
Returns a figure with an appropriate size and tight layout.
"""
def figure(width=default_width, ratio=default_ratio, pad=0, tight_layout=False, *args, **kwargs):
    fig = plt.figure(figsize=(width, width * ratio), *args, **kwargs)
    
    if tight_layout:
        fig.set_tight_layout({
            'pad': pad
            })
    return fig

def figure1(width=default_width, ratio=default_ratio, pad=0, tight_layout=False, *args, **kwargs):
    fig = plt.figure(figsize=(width, width * ratio), *args, **kwargs)
    
    if tight_layout:
        fig.set_tight_layout({
            'pad': pad
            })
    return fig

"""
Returns subplots with an appropriate figure size and tight layout.
"""
def subplots(width=default_width, ratio=default_ratio, pad=0, *args, **kwargs):
    fig, axes = plt.subplots(figsize=(width, width * ratio), *args, **kwargs)
    #fig.set_tight_layout({
    #    'pad': pad
    #})
    return fig, axes


"""
Save both a PDF and a PGF file with the given filename.
"""
def savefig(filename, *args, **kwargs):
    plt.savefig(filename + "_pub"+ '.pdf', *args, **kwargs)
    plt.savefig(filename + "_pub" + '.pgf', *args, **kwargs)
    plt.close()
    
def savefig_pub(filename, *args, **kwargs):
    plt.savefig(filename + "_pub"+ '.pdf', dpi=1500, *args, **kwargs)
    file_path = filename + "_pub" + '.pdf'
    plt.savefig(filename + "_pub" + '.pgf', *args, **kwargs)
    print ("Saving figure: @" + str( file_path))
    plt.close()
