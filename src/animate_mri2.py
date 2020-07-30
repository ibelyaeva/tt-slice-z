from mayavi import mlab
from tvtk.tools import visual
from numpy import array
from mayavi.mlab import *

import tensorflow as tf
import numpy as np
np.random.seed(0)
from nilearn import image
import copy
from nilearn import plotting
from scipy import stats
from mayavi.mlab import *
from tvtk.pyface.movie_maker import MovieMaker

import math
import glob
import  moviepy.editor as mpy
import os

from mayavi.core.ui.api import SceneEditor, MayaviScene, \
                                MlabSceneModel

class MRIAnimator(object):
    
    def __init__(self, folder, title, x_ref, x_true, n=0, x_hat=None, x_miss=None, observed_ratio=1):
        self.folder = folder
        self.title = title
        self.x_ref = x_ref
        self.x_true = x_true
        self.x_hat = x_hat
        self.x_miss = x_miss
        self.observed_ratio = observed_ratio
        self.n = n     
        self.initialize()
       
       
    def initialize(self):
        
        self.set_missing_ratio()
                
        #self.animation_file_prefix = 'anim' + 'missing_ratio'+ str(self.missing_ratio)
        self.animation_file_prefix = str('anim_' + 'missing_ratio'+ str(self.missing_ratio) + '_' + str('%05d.png'))
            
        self.missing_file_fig_id =  str(self.folder) + "/" + "missing_ratio_" + str(self.missing_ratio)
        self.x_true_file_fig_id =  str(self.folder) + "/" + "x_true_" + str(self.missing_ratio)
        self.x_hat_file_fig_id =  str(self.folder) + "/" + "x_hat_" + str(self.missing_ratio)
        self.animation_file = self.missing_file_fig_id  +'.gif'
        
       
    def set_missing_ratio(self):
        
        if self.observed_ratio is not None:
            missing_ratio = self.floored_value((1.0 - self.observed_ratio),2)
            self.missing_ratio = str(missing_ratio)
        else:
            self.missing_ratio = self.floored_value(0,2)
        return self.missing_ratio
    
    def create_movie_maker(self):
        self.MovieMaker = MovieMaker(record=True, directory=self.folder, filename=self.animation_file)
            
    
    def get_volume(self, data, n):
        return np.array(image.index_img(data, n).get_data())
       
    
    def get_slice(self,index, data):
        if index == 'x':
            res = data[0:data.shape[0],:,:]
        elif index == 'y':
                res = data[:,0:data.shape[1],:]
        else:
            res = data[:,:,0:data.shape[2]]
            return res
    
    def generate_scene(self, data, n):
        volume = self.get_volume(data, n)
        x = np.linspace(0,data.shape[0],data.shape[0], endpoint=True, dtype=int)
        y = np.linspace(0,data.shape[1],data.shape[1], endpoint=True, dtype=int)
        z = np.linspace(0,data.shape[2],data.shape[2], endpoint=True, dtype=int)
        
        xv, yv, zv = np.meshgrid(x, y, z, indexing = 'ij', sparse=False)
        x2v, y2v, z2v = np.meshgrid(x, y, z, indexing = 'ij', sparse=False)
        dfdx, dfdy, dfdz = np.gradient(volume)
        
        f = mlab.figure(12, fgcolor=(.0, .0, .0), bgcolor=(1.0, 1.0, 1.0))
        
        
        #countour_sf = mlab.contour3d(xv, yv, zv, volume, contours=7, opacity=0.5, colormap='jet')
        #vectors = mlab.quiver3d(x2v, y2v, z2v, dfdx, dfdy, dfdz, mode='arrow',scale_mode='vector', mask_points=8, opacity=0.8, colormap='jet')
        #mlab.outline(countour_sf, color=(0.7, .7, .7))
        image_plane = mlab.pipeline.image_plane_widget(volume,
                            plane_orientation='z_axes',
                            slice_index=38,transparent=True, opacity = 0.8
                        )
        image_source = image_plane.mlab_source
        slice_range = np.linspace(data.shape[2],0, endpoint=True, dtype=int)
        title = mlab.title(self.title, size = 8, color=(0,0,0))
        title.property.font_size = 8
        
        
        return f, image_source, image_plane, slice_range
    
    @mlab.show
    @mlab.animate(delay=250, ui=True)
    def _animate_volume(self, data, n):
        f, image_source, image_plane, slice_range = self.generate_scene(data, n)
        
        while 1:
            for i in slice_range:
                image_plane.ipw.slice_index = i
                yield
                
    def animate_volume(self, data, n):
        self._animate_volume(data, n)
    
    def animate_volume_by_axis_z(self, data, n):
        self. _animate_volume_by_axis_z(data, n)
            
            
    def formatted_percentage(self, value, digits):
        format_str = "{:." + str(digits) + "%}"
    
        return format_str.format(value) 
    
    def save_anim_to_file(self, data, n):
        
        f, image_source, image_plane, slice_range = self.generate_scene(data, n)
        zz1 = np.linspace(data.shape[2],0, endpoint=True, dtype=int)
        
        image_plane.scene.movie_maker.directory=self.folder
        #image_plane.scene.movie_maker.filename = str('anim%05d.png')
        image_plane.scene.movie_maker.filename = self.animation_file_prefix
        
        with image_plane.scene.movie_maker.record_movie() as mm:
            for i in zz1:
                image_plane.ipw.slice_index = i
                mm.animation_step()
                
            file_name = self.missing_file_fig_id  + str('.tiff')
            mm.scene.save_tiff(file_name)
            file_items = self.collect_files()
            print "File List Size: " + str(len(file_items))
            self.save_gif(file_items)
            
    def collect_files(self):
        files = []
            
        print "Folder: " + str(self.folder)
        for f in glob.iglob(self.folder + '/movie*/*.png'):
            files.append(str(f))
        return files
    
    def save_gif(self, items):
        clip = mpy.ImageSequenceClip(items, fps=15)
        clip.write_gif(self.animation_file)
        print "Saving animation @: " + str(self.animation_file)
        
    def floored_value(self,val, digits):
        val *= 10 ** (digits + 2)
        return '{1:.{0}f}'.format(digits, math.floor(val) / 10 ** digits)
        