# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MIT License

Copyright (c) 2016 Yale University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is based on the source for the paper 

"Multi-scale label-map extraction for texture synthesis"
in the ACM Transactions on Graphics (TOG) - Proceedings of ACM SIGGRAPH 2016,
Volume 35 Issue 4, July 2016 
by
Lockerman, Y.D., Sauvage, B., Allegre, 
R., Dischler, J.M., Dorsey, J. and Rushmeier, H.

http://graphics.cs.yale.edu/site/publications/multi-scale-label-map-extraction-texture-synthesis

If you find it useful, please consider giving us credit or citing our paper.   
-------------------------------------------------------------------------------

Created on Mon Jul 08 18:38:55 2013

@author: Yitzchak David Lockerman
"""
from __future__ import print_function

#if __name__ == '__main__':
#    import matplotlib
#    import mplh5canvas
#    mplh5canvas.set_log_level("warning")
     # set the log level before using module
     # default is 'warning', it is done here
     # to illustrate the syntax
#    matplotlib.use('module://mplh5canvas.backend_h5canvas')

#if __name__ == '__main__':
#    import matplotlib
#    matplotlib.use('wxAgg')

import os
import site
site.addsitedir(os.getcwd());

import multiprocessing 

import diffution_system.tile_map
import diffution_system.feature_space
import diffution_system.gui
import diffution_system.clustering_gpu as clustering
from diffution_system import diffusion_graph
import diffution_system.SLIC_gpu
import diffution_system.SLIC_multiscale


from gpu import opencl_tools


cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_algorithm = opencl_tools.cl_algorithm
elementwise = opencl_tools.cl_elementwise
cl_reduction = opencl_tools.cl_reduction
cl_scan = opencl_tools.cl_scan


import numpy as np


from skimage import io, color
import skimage.segmentation

import scipy.misc


import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Slider, Button

#matplotlib.rcParams['toolbar'] = 'None'

import sklearn.cluster

def to_color_space(image):
    return color.rgb2lab(image)

def from_color_space(image):
    return color.lab2rgb(image.astype(np.float64))
    
    
constants = {
    'SLIC_multiscale' : { 'm_base' : 20, 
                          'm_scale' : 1,
                          'total_runs' : 1,
                          'max_itters' : 1000,
                          'min_cluster_size': 25,
                          'sigma' : 3
                         },             
    'knn_graph_to_diffution' : {
                            'smothing_factor' : "intrinsic dimensionality",
                            'norm_type' : 'local'
                            },
    'optimisation_clustering' : {
                            'precondition_runs' : 55,
                            'steps' : 5,
                            'max_itter' : 40000,
                            'cutt_off' : 1e-20,
                            'r_guess_precondition_runs' : 2,
                        },
                                                
    
}

if __name__ == '__main__':
    multiprocessing.freeze_support()
    data_management.tee_out()
    
    import time
    import glob
    import os
    ctx = opencl_tools.get_a_context()
         
#    #Create an experiment
#    data_management.create_experiment("eage_aware_editing");
#    
#    #add the source files
#    source_files = glob.glob(os.path.join(os.path.dirname(__file__),'*.py'))
#    dest_files = [os.path.join('source_files',os.path.basename(fi)) 
#                                                    for fi in source_files ]
#    data_management.current_experiment().add_files(zip(source_files,dest_files))     
#    
#    source_files = glob.glob(os.path.join(os.path.dirname(__file__),'diffution_system','*.py'))
#    dest_files = [os.path.join('source_files','diffution_system',os.path.basename(fi)) 
#                                                     for fi in source_files ]
#    data_management.current_experiment().add_files(zip(source_files,dest_files))      
    
    
    #From http://stackoverflow.com/questions/3579568/choosing-a-file-in-python-simple-gui
    from Tkinter import Tk
    from tkFileDialog import askopenfilename, asksaveasfilename
    
    Tk().withdraw()
    file_name = askopenfilename(filetypes=[('image file','*.jpg *.png *.bmp *.tif')])

    if(file_name is None):
        import sys;
        data_management.print_("Uses quit without selecting a file");
        sys.exit(0);
        

    
    image = io.imread(file_name)/255.0
    image_lab = to_color_space(image)
    
    #turn grayscale to color
#    if( len(image.shape) == 2 ):
#        old_image = image
#        image = np.zeros(old_image.shape+(3,));
#        for ii in xrange(3):
#            image[:,:,ii] = old_image
    
    #add the orignal image to the file
    #data_management.add_image("orignal_image.jpg",image)
    
    #Create a tileset from the image, with the user selecting the scale
    min_tile_size = .02*max(image.shape[:2])#diffution_system.gui.scale_selector(image);
    max_tile_size = .2*max(image.shape[:2])#diffution_system.gui.scale_selector(image);    
    if(min_tile_size is None):
        import sys;
        #data_management.print_("User quit without selecting a scale");
        sys.exit(0);    
    
    

    #tile_map =  diffution_system.SLIC_gpu.SLICTileMap(image_lab,tile_size,**constants['SLIC'])     
    tile_map_multi_scale =  diffution_system.SLIC_multiscale.SLICMultiscaleTileMap(image_lab,min_tile_size,max_tile_size,**constants['SLIC_multiscale'])
    #tile_map_multi_scale = tile_map_multi_scale.copy_map_for_image(image)
    
    

    
    
        

    
    def get_diffution_graph(curent_scale):
        texturemap_scale = tile_map_multi_scale.get_single_scale_map(curent_scale)
        current_space = diffution_system.feature_space.ManyMomentFeatureSpace(5)
        discriptor = current_space.create_image_discriptor(texturemap_scale);
        #data_management.print_("Feature space size of %d" % discriptor.shape[1])
        
        use_gpu = False #discriptor.shape[1] > 15
            
        
        # Do the diffution on the cpu or gpu
        if use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_gpu(discriptor,256)
        if not use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_ann(discriptor,256)

        
        #data_management.print_( ("Ann time (0 if not used): %f"+
        #                         " gpu time (0 if not used): %f") %
        #                                           (tc-tb,tb-ta) );
#        smothing_factor = np.logspace(-6,6,100)
#        values = np.zeros(smothing_factor.shape)
#        dvalues = np.zeros(smothing_factor.shape)        
#        for idx in xrange(smothing_factor.shape[0]):
#            nearist,dist_sqr = knn_graph
#                
#            n_points = dist_sqr.shape[0]
#            #We need to construct a sparce matrix quicly, to do that we prealocate 
#            #an ij matrix and a data matrix, fill them, then convert to csr
#    
#        
#            weight_val = 0
#            dweight_val = 0
#            
#            #first_good_index = np.nonzero(np.all(dist_sqr>0,axis=0))[0][0]
#            good_vals = np.isfinite(dist_sqr[:,1])
#            #normalizer = np.sqrt(dist_sqr[:,first_good_index+6])
#            #normalizer[np.isinf(normalizer)] = 0
#            normalizer = np.ones((n_points))
#            
#            for r in xrange(n_points):
#                good_vals = np.nonzero(nearist[r,:]>=0)
#                f = np.exp(- (dist_sqr[r,good_vals]/(normalizer[r]*normalizer[nearist[r,:]] ))/(2*smothing_factor[idx]) )
#                weight_val += np.sum( f )
#                dweight_val += np.sum( ( (dist_sqr[r,good_vals]/(normalizer[r]*normalizer[nearist[r,:]]) )/(2*smothing_factor[idx]**2))*f )
#
#            values[idx] = weight_val
#            dvalues[idx] = dweight_val
#              
#        print(smothing_factor[np.argmax(smothing_factor*(dvalues)/weight_val)])
#        plt.figure(2)
#        #plt.plot(smothing_factor*(dvalues)/weight_val,np.log10(values));
#        plt.plot(np.log10(smothing_factor),np.log10(values));
#        #plt.plot(np.log10(smothing_factor),smothing_factor*(dvalues)/weight_val);
#        #plt.plot(np.log10(smothing_factor)[1::],np.diff(np.log10(values))/np.diff(np.log10(smothing_factor)),'o' );
#        
#        plt.plot(np.log10(smothing_factor),smothing_factor*(dvalues/(values) ) );
#        plt.plot( (np.log10(smothing_factor)[1::]+np.log10(smothing_factor)[:-1:])/2   ,
#                 np.diff(np.log10(values))/np.diff(np.log10(smothing_factor)),'o' );
        
        diff_mat = diffusion_graph.knn_graph_to_diffution_matrix(knn_graph,
                                            normalizer_in=np.linalg.norm(discriptor,axis=1),
                                             **constants['knn_graph_to_diffution'] )   
        
        return texturemap_scale,knn_graph, diff_mat, discriptor
                   

    all_scales =  tile_map_multi_scale.get_scales()  
    min_scale = min(all_scales)
    max_scale = max(all_scales)
    


    
    

    def disable_axies_for(ax):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)          
    
    class TextureSelector(object):
        
        def __init__(self):
            self.selected_location = None
            self.selected_scale = None
            self.knn_graph = None            
            self.diff_mat = None
            
            
            self.fig = plt.figure()
            
            self.number_of_step_cols= 5
            
            top_row_size = 3
            rows,cols = 6,self.number_of_step_cols
            middle_rows = rows - top_row_size - 1
            
            self._number_of_steps_rows = middle_rows
            self.number_of_steps = self.number_of_step_cols*self._number_of_steps_rows
            
            self.ax_image = plt.subplot2grid((rows,cols),(0,0),rowspan=top_row_size,colspan=cols/2)
            disable_axies_for(self.ax_image)
            
            self.ax_othr = plt.subplot2grid((rows,cols),(0,cols/2),rowspan=top_row_size,colspan=cols/2)
            #disable_axies_for(self.ax_othr)
            
            self.axs_diffution =[ plt.subplot2grid((rows,cols),(top_row_size+step_row,step_col),rowspan=1,colspan=1) 
                                                                            for step_row in xrange(self._number_of_steps_rows)
                                                                            for step_col in xrange(self.number_of_step_cols)
                                                                            ] 

            for ax in self.axs_diffution:
                disable_axies_for(ax)
            
            self.ax_scale = plt.subplot2grid((rows,cols),(rows-1,0),colspan=cols-1)
            self.ax_button = plt.subplot2grid((rows,cols),(rows-1,cols-1))
            disable_axies_for(self.ax_button)
            
            self.initial_image = np.zeros(image_lab.shape,np.float32)
            
            self.image_handel =  self.ax_image.imshow(image.astype(np.float32),interpolation='none')
            #self.slic_image = self.ax_slic.imshow(self.initial_image,interpolation='none')   
            
            
            self.diffution_images = [ ax_diffution.imshow(self.initial_image,interpolation='none')  
                                                for ax_diffution in  self.axs_diffution]            
            
            self.clic_ivent = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.cursor = Cursor(self.ax_image, useblit=True, color='red', linewidth=2 )
            
            initial_scale = (np.max(all_scales) + np.min(all_scales))/2
            self.scale_slider = Slider(self.ax_scale, 'Scale', np.min(all_scales), np.max(all_scales), valinit=initial_scale )
                                         
            self.scale_slider.on_changed(self.set_scale)    
            self.set_scale(initial_scale)
            
            self.fig.tight_layout()

                
        def diffution_indicator(self,indicator):
            #rgb_map = self.tile_map.copy_map_for_image(image.astype(np.float32))        
            
            #out_image = np.zeros(image_lab.shape,np.float32)
            #out_image_map = self.tile_map.copy_map_for_image(out_image) 
            
            #nn_av = 0
            #for loc in  xrange(indicator.shape[0]):
            #    key = self.tile_map.key_from_index(loc)
            #    out_image_map[key] = rgb_map[key]*indicator[loc]

            out_image = image.astype(np.float32)*indicator[self.index_indiator]

            return out_image
                
        def show_diffution_indicator_for_tile(self,tile):
            
            indicator = np.zeros(self.diff_mat.shape[0],np.float32)
            indicator[tile] = 1            
            
            scale_values = []
            self_value = []
            
            diff_mat = self.diff_mat.copy()
            #diff_mat.data[:] = 1
            
            #indicator_sums = np.zeros(self.diff_mat.shape[0],np.float32)
            
            for steps, imhandel in zip(range(self.number_of_steps),self.diffution_images ):
                print("Steps", steps)
                #indicator_sums += indicator
                #indicator_norm = indicator_sums/np.max(indicator_sums)
                
                indicator_norm = indicator/np.max(indicator)
                next_indicator = diff_mat.dot(indicator)
                
                out_image = self.diffution_indicator(indicator_norm)
                imhandel.set_data(out_image)
                
                scale_values.append(steps)
                self_value.append(np.sum( (next_indicator - indicator > 0 ) *indicator) )
                
                indicator = next_indicator
                        
            self.ax_othr.cla()
            #self.ax_othr.plot(scale_values,self_value)
            #self.ax_othr.set_ylim((0,1))
            self.ax_othr.plot(self.knn_graph[1][tile,:10] )
            print("self value", self.diff_mat[tile,tile] )
            print("discriptor", self.discriptor[tile,:] )
            
        def get_selected_tile(self):           
            if self.selected_location is None:
                return None
                
            posible_sp = self.tile_map.get_indexes_from_location(self.selected_location)
    
            #for idx,other_start_value in enumerate(posible_sp):
            if len(posible_sp) > 0:
                return posible_sp[0]        
        
            return None
        
        def update_image(self):
            selected_tile = self.get_selected_tile()
            if selected_tile is not None and self.selected_scale is not None:
                self.show_diffution_indicator_for_tile(selected_tile)
            self.fig.canvas.draw()

        def on_click(self,event):
            if(event.inaxes == self.ax_image and
                    event.button == 1 and 
                        0 <= event.ydata < image.shape[0] and 
                                0 <= event.xdata < image.shape[1]):
                            
                self.selected_location = (int(event.ydata),int(event.xdata))
    
                self.update_image()
            

            
        def set_scale(self,scale):
            self.selected_scale = scale
            
            (self.tile_map,self.knn_graph,
             self.diff_mat,self.discriptor) = get_diffution_graph(scale)
            
            self.index_indiator = np.zeros(image_lab.shape,np.int32)
            index_image_map = self.tile_map.copy_map_for_image(self.index_indiator) 
            
            #nn_av = 0
            for loc in  xrange(len(index_image_map)):
                key = self.tile_map.key_from_index(loc)
                index_image_map[key] = loc

            
            self.update_image()




    
    class Tevt(object):
        
        def __init__(self):
            self.button = 1
            self.xdata = np.random.choice(image.shape[0])
            self.ydata = np.random.choice(image.shape[1])
    
    
    ts = TextureSelector()
    plt.show(block=True)
#    for ii in xrange(600):
#        try:
#            on_click(Tevt())
#        except:
#            print("err")
#            pass
#    image_mod.data.release()
    #import sys
    #sys.exit(0)