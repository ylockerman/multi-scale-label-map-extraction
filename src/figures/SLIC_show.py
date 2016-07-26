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


import os
import site
site.addsitedir(os.getcwd());

import multiprocessing 

import diffution_system.tile_map
import diffution_system.feature_space
import diffution_system.SLIC_gpu


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

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Slider, Button



def to_color_space(image):
    return color.rgb2lab(image)

def from_color_space(image):
    return color.lab2rgb(image.astype(np.float64))
    
    
constants = {
    'SLIC' : { 'm' : 20, 
              'total_runs' : 1,
              'max_itters' : 1000,
              'min_cluster_size': 25,
             },             

}

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
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
        print("Uses quit without selecting a file");
        sys.exit(0);
        
#    save_file_name = asksaveasfilename(filetypes=[('textured image','*.tximg')],
#                                 defaultextension=".tximg",
#                                 initialdir=data_management.get_full_data_path(''))
#
#    if(save_file_name is None):
#        import sys;
#        data_management.print_("Uses quit without selecting a sve file");
#        sys.exit(0);
    #file_name = data_management.get_full_data_path('For Paper/DSC_4930.jpg')
    
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
    tile_size = .02*max(image.shape[:2])

    

    #tile_map =  diffution_system.SLIC_gpu.SLICTileMap(image_lab,tile_size,**constants['SLIC'])     
    tile_map =  diffution_system.SLIC_gpu.SLICTileMap(image_lab,tile_size,**constants['SLIC'])
    
                                         
    def get_indicator(tm):
        indicator = np.zeros(image_lab.shape[:2]+(3,),np.float32)
        indicator_map = tm.copy_map_for_image(indicator)
        
        tm_color = tm.copy_map_for_image(image)        
        
        #data_management.add_array('diff_mat',diff_mat) steps,precondition_runs,accept_ratio
        for loc in xrange(len(tm)):
            key = tm.key_from_index(loc)
            im_data = np.reshape(tm_color[key],(-1,3))
            color = np.mean(im_data,axis=0)
            
            for c in xrange(color.shape[0]):
                indicator_map[key][:,:,c] = color[c]
            
        return indicator
        #return skimage.segmentation.find_boundaries(indicator)[:,:,0]
        
    def get_lines(tm):
        indicator = np.zeros(image_lab.shape[:2]+(1,),np.int32)
        indicator_map = tm.copy_map_for_image(indicator)
                
        #data_management.add_array('diff_mat',diff_mat) steps,precondition_runs,accept_ratio
        for loc in xrange(len(tm)):
            key = tm.key_from_index(loc)            
            indicator_map[key][:,:,0] = loc
            
        return skimage.segmentation.mark_boundaries(image,indicator[:,:,0])   
    
    
    fig = plt.figure()
    
    ax1 = plt.subplot(1,2,1, aspect='equal',frameon=False)
    ax1.imshow(get_lines(tile_map))
    ax1.axis('off')
    
    ax2 = plt.subplot(1,2,2, aspect='equal', sharex=ax1, sharey=ax1)
    ax2.imshow(get_indicator(tile_map))  
    ax2.axis('off')
    

    #fig.tight_layout(pad=0, w_pad=0, h_pad=0)
#
#    image_mark = np.copy(image)
#
#    image_mark[edges_our,0] = 0
#    image_mark[edges_our,1] = 1
#    image_mark[edges_our,2] = 0
#    
#    image_mark[edges_blure,0] = 1
#    image_mark[edges_blure,1] = 0
#    image_mark[edges_blure,2] = 0  
    
#    plt.imshow(image_mark);
    plt.tight_layout()
    plt.show()
    