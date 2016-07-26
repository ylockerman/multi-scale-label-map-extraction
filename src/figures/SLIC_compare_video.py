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
from __future__ import unicode_literals
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


import matplotlib
#matplotlib.verbose.set_level('debug')
#matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
#matplotlib.rcParams['animation.ffmpeg_args'] = ['-report', '-q:v', '1']
matplotlib.rcParams['animation.ffmpeg_args'] = ['-q:v', '1']
matplotlib.rcParams['animation.bitrate'] = -1
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True

#matplotlib.rcParams['verbose.level'] = 'debug'
matplotlib.use('Agg')

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import multiprocessing 



import diffution_system.tile_map
import diffution_system.feature_space
import diffution_system.gui
import diffution_system.SLIC_gpu
import diffution_system.SLIC_multiscale


from gpu import opencl_tools
from gpu import gpu_image_filter

cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_algorithm = opencl_tools.cl_algorithm
elementwise = opencl_tools.cl_elementwise
cl_reduction = opencl_tools.cl_reduction
cl_scan = opencl_tools.cl_scan


import numpy as np


from skimage import io, color
import skimage.segmentation


import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Slider, Button


import run_existing_SLIC


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

}
m_regular_SLIC = 1

if __name__ == '__main__':
    multiprocessing.freeze_support()
    #data_management.tee_out()
    
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

    if(file_name is None or len(file_name) == 0 ):
        import sys;
        print("Uses quit without selecting a file");
        sys.exit(0);
        

        
    save_file_name = asksaveasfilename(filetypes=[('mp4','*.mp4')],
                         defaultextension=".mp4")

    if(save_file_name is None or len(save_file_name) == 0):
        import sys;
        print("Uses quit without selecting a file");
        sys.exit(0);
        
    copy_wright_info = raw_input("Please enter copyright info >")
    
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
    min_tile_size = .01*max(image.shape[:2])#diffution_system.gui.scale_selector(image);
    max_tile_size = .3*max(image.shape[:2])#diffution_system.gui.scale_selector(image);    
    if(min_tile_size is None):
        import sys;
        #data_management.print_("User quit without selecting a scale");
        sys.exit(0);    
    
    

    #tile_map =  diffution_system.SLIC_gpu.SLICTileMap(image_lab,tile_size,**constants['SLIC'])     
    tile_map_multi_scale =  diffution_system.SLIC_multiscale.SLICMultiscaleTileMap(image_lab,min_tile_size,max_tile_size,**constants['SLIC_multiscale'])
    
    scale_list = np.linspace(min_tile_size,max_tile_size,50)
    
    
    
  
                                         
    def get_indicator(tm):
        if isinstance(tm,np.ndarray):
            edges_plain = np.zeros_like(image)

            counts = np.bincount(tm.ravel())
            for c in xrange(image.shape[2]):
                vals = np.bincount(tm.ravel(),image[:,:,c].ravel())
                edges_plain[:,:,c] = (vals/counts)[tm]
                
            return edges_plain
        else:
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

    
    
    
    
    queue = cl.CommandQueue(ctx,properties=opencl_tools.profile_properties)    
    
    image_gpu = cl_array.to_device(queue,image_lab.astype(np.float32))
    image_gpu_buffer = cl_array.empty_like(image_gpu)
    sigma = constants['SLIC_multiscale']['sigma']
    
    #Create the blure kernal
    range_vales = np.arange(-(3*sigma),3*sigma + 1);
    filter_vals = np.exp(-range_vales**2/(2*sigma**2));
    
    row_filter = filter_vals.reshape(-1,1);
    col_filter = filter_vals.reshape(1,-1);
    blure_filter = gpu_image_filter.SeparableImageFilter(image.shape,row_filter,
                                                         col_filter,queue=queue)
    
    S_current = min_tile_size

    tile_map = None
    tile_map_plain = None
    tile_map_blue = None
    image_blure = None
    edges_our = None
    edges_plain = None
    edges_blure = None
    
    def set_scale(scale):
        global tile_map, tile_map_plain, tile_map_blue 
        global image_gpu, image_gpu_buffer,image_blure, S_current
        global edges_our, edges_plain, edges_blure
        tile_map = tile_map_multi_scale.get_single_scale_map(scale)
#        tile_map_plain = diffution_system.SLIC_gpu.SLICTileMap(image_lab,scale,
#                                                                m=constants['SLIC_multiscale']['m_base'],
#                                                                max_itters=constants['SLIC_multiscale']['max_itters'],
#                                                                min_cluster_size=constants['SLIC_multiscale']['min_cluster_size']
#                                                              )  
        nosp = len(tile_map)
        tile_map_plain = run_existing_SLIC.get_SLIC_regions(image,m_regular_SLIC,nosp)
        if(S_current > scale):
            #Restart if we are over blured
            S_current = min_tile_size
            image_gpu.set(image_lab.astype(np.float32),queue)
            
        while(S_current < scale):
            blure_filter(image_gpu,image_gpu_buffer)
            image_gpu, image_gpu_buffer = image_gpu_buffer,image_gpu
        
            S_current += sigma;

        image_blure = image_gpu.get(queue)
#        tile_map_blue = diffution_system.SLIC_gpu.SLICTileMap(image_blure,scale,
#                                                                m=constants['SLIC_multiscale']['m_scale'],
#                                                                max_itters=constants['SLIC_multiscale']['max_itters'],
#                                                                min_cluster_size=constants['SLIC_multiscale']['min_cluster_size']
#                                                              )
        tile_map_blue = run_existing_SLIC.get_SLIC_regions(from_color_space(image_blure),m_regular_SLIC,nosp)                                                  
        edges_our =  get_indicator(tile_map)
        edges_plain = get_indicator(tile_map_plain)
        edges_blure = get_indicator(tile_map_blue)
    
    set_scale(min_tile_size)
    
    fig = plt.figure(figsize=(192, 108),dpi=10)
    txt = fig.suptitle("Scale: %d Pixels"%min_tile_size,size=300,fontdict={'weight' : 'heavy'})
    
    ax1 = plt.subplot(2,2,1, aspect='equal',frameon=False)
    image_blure_hdl = ax1.imshow(from_color_space(image_blure).astype(np.float32))
    ax1.set_title('Blurred Image',size=300,fontdict={'verticalalignment': 'bottom','weight' : 'heavy'})
    ax1.axis('off')
    
    ax2 = plt.subplot(2,2,2, aspect='equal', sharex=ax1)
    edges_plain_hdl = ax2.imshow(edges_plain)  
    ax2.set_title('Original SLIC Algorithm',size=300,fontdict={'verticalalignment': 'bottom','weight' : 'heavy'})
    ax2.axis('off')
    
    ax3 = plt.subplot(2,2,3, aspect='equal', sharex=ax1)
    edges_blure_hdl = ax3.imshow(edges_blure)
    ax3.set_title('Blur then SLIC',size=300,fontdict={'verticalalignment': 'bottom','weight' : 'heavy'})
    ax3.axis('off')
    
    ax4 = plt.subplot(2,2,4, aspect='equal', sharex=ax1)
    edges_our_hdl = ax4.imshow(edges_our)
    ax4.set_title('Our Hierarchical SLIC',size=300,fontdict={'verticalalignment': 'bottom','weight' : 'heavy'})
    ax4.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.02)
    
    if len(copy_wright_info) > 0:
        fig.text(0.05,.005,'Image © '+copy_wright_info,size=100,fontdict={'verticalalignment': 'bottom'})
    import matplotlib.animation as animation
    
    # Set up formatting for the movie files
    print(animation.writers.list())
    Writer = animation.writers['ffmpeg']
    
    writer = Writer(fps=5)#bitrate=32000)#fps=5)#, metadata=dict(artist='Me'), bitrate=32000)

    def show_scale(scale):
        set_scale(scale)
        fig.suptitle("Scale: %d Pixels"%scale,size=300,fontdict={'weight' : 'heavy'})
        
        image_blure_hdl.set_data(from_color_space(image_blure).astype(np.float32))
        edges_plain_hdl.set_data(edges_plain)
        edges_blure_hdl.set_data(edges_blure)
        edges_our_hdl.set_data(edges_our)
        
    line_ani = animation.FuncAnimation(fig, show_scale,scale_list, repeat=False)
    line_ani._fig.canvas.mpl_disconnect(line_ani._first_draw_id)
    line_ani._first_draw_id=None
    line_ani.save(save_file_name, writer=writer,dpi=10)
        
    #plt.show()