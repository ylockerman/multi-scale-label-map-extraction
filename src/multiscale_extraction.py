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

import scipy.io

from  diffution_system import hierarchical_SLIC
from diffution_system import label_map
from diffution_system import region_map

from gpu import opencl_tools


cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_algorithm = opencl_tools.cl_algorithm
elementwise = opencl_tools.cl_elementwise
cl_reduction = opencl_tools.cl_reduction
cl_scan = opencl_tools.cl_scan


import numpy as np
import numpy.ma as ma

from skimage import io, color

import time


#A number of defualt values
#These will be updated from the command arguments
constants = {
    'hierarchical_SLIC' : { 'm_base' : 20, 
                          'm_scale' : 1,
                          'total_runs' : 10,
                          'max_itters' : 1000,
                          'min_cluster_size': 25,
                          'sigma' : 3
                         },             
    'knn_graph_to_diffution' : {
                            'smothing_factor' : "intrinsic dimensionality",
                            'norm_type' : 'local'
                            },
    'feature_space' : {
                          'feature_space_name' : 'histogram_moments',
                          'moment_count' : 5,
                          'gabor_octave_count' : 1,
                          'gabor_octave_jump' : 1,
                          'gabor_lowest_central_freq': None,
                          'gabor_lowest_central_freq_ratio': 10,
                          'gabor_angle_sensitivity': 2*0.698132,
                          'quadtree_number_of_levels': 3
                        },
    'stochastic_NMF' : {
                            'precondition_runs' : 55,
                            'max_itter' : 40000,
                            'cutt_off' : 1e-20,
                            'r_guess_precondition_runs' : 2,
                        },
    'NMF_boosted' : {
                            'total_NMF_boost_runs': 6,
                            'boosting_calulation_runs' : 5
                        }
                            
}

#an argument type that can handel a percent 
class PercentArgument(object):
    
    def __init__(self,val):
        val = val.strip()
        
        if val[-1] == '%':
            self._is_relitive = True
            val = val[:-1]
        else:
            self._is_relitive = False
            
        try:
            self.val = float(val)
        except:
            import argparse
            raise argparse.ArgumentTypeError("%r is not a number, or a percent" % val)
            
    
    def __call__(self,rel_value):
        if self._is_relitive:
            return self.val/100.0 * rel_value
        else:
            return self.val


def prase_arguments():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    

        
    parser.add_argument('input_image', nargs='?', type=str,   
                         help="The input image to run the algorithm on")
    parser.add_argument('output_file', nargs='?', type=str,  
                        help="The location to save the output file")
    
    parser.add_argument('--show_SLIC_gui' ,action="store_true",  
                        help="Will display a GUI to visualize the Hierarchical SLIC before continuing")        

    parser.add_argument('--clustering_algorithum' ,default="kmeans",  
                        help="The clustering algorithum to use, can be NMF,NMF-boosted, kmeans, spectral")  
                        
    parser.add_argument('--show_labels_gui' ,action="store_true",  
                        help="Will display a GUI to visualize the hierarchical labels before continuing")        

    parser.add_argument('--enable_NMF_debug_vector' ,action="store_true",  
                        help="Displaies debug vectors for the NMF based code")  
                        
                        
    scale_args = parser.add_argument_group('Scale Selection',"Parameters to select the scales to use")
    scale_args.add_argument('--base_scale',type=PercentArgument,default='2%',nargs='*',
                                help="The base (smallest) scale. The initial SLIC super pixels will calculated at this level."
                                "Can be a percentage (i.e. 5%%), which indicates the number will be a percent of the maximum image dimension."
                                "If multiple values are provided, the maximum value is used")        
    scale_args.add_argument('--max_scale',type=PercentArgument,default='30%',
                                help="The highest (largest) scale. The merging algorithm will stop at this scale."
                                "Can be a percentage (i.e. 5%%), which indicates the number will be a percent of the maximum image dimension")        

    
    
    intermediate_levels = scale_args.add_mutually_exclusive_group()
    intermediate_levels.add_argument('--number_of_levels',type=int,
                                        help="The number of logarithmically spaced scales")        
    intermediate_levels.add_argument('--scale_decay_rate',type=float,
                                        help="The decay rate used to space scales")        
    intermediate_levels.add_argument('--auto-scale', action='store_true',
                                        help="Automaticly select scales",default=True)                                                    
                                        
    hierarchical_SLIC_args = parser.add_argument_group('SLIC Multiscale',"Parameters for the multiscale SLIC algorithm")
    hierarchical_SLIC_args.add_argument('--m_base',type=float,
                                        help="The m value to use for the initial SLIC",
                                        default=constants['hierarchical_SLIC']['m_base'])
    hierarchical_SLIC_args.add_argument('--m_scale',type=float,
                                        help="The m value to use for the merging step",
                                        default=constants['hierarchical_SLIC']['m_scale'])        
    hierarchical_SLIC_args.add_argument('--total_runs',type=int,
                                        help="The number of times to run the initial SLIC",
                                        default=constants['hierarchical_SLIC']['total_runs'])   
    hierarchical_SLIC_args.add_argument('--SLIC_max_itters',type=int,
                                        help="The maximum iterations of the initial SLIC",
                                        default=constants['hierarchical_SLIC']['max_itters']) 
    hierarchical_SLIC_args.add_argument('--min_cluster_size',type=int,
                                        help="The minimum size of a SLIC superpixel. Anything smaller then this size will be greedily merged",
                                        default=constants['hierarchical_SLIC']['min_cluster_size']) 
    hierarchical_SLIC_args.add_argument('--sigma',type=float,
                                        help="The smoothing parameter to use for each stop of the multiscale SLIC. Smaller values will produce more acurite results, but will be slower",
                                        default=constants['hierarchical_SLIC']['sigma']) 
                                         
    
    knn_graph_to_diffution_args = parser.add_argument_group('KNN graph',"Parameters for the knn graph and diffusion matrix")
    knn_graph_to_diffution_args.add_argument('--smothing_factor',
                                             help="The relative exponential decay rate for creation of the similarity matrix. In the paper this is hardcoded to 1.",
                                             default=constants['knn_graph_to_diffution']['smothing_factor'])        
    knn_graph_to_diffution_args.add_argument('--norm_type',choices=['local','global'],
                                             help="method of normalizing the graph",
                                             default=constants['knn_graph_to_diffution']['norm_type'])          
    knn_graph_to_diffution_args.add_argument('--knn',type=int,
                                             help="The number of neghbors to use for the knn graph. ",
                                             default=256)    
    knn_graph_to_diffution_args.add_argument('--feature_space',type=str,
                                             help="which feature space to use. Options are histogram_moments, gabor_filters, and quadtree ",
                                             default=constants['feature_space']['feature_space_name'])                                                  
                                             
                                             
    histogram_moment_args = parser.add_argument_group('Histogram Moments Features',"Parameters for histagram moment features (must set feature_space to histogram_moments)")
                                             
    histogram_moment_args.add_argument('--moment_count',type=int,
                                             help="The number of moments to use for the metrix",
                                             default=constants['feature_space']['moment_count'])
                                             
                                             
    gabor_feature_args = parser.add_argument_group('Gabor Filter Features',"Parameters for Gabor Filter features (must set feature_space to gabor_filters)")
                                             
    gabor_feature_args.add_argument('--gabor_octave_count',type=int,
                                             help="The number of octaves for the gabor filter",
                                             default=constants['feature_space']['gabor_octave_count'])        
    gabor_feature_args.add_argument('--gabor_octave_jump',type=float,
                                             help="The size of each octave in dB",
                                             default=constants['feature_space']['gabor_octave_jump'])                                                      
    gabor_feature_args.add_argument('--gabor_lowest_central_freq',type=float,
                                             help="The smallest central frequency in cycles per pixel",
                                             default=constants['feature_space']['gabor_lowest_central_freq'])  
    gabor_feature_args.add_argument('--gabor_lowest_central_freq_ratio',type=float,
                                             help="The smallest central frequency as proportion to the inverse of the scale. Will be ignored if gabor_lowest_central_freq is set.",
                                             default=constants['feature_space']['gabor_lowest_central_freq_ratio'])                                                                                                      
    gabor_feature_args.add_argument('--gabor_angle_sensitivity',type=float,
                                             help="The angular sensitivity of each filter in radions",
                                             default=constants['feature_space']['gabor_angle_sensitivity'])    
   
    quad_tree_args = parser.add_argument_group('Quad Tree Features',"Parameters for Quad Tree features (must set feature_space to quadtree)")
    quad_tree_args.add_argument('--quadtree_number_of_levels',type=float,
                                             help="The number of levels in the quad tree",
                                             default=constants['feature_space']['quadtree_number_of_levels'])                                                      

   
   
    stochastic_NMF_args = parser.add_argument_group('NMF',"Parameters for the stochastic NMF clustering")
    stochastic_NMF_args.add_argument('--precondition_runs',type=int,
                                             help="The number of attempts to run the preconditioning algorithms",
                                             default=constants['stochastic_NMF']['precondition_runs'])  
    stochastic_NMF_args.add_argument('--steps',type=int,
                                             help="The power of the diffusion matrix to use. If not included the amount will be automaticly calulated.",
                                             default=None)     
    stochastic_NMF_args.add_argument('--NMF_max_itter',type=int,
                                             help="The maximum number of iterations of the gradient descent algorithm",
                                             default=constants['stochastic_NMF']['max_itter'])                  
    stochastic_NMF_args.add_argument('--cutt_off',type=float,
                                             help="The change of error needed to assume convergence",
                                             default=constants['stochastic_NMF']['cutt_off'])         
    stochastic_NMF_args.add_argument('--r_guess_precondition_runs',type=int,
                                             help="The number of full matrix runs used to guess r. Used to decide how many projection points are needed",
                                             default=constants['stochastic_NMF']['r_guess_precondition_runs'])  
                                             
    stochastic_NMF_args = parser.add_argument_group('NMF Boosted',"Parameters for the Boosted stochastic NMF clustering")                                             
    stochastic_NMF_args.add_argument('--total_NMF_boost_runs', type=int,
                                              help="The total number of times to run the NMF code before boosting.",
                                              default=constants['NMF_boosted']['total_NMF_boost_runs'])                                                    
    stochastic_NMF_args.add_argument('--boosting_calulation_runs', type=int,
                                              help="The number of times to try boosting (set to zero to disable and just take the best).",
                                              default=constants['NMF_boosted']['boosting_calulation_runs'])                                                    

    parser.add_argument_group()
    
    return parser.parse_args()


def to_color_space(image):
    return color.rgb2lab(image)

def from_color_space(image):
    return color.lab2rgb(image.astype(np.float64))
        
def output_RLE(image):
    out = [] 

    for line in xrange(image.shape[0]):
        place = 1
        run_lenght = 1
        value = image[line,0]
        
        while place < image.shape[1]:
            if value != image[line,place]:
                out.append( np.array( [run_lenght,value] ) )
                run_lenght = 1
                value = image[line,place]
            else:
                run_lenght += 1
                
            place += 1
        
        #output the final run
        out.append( np.array( [run_lenght,value] ) )
    out = np.vstack(out)
    return out
    
    
if __name__ == '__main__':    
    multiprocessing.freeze_support()

    args = prase_arguments()
    
    constants['hierarchical_SLIC']['m_base']            = args.m_base
    constants['hierarchical_SLIC']['m_scale']           = args.m_scale
    constants['hierarchical_SLIC']['total_runs']        = args.total_runs
    constants['hierarchical_SLIC']['max_itters']        = args.SLIC_max_itters
    constants['hierarchical_SLIC']['min_cluster_size']  = args.min_cluster_size    
    constants['hierarchical_SLIC']['sigma']             = args.sigma
    
    constants['knn_graph_to_diffution']['smothing_factor'] = args.smothing_factor
    constants['knn_graph_to_diffution']['norm_type']       = args.norm_type  
    
    constants['feature_space']['feature_space_name']              = args.feature_space
    constants['feature_space']['moment_count']                    = args.moment_count
    constants['feature_space']['gabor_octave_count']              = args.gabor_octave_count 
    constants['feature_space']['gabor_octave_jump']               = args.gabor_octave_jump
    constants['feature_space']['gabor_lowest_central_freq']       = args.gabor_lowest_central_freq
    constants['feature_space']['gabor_lowest_central_freq_ratio'] = args.gabor_lowest_central_freq_ratio
    constants['feature_space']['gabor_angle_sensitivity']         = args.gabor_angle_sensitivity
    constants['feature_space']['quadtree_number_of_levels']       = args.quadtree_number_of_levels

    
    constants['stochastic_NMF']['precondition_runs']           = args.precondition_runs       
    constants['stochastic_NMF']['max_itter']                   = args.NMF_max_itter      
    constants['stochastic_NMF']['cutt_off']                    = args.cutt_off      
    constants['stochastic_NMF']['r_guess_precondition_runs']   = args.r_guess_precondition_runs   
    
    constants['NMF_boosted']['total_NMF_boost_runs']        = args.total_NMF_boost_runs      
    constants['NMF_boosted']['boosting_calulation_runs']    = args.boosting_calulation_runs        
    
    ctx = opencl_tools.get_a_context()

    
    #From http://stackoverflow.com/questions/3579568/choosing-a-file-in-python-simple-gui
    from Tkinter import Tk
    from tkFileDialog import askopenfilename, asksaveasfilename
    
    Tk().withdraw()
    
    #if constants['knn_graph_to_diffution']['smothing_factor'] != 1:
    #    if args.input_image is None or args.output_file is None: #If their will be user input, show it as a message
    #        tkMessageBox.showinfo(title="Warning", message="smothing_factor is hard coded as 1 in the paper. The results will not be representtitive of the paper!")
   #     else: #just print it
    #        print("Warning smothing_factor is hard coded as 1 in the paper. The results will not be representtitive of the paper!")
    
    if args.input_image is None:
        file_name = askopenfilename(filetypes=[('image file','*.jpg *.png *.bmp *.tif')])
    
        if(file_name is None or file_name == ""):
            import sys;
            print("Uses quit without selecting a file");
            sys.exit(0);
    else:
        file_name = args.input_image
        
           
    if args.output_file is None:
        save_file_name = asksaveasfilename(filetypes=[('mat','*.mat')],
                             defaultextension=".htex")
    
        if(save_file_name is None or file_name == ""):
            import sys;
            print("Uses quit without selecting a file");
            sys.exit(0);
    else:
        save_file_name = args.output_file


    image = io.imread(file_name)/255.0
    if len(image.shape) == 2:
        image = np.concatenate((image[:,:,None],image[:,:,None],image[:,:,None]),axis=2)
    if image.shape[2] > 3:
        image = image[:,:,:3]
    image_lab = to_color_space(image)
    relitive_value = max(image.shape[:2])


    #Create a tileset from the image, with the user selecting the scale
    if isinstance(args.base_scale,list):
        tile_size = max([bsf(relitive_value) for bsf in args.base_scale])
    else:
        tile_size = args.base_scale(relitive_value)#diffution_system.gui.scale_selector(image);
    
    print("Base Scale Chosen",tile_size)
    
    max_tile_size = args.max_scale(relitive_value)
    if(tile_size is None):
        import sys;
        print("User quit without selecting a scale");
        sys.exit(0);
        
    start_tile_map_time = time.time()
    tile_map_multi_scale =  hierarchical_SLIC.HierarchicalSLIC(image_lab,
                                                               tile_size,
                                                               max_tile_size,
                                                               **constants['hierarchical_SLIC'])
    end_tile_map_time = time.time()    
    
    tile_map_time = end_tile_map_time - start_tile_map_time
    
    if args.show_SLIC_gui:
        tile_map_multi_scale.show_gui(image)
        
    
    tmp_save_file_name = save_file_name
                

    def make_debug_display_vector(superpixel_map):
        if not args.enable_NMF_debug_vector:
            return None
        
        import matplotlib.pyplot as plt

        def debug_display_vector(F,G,title):
            index_image = np.zeros(image_lab.shape[:2],np.int32)
            index_image_map = superpixel_map.copy_map_for_image(index_image) 
            
            
            indicator = np.argmax(G,0)
            for loc in xrange(indicator.shape[0]):
                key = index_image_map.key_from_index(loc)
                index_image_map[key] = indicator[loc];
                
            fig = plt.figure() 
            
            main_ax = plt.subplot(1,2,1)
            plt.imshow(index_image);
            plt.colorbar();
            main_ax.set_title(title)    
            
            
            show_ax = plt.subplot(1,2,2)
            show_image_obj = show_ax.imshow(image);
            
            def on_click(event):
                if(event.inaxes == main_ax and
                            event.button == 1 and 
                            0 <= event.ydata < index_image.shape[0] and 
                                    0 <= event.xdata < index_image.shape[1]):
                                        
                    indicator = G[index_image[event.ydata,event.xdata],:]
                    
                    rgb_map = superpixel_map.copy_map_for_image(image.astype(np.float32))        
                    
                    out_image = np.zeros(image_lab.shape,np.float32)
                    out_image_map = superpixel_map.copy_map_for_image(out_image) 
                    
                    #nn_av = 0
                    for loc in  xrange(indicator.shape[0]):
                        key = superpixel_map.key_from_index(loc)
                        out_image_map[key] = rgb_map[key]*indicator[loc]
                    
                    show_image_obj.set_data(out_image)
                    fig.canvas.draw()
            click_evnt = fig.canvas.mpl_connect('button_press_event', on_click)
            
            return click_evnt
        return debug_display_vector

    start_scale_select_time = time.time()
    

    
    #If we have a set set of scales record them
    if (args.number_of_levels is not None or 
                    args.scale_decay_rate is not None):
        scale_list = []
        
        all_scales =  tile_map_multi_scale.get_scales()  
        min_scale = min(all_scales)
        max_scale = max(all_scales)

        if args.scale_decay_rate is not None:
            decay_rate = args.scale_decay_rate
            number_of_levels = np.inf;
        else:
            number_of_levels = args.number_of_levels;
            decay_rate = np.power(min_scale/max_scale,1.0/number_of_levels)   
            
        print("Decay rate",decay_rate)

        scale_at = max_scale
        while scale_at > min_scale:
            scale_list.append(scale_at)
            scale_at *= decay_rate
        
        if len(scale_list) > number_of_levels:
            scale_list = scale_list[:number_of_levels]
    else:
        scale_list = None
        
        
    start_segmintation_time = time.time()
    build_label_map_prams = {
      'knn' : args.knn,
      'steps' : args.steps,
      'clustering_algorithum'            : args.clustering_algorithum,
      'r_guess_precondition_runs'        : args.r_guess_precondition_runs,
      'feature_space_constants'          : constants['feature_space'],
      'knn_graph_to_diffution_constants' : constants['knn_graph_to_diffution'],
      'stochastic_NMF_constants'         : constants['stochastic_NMF'],
      'NMF_boosted_constants'            : constants['NMF_boosted'],
      'make_debug_display_vector'        : make_debug_display_vector
    }
    
    label_map_dict, aditinal_info_dict =\
                label_map.build_label_map_stack(tile_map_multi_scale,
                                  build_label_map_prams=build_label_map_prams)
                                                         
    hierarchical_labels = region_map.hierarchical_region_map_from_stack(label_map_dict)

    end_segmintation_time = time.time()

    
    segmintation_time = end_segmintation_time - start_segmintation_time
    
    total_time = end_segmintation_time - start_tile_map_time
    
    if args.show_labels_gui:
        hierarchical_labels.show_gui(image);
    
    SLIC_raw,HSLIC_raw = tile_map_multi_scale.get_raw_data()
    _ , texture_tree_raw = hierarchical_labels.get_raw_data()
    
    

    
    #Create a raw list of superpixels for each texture on each level
    raw_texture_list = []
    for scale,label_at_scale in label_map_dict.iteritems():
        texture_makeup_list = [
                                 label_at_scale.get_atomic_keys(key)
                                                  for key in label_at_scale
                              ] 

        scale_texture_data_dict = { 
                            'scale' : scale,
                            'list_of_atomic_superpixels' : texture_makeup_list
                           };
        scale_texture_data_dict.update(aditinal_info_dict[scale])
        
        raw_texture_list.append(scale_texture_data_dict)
    
       
    
    scipy.io.savemat(   save_file_name, 
                        {'image_shape' : image.shape,
                         'atomic_SLIC_rle' : output_RLE(SLIC_raw),
                         'HSLIC' : HSLIC_raw,
                         'texture_lists' : raw_texture_list,
                         'texture_tree' : texture_tree_raw,
                         'SLIC_time' : tile_map_time,
                         'total_time' : total_time
                        }, 
                        appendmat = False 
                    )
