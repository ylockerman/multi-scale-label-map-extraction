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

import scipy.sparse.linalg as splinalg


import scipy.io

import diffution_system.tile_map
import diffution_system.feature_space
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
import numpy.ma as ma

from skimage import io, color


import sklearn.cluster

import joblib
import time


#A number of defualt values
constants = {
    'SLIC_multiscale' : { 'm_base' : 20, 
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
    'optimisation_clustering' : {
                            'precondition_runs' : 55,
                            'max_itter' : 40000,
                            'cutt_off' : 1e-20,
                            'r_guess_precondition_runs' : 2,
                        }
                        
    
    
}



def prase_arguments():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
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
                raise argparse.ArgumentTypeError("%r is not a number, or a percent" % val)
                
        
        def __call__(self,rel_value):
            if self._is_relitive:
                return self.val/100.0 * rel_value
            else:
                return self.val

        
    parser.add_argument('input_image', nargs='?', type=str,   
                         help="The input image to run the algorithm on")
    parser.add_argument('output_file', nargs='?', type=str,  
                        help="The location to save the output file")
    
    parser.add_argument('--show_SLIC_gui' ,action="store_true",  
                        help="Will display a GUI to visualize the Multiscale SLIC before continuing")        

    parser.add_argument('--clustering_algorithum' ,default="kmeans",  
                        help="The clustering algorithum to use, can be NMF,kmeans, spectral")  
    
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
                                        
    SLIC_multiscale_args = parser.add_argument_group('SLIC Multiscale',"Parameters for the multiscale SLIC algorithm")
    SLIC_multiscale_args.add_argument('--m_base',type=float,
                                        help="The m value to use for the initial SLIC",
                                        default=constants['SLIC_multiscale']['m_base'])
    SLIC_multiscale_args.add_argument('--m_scale',type=float,
                                        help="The m value to use for the merging step",
                                        default=constants['SLIC_multiscale']['m_scale'])        
    SLIC_multiscale_args.add_argument('--total_runs',type=int,
                                        help="The number of times to run the initial SLIC",
                                        default=constants['SLIC_multiscale']['total_runs'])   
    SLIC_multiscale_args.add_argument('--SLIC_max_itters',type=int,
                                        help="The maximum iterations of the initial SLIC",
                                        default=constants['SLIC_multiscale']['max_itters']) 
    SLIC_multiscale_args.add_argument('--min_cluster_size',type=int,
                                        help="The minimum size of a SLIC superpixel. Anything smaller then this size will be greedily merged",
                                        default=constants['SLIC_multiscale']['min_cluster_size']) 
    SLIC_multiscale_args.add_argument('--sigma',type=float,
                                        help="The smoothing parameter to use for each stop of the multiscale SLIC. Smaller values will produce more acurite results, but will be slower",
                                        default=constants['SLIC_multiscale']['sigma']) 
                                         
    
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
                                             default='histogram_moments')                                                  
                                             
                                             
    histogram_moment_args = parser.add_argument_group('Histogram Moments Features',"Parameters for histagram moment features (must set feature_space to histogram_moments)")
                                             
    histogram_moment_args.add_argument('--moment_count',type=int,
                                             help="The number of moments to use for the metrix",
                                             default=5)
                                             
                                             
    gabor_feature_args = parser.add_argument_group('Gabor Filter Features',"Parameters for Gabor Filter features (must set feature_space to gabor_filters)")
                                             
    gabor_feature_args.add_argument('--gabor_octave_count',type=int,
                                             help="The number of octaves for the gabor filter",
                                             default=1)        
    gabor_feature_args.add_argument('--gabor_octave_jump',type=float,
                                             help="The size of each octave in dB",
                                             default=1)                                                      
    gabor_feature_args.add_argument('--gabor_lowest_central_freq',type=float,
                                             help="The smallest central frequency in cicles per pixel",
                                             default=None)  
    gabor_feature_args.add_argument('--gabor_lowest_central_freq_ratio',type=float,
                                             help="The smallest central frequency as proportion to the inverse of the scale. Will be ignored if gabor_lowest_central_freq is set.",
                                             default=10)                                                                                                      
    gabor_feature_args.add_argument('--gabor_angle_sensitivity',type=float,
                                             help="The angular sensitivity of each filter in radions",
                                             default=2*0.698132)    
   
    optimisation_clustering_args = parser.add_argument_group('clustering',"Parameters for the NMF clustering")
    optimisation_clustering_args.add_argument('--precondition_runs',type=int,
                                             help="The number of attempts to run the preconditioning algorithms",
                                             default=constants['optimisation_clustering']['precondition_runs'])  
    optimisation_clustering_args.add_argument('--steps',type=int,
                                             help="The power of the diffusion matrix to use. If not included the amount will be automaticly calulated.",
                                             default=None)     
    optimisation_clustering_args.add_argument('--NMF_max_itter',type=int,
                                             help="The maximum number of iterations of the gradient descent algorithm",
                                             default=constants['optimisation_clustering']['max_itter'])                  
    optimisation_clustering_args.add_argument('--cutt_off',type=float,
                                             help="The change of error needed to assume convergence",
                                             default=constants['optimisation_clustering']['cutt_off'])         
    optimisation_clustering_args.add_argument('--r_guess_precondition_runs',type=int,
                                             help="The number of full matrix runs used to guess r. Used to decide how many projection points are needed",
                                             default=constants['optimisation_clustering']['r_guess_precondition_runs'])  
    optimisation_clustering_args.add_argument('--total_NMF_boost_runs', type=int,
                                              help="The total number of times to run the NMF code before boosting.",
                                              default=6)                                                    
    optimisation_clustering_args.add_argument('--boosting_runs', type=int,
                                              help="The number of times to try boosting (set to zero to disable and just take the best).",
                                              default=5)                                                    

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
    
    constants['SLIC_multiscale']['m_base']            = args.m_base
    constants['SLIC_multiscale']['m_scale']           = args.m_scale
    constants['SLIC_multiscale']['total_runs']        = args.total_runs
    constants['SLIC_multiscale']['max_itters']        = args.SLIC_max_itters
    constants['SLIC_multiscale']['min_cluster_size']  = args.min_cluster_size    
    constants['SLIC_multiscale']['sigma']             = args.sigma
    
    constants['knn_graph_to_diffution']['smothing_factor'] = args.smothing_factor
    constants['knn_graph_to_diffution']['norm_type']       = args.norm_type  
    
    constants['optimisation_clustering']['precondition_runs']           = args.precondition_runs       
    constants['optimisation_clustering']['max_itter']                   = args.NMF_max_itter      
    constants['optimisation_clustering']['cutt_off']                    = args.cutt_off      
    constants['optimisation_clustering']['r_guess_precondition_runs']   = args.r_guess_precondition_runs       
    
    
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
        #data_management.print_("User quit without selecting a scale");
        sys.exit(0);
        
    start_tile_map_time = time.time()
    tile_map_multi_scale =  diffution_system.SLIC_multiscale.SLICMultiscaleTileMap(image_lab,tile_size,max_tile_size,**constants['SLIC_multiscale'])
    end_tile_map_time = time.time()    
    
    tile_map_time = end_tile_map_time - start_tile_map_time
    
    if args.show_SLIC_gui:
        tile_map_multi_scale.show_gui(image)
        
    
    tmp_save_file_name = save_file_name
                

    def make_debug_display_vector(tilemap):
        if not args.enable_NMF_debug_vector:
            return None
        
        import matplotlib.pyplot as plt

        def debug_display_vector(F,G,title):
            index_image = np.zeros(image_lab.shape[:2],np.int32)
            index_image_map = tilemap.copy_map_for_image(index_image) 
            
            
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
                    
                    rgb_map = tilemap.copy_map_for_image(image.astype(np.float32))        
                    
                    out_image = np.zeros(image_lab.shape,np.float32)
                    out_image_map = tilemap.copy_map_for_image(out_image) 
                    
                    #nn_av = 0
                    for loc in  xrange(indicator.shape[0]):
                        key = tilemap.key_from_index(loc)
                        out_image_map[key] = rgb_map[key]*indicator[loc]
                    
                    show_image_obj.set_data(out_image)
                    fig.canvas.draw()
            click_evnt = fig.canvas.mpl_connect('button_press_event', on_click)
            
            return click_evnt
        return debug_display_vector
        
    def create_feature_space(curent_scale):
        print("--------------------------------------------")
        print("Using feature space %s" % args.feature_space)
        print("--------------------------------------------")        
        
        if args.feature_space == 'histogram_moments':
            return diffution_system.feature_space.ManyMomentFeatureSpace(args.moment_count)
        elif args.feature_space == 'gabor_filters':
            if args.gabor_lowest_central_freq:
                gabor_lowest_central_freq = args.gabor_lowest_central_freq
            else:
                gabor_lowest_central_freq = args.gabor_lowest_central_freq_ratio/curent_scale
            print("Frequency used",gabor_lowest_central_freq)

            return diffution_system.feature_space.GaborFeatureSpace(octave_count=args.gabor_octave_count,
                                                                    octave_jump=args.gabor_octave_jump,
                                                                    lowest_centrail_freq=gabor_lowest_central_freq,
                                                                    angle_sensitivity=args.gabor_angle_sensitivity)
        elif args.feature_space == 'quadtree':
            return diffution_system.feature_space.QuadTreeFeatureSpace()
        else:
            raise Exception('Unknown feature space %s' % args.feature_space)
            
    def get_steps(diff_mat):
        
        if args.steps is not None:
            steps = args.steps
        else:
            w,v = splinalg.eigs(diff_mat,min(500,diff_mat.shape[0]-2) )
            w_sorted = np.real(np.sort(w)[::-1])
            
            first_good_index = np.nonzero(w_sorted < .9999)[0][0]
            steps = max(abs(int( np.log(.95*w_sorted[first_good_index])/np.log(w_sorted[first_good_index])  )) ,1)
            print("Steps selected",steps)     
        return steps

                
                
    def get_FG_error(tilemap,curent_scale):
        current_space = create_feature_space(curent_scale)
        discriptor = current_space.create_image_discriptor(tilemap);
        #data_management.print_("Feature space size of %d" % discriptor.shape[1])
        

        
        use_gpu = False #discriptor.shape[1] > 15
            
        
        # Do the diffution on the cpu or gpu
        if use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_gpu(discriptor,args.knn)
        if not use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_ann(discriptor,args.knn)

        
        #data_management.print_( ("Ann time (0 if not used): %f"+
        #                         " gpu time (0 if not used): %f") %
        #                                           (tc-tb,tb-ta) );
        
        
        diff_mat = diffusion_graph.knn_graph_to_diffution_matrix(knn_graph,
                                             **constants['knn_graph_to_diffution'] )        
        
        steps=get_steps(diff_mat)
        
        debug_display_vector=make_debug_display_vector(tilemap)
        all_calls = [joblib.delayed(clustering.optimisation_clustering_projection)
                        (diff_mat,steps=steps,**constants['optimisation_clustering'])
                            for _ in xrange(args.total_NMF_boost_runs)]
                                
        n_jobs = 1 if debug_display_vector is None else 1
        results = joblib.Parallel(n_jobs=n_jobs,verbose=10)(all_calls)
            
        if args.boosting_runs > 0:
            all_Fs,all_Gs,texture_count = zip(* ( (F,G,G.shape[0]) for F,G,error in results ) )
      
            all_Fs = diffusion_graph.weight_matrix_to_diffution_matrix(np.hstack(all_Fs))   
            all_Gs = diffusion_graph.weight_matrix_to_diffution_matrix(np.vstack(all_Gs))
            
            simular_map = np.dot(all_Gs,all_Fs)
    
            G_diff_mat = diffusion_graph.weight_matrix_to_diffution_matrix(simular_map)
            
            
            best_error = np.inf
            best_F_F = None
            best_G_G = None
            
            for idx in xrange(args.boosting_runs):
                F_F,G_G,error = clustering.optimisation_clustering_projection(G_diff_mat,steps=2,r_guess_precondition_runs=600,precondition_runs=500)
            
                if error < best_error:
                    best_error = error
                    best_F_F = F_F
                    best_G_G = G_G
            
            
            F_F = best_F_F
            G_G = best_G_G
            
            F = np.dot(all_Fs,F_F)#projection_matrix.T)
            G = np.dot(G_G,all_Gs)
        else:
            best_error = np.inf
            best_F = None
            best_G = None
            
            for F,G,error in results:
                if error < best_error:
                    best_error = error
                    best_F = F
                    best_G = G            
                F = best_F
                G = best_G
        return F,G

        
    def get_n_clusters(tilemap,curent_scale):
        current_space = create_feature_space(curent_scale)
        discriptor = current_space.create_image_discriptor(tilemap);
        #data_management.print_("Feature space size of %d" % discriptor.shape[1])
        
    
        use_gpu = False #discriptor.shape[1] > 15
            
        
        # Do the diffution on the cpu or gpu
        if use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_gpu(discriptor,args.knn)
        if not use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_ann(discriptor,args.knn)

        
        #data_management.print_( ("Ann time (0 if not used): %f"+
        #                         " gpu time (0 if not used): %f") %
        #                                           (tc-tb,tb-ta) );
        
        
        diff_mat = diffusion_graph.knn_graph_to_diffution_matrix(knn_graph,
                                             **constants['knn_graph_to_diffution'] )        
        best_F,_, _ = \
            clustering.precondition_starting_point_multiple_runs(diff_mat,
                                                                 get_steps(diff_mat),
                                                                 constants['optimisation_clustering']['r_guess_precondition_runs'],
                                                                 debug_display_vector=make_debug_display_vector(tilemap))
        return best_F.shape[1],  discriptor, diff_mat                         
        

 

        
        
    def do_clustering_at_level(texturemap_at_scale,curent_scale):
        if args.clustering_algorithum == 'NMF':
            F,G = get_FG_error(texturemap_at_scale,curent_scale)
            n_clusters = G.shape[0]
            G_max = np.argmax(G[:,:],axis = 0)
        else:
            n_clusters,discriptor,diff_mat = get_n_clusters(texturemap_at_scale,curent_scale)
            #means = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters,batch_size=max(int(.25*len(texturemap_at_scale)),200),n_init=2000,reassignment_ratio=.2 )
            
            if args.clustering_algorithum == 'kmeans':
                means = sklearn.cluster.KMeans(n_clusters=n_clusters,max_iter=1000,n_init=100,n_jobs=3 )
                means.fit(discriptor)
                G_max = means.labels_ 
            elif args.clustering_algorithum == 'spectral':
                spectral = sklearn.cluster.SpectralClustering(n_clusters=n_clusters,affinity='precomputed',n_init=100 )
                spectral.fit(diff_mat)
                G_max = spectral.labels_ 
            else:
                raise Exception("Unknown clustering algorithum %s"%args.clustering_algorithum)
        return G_max, n_clusters

    #This is a debug code
    def get_slic_indicator(tm):
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
            
    def get_indicator_at_scale(curent_scale):
        texturemap_scale = tile_map_multi_scale.get_single_scale_map(curent_scale)
        G_max, n_clusters = do_clustering_at_level(texturemap_scale,curent_scale)
        
        texture_indicator = np.zeros(image_lab.shape[:2],np.int32)
        texture_indicator_map = tile_map_multi_scale.copy_map_for_image(texture_indicator)

        for textrue_id in xrange(n_clusters):
            for idx in np.nonzero(G_max == textrue_id)[0]:
                texture_indicator_map[texturemap_scale.key_from_index(idx) ] = textrue_id
            
        return texture_indicator,texture_indicator_map

        

    start_scale_select_time = time.time()
    
    all_scales =  tile_map_multi_scale.get_scales()  
    min_scale = min(all_scales)
    max_scale = max(all_scales)
    
    
    scale_list = []
    indicator = [] 
    indicator_lookup_list = []
    
    #This is the defualt setting, use it if none is set
    if (args.number_of_levels is None and args.scale_decay_rate is None):
        #Auto scale selection
    
        current_index = 0
        while(current_index < len(all_scales) ):   
            
            curent_scale = all_scales[current_index]
            print(curent_scale)
            scale_list.append(curent_scale)
            
            texture_indicator,texture_indicator_map = get_indicator_at_scale(curent_scale)
            indicator.append(texture_indicator)
            indicator_lookup_list.append(texture_indicator_map)

            #Use a binery search to find the jump point
            min_scale_index = current_index+1
            max_scale_index = len(all_scales) -1
            
            has_set = False
            while min_scale_index + 1 < max_scale_index:
                mid_scale_index = (max_scale_index+min_scale_index) /2
                scale_at_mid = all_scales[mid_scale_index]
                
                texturemap_at_scale = texture_indicator_map.get_single_scale_map(scale_at_mid)
                all_num = np.zeros(len(texturemap_at_scale))
                
                for tid,super_pixel in enumerate(texturemap_at_scale):
                    all_num[tid] = len(np.unique(texturemap_at_scale[super_pixel]))
                    
                if np.min(all_num) >= 2 and not has_set:
                    max_scale_index = mid_scale_index
                else:
                    min_scale_index = mid_scale_index
                    
                
                if max_scale_index == min_scale_index + 1:
                    current_index = max_scale_index
                    has_set = True
                    
            if not(has_set):
                break

            
        scale_list.reverse()
        indicator.reverse()
        indicator_lookup_list.reverse()
        print("Scales selected", scale_list)

    else:
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
        
    end_scale_select_time  = time.time()
    scale_select_time = end_scale_select_time - start_scale_select_time

        

    start_segmintation_time = time.time()

    if len(indicator) == 0: #If we did not cache the indcators for each level, calulate them
        for scale in scale_list:
            texture_indicator,texture_indicator_map = get_indicator_at_scale(scale)
            indicator.append(texture_indicator)
            indicator_lookup_list.append(texture_indicator_map)

        
    class TextureTreeNode(object):
        
        def __init__(self,keyset,scale):
            self.children = []
            self.parent = None
            self.keyset = keyset
            self.scale = scale
        

    #Convert a searese of flat segmentations to the tree
    def segmentation_to_tree(texturemap,scale_indicator_list):
        if len(scale_indicator_list) == 0:
            return []
            
        scale,indicator,indicator_map = scale_indicator_list[0]
        n_clusters = np.max(indicator) + 1
        texturemap_at_scale = texturemap.get_single_scale_map(scale)
        
        all_nodes = []
        
        for textrue_id in xrange(n_clusters):
            keys = [key for key in texturemap_at_scale 
                            if ma.all(indicator_map[key]==textrue_id) ]
            print('---->',textrue_id,'/',n_clusters,':',len(keys))
            if len(keys) > 0:
                new_node = TextureTreeNode(keys,scale)
                new_texturemap = texturemap.get_submap(keys)
                        
                try:
                    new_node.children = segmentation_to_tree(new_texturemap,scale_indicator_list[1:])
                except:
                    import traceback
                    traceback.print_exc()
                    
                all_nodes.append(new_node)
        
        return all_nodes

    texture_tree = segmentation_to_tree(tile_map_multi_scale,zip(scale_list,indicator,indicator_lookup_list))
    end_segmintation_time = time.time()
    
    segmintation_time = end_segmintation_time - start_segmintation_time
    
    total_time = end_segmintation_time - start_tile_map_time

    rgb_map = tile_map_multi_scale.copy_map_for_image(image.astype(np.float32)) 


    def recusive_build_tree(list_of_nodes):      
        return [
                     { 
                      'scale' : node.scale,
                      'list_of_base_superpixels' : np.concatenate( [ks.list_of_sp for ks in node.keyset] ),
                      'children' : recusive_build_tree(node.children)
                     } for node in list_of_nodes 
               ]
                   
    SLIC_raw,HSLIC_raw = tile_map_multi_scale.get_raw_data()
    texture_tree_raw = recusive_build_tree(texture_tree)
    
    

    
    #Create a raw list of superpixels for each texture on each level
    raw_texture_list = []
    for scale_id,scale in enumerate(scale_list):
        local_indicator = indicator[scale_id]
        indicator_map = indicator_lookup_list[scale_id]
        texturemap_at_scale = tile_map_multi_scale.get_single_scale_map(scale)
        
        n_clusters = np.max(local_indicator) + 1


        
        texture_makeup_list = [
                                  np.concatenate(
                                      [key.list_of_sp 
                                          for key in texturemap_at_scale 
                                             if ma.all(indicator_map[key]==texture_id) 
                                      ] ) for texture_id in xrange(n_clusters)
                              ] 

        raw_texture_list.append(                
                       { 
                        'scale' : scale,
                        'list_of_base_superpixels' : texture_makeup_list
                       }
                     )
    
       
    
    scipy.io.savemat(   save_file_name, 
                        {'image_shape' : image.shape,
                         'base_SLIC_rle' : output_RLE(SLIC_raw),
                         'HSLIC' : HSLIC_raw,
                         'texture_lists' : raw_texture_list,
                         'texture_tree' : texture_tree_raw,
                         'SLIC_time' : tile_map_time,
                         'total_time' : total_time
                        }, 
                        appendmat = False 
                    )
