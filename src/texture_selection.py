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
import scipy.sparse.linalg as splinalg
import scipy.optimize as spopt
import scipy.sparse as sparse

from skimage import io, color
import skimage.segmentation



import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Slider, Button

#matplotlib.rcParams['toolbar'] = 'None'
import sklearn.linear_model
import sklearn.cluster

import joblib

def to_color_space(image):
    return color.rgb2lab(image)

def from_color_space(image):
    return color.lab2rgb(image.astype(np.float64))
    
    
constants = {
    'SLIC_multiscale' : { 'm_base' : 20, 
                          'm_scale' : 1,
                          'total_runs' : 10,
                          'max_itters' : 1000,
                          'min_cluster_size': 15,
                          'sigma' : 3
                         },             
    'knn_graph_to_diffution' : {
                            'smothing_factor' : "intrinsic dimensionality",
                            'norm_type' : 'local'
                            },
    'optimisation_clustering' : {
                            'precondition_runs' : 75,
                            'max_itter' : 40000,
                            'cutt_off' : 1e-20,
                            'r_guess_precondition_runs' : 20,
                        },
                        
    'options' : { 'useNMF' : True,
                  'number_of_full_trys' : 4}
                        
    
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
        print "Uses quit without selecting a file";
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
    tile_map_multi_scale = tile_map_multi_scale.copy_map_for_image(image)

    def get_steps(diff_mat):
#        steps = int( 1 +  (diff_mat.shape[0]/128)^2 )
#        print("Steps selected",steps)     
#        return steps   
#######################     
#        diff_mat_ones = diff_mat.copy()
#        diff_mat_ones.data[:] = 1
#        
#        diff_mat_ones =  diffusion_graph.weight_matrix_to_diffution_matrix(diff_mat_ones)
#        
#        w_ones,v_ones = splinalg.eigs(diff_mat_ones,min(500,diff_mat.shape[0]-2) )
#        ratio_ones = np.sum(diff_mat_ones.diagonal())/np.sum(w_ones)
#        print("w_ones",np.sort(np.real(w_ones) ))
##        
        w,v = splinalg.eigs(diff_mat,min(500,diff_mat.shape[0]-2) )
        ratio = np.sum(diff_mat.diagonal())/np.sum(w)
        w_sorted = np.real(np.sort(w)[::-1])
        print("eigenvalues",w_sorted)
#####################        
#        print("ratio",np.sort(np.real(w))/np.sort(np.real(w_ones)))        
        #print(w,ratio)
#        
#        vid = np.arange(w.shape[0])
#        fitting_samples = max(w.shape[0]/4,3)
#
#        # Robustly fit linear model with RANSAC algorithm
#        model_ransac = sklearn.linear_model.RANSACRegressor(
#                            sklearn.linear_model.LinearRegression(),
#                            residual_threshold = .1,min_samples=.2 )
#                            
#        model_ransac.fit(vid[:fitting_samples, np.newaxis], np.log(w[:fitting_samples]) )
#        inlier_mask = model_ransac.inlier_mask_
#        print(model_ransac.estimator_.coef_)
#        print(model_ransac.estimator_.intercept_)
#
#        
#        # Predict data of estimated models
#        line_y_ransac = model_ransac.predict(vid[:, np.newaxis])
#
#        plt.figure(3)
#        plt.plot(vid,np.log(w))
#        plt.plot(vid,line_y_ransac)
#        plt.plot(vid[inlier_mask],np.log(w[inlier_mask]),'og')
#########################
#        steps_array = np.arange(200)
#        dof = np.sum( w[:,None]**steps_array[None,:] ,axis=0) 
#        ddof = np.sum( w[:,None]**steps_array[None,:]*np.log(w[:,None]) ,axis=0) 
#        
#        
#        
#        fitting_samples = max(steps_array.shape[0],3)
#
#        # Robustly fit linear model with RANSAC algorithm
#        model_ransac = sklearn.linear_model.RANSACRegressor(
#                            sklearn.linear_model.LinearRegression(),
#                            residual_threshold = .01,min_samples=.2 )
#                            
#        fit_array = np.zeros((steps_array.shape[0],2))
#        fit_array[:,0] = steps_array
#        fit_array[:,1] = steps_array**2
#        #fit_array[:,2] = steps_array**3
#        #fit_array[:,3] = steps_array**4
#        
#        model_ransac.fit(fit_array[:fitting_samples,:], np.log(dof[:fitting_samples]) )
#        inlier_mask = model_ransac.inlier_mask_
#        print(model_ransac.estimator_.coef_)
#        print(model_ransac.estimator_.intercept_)
#
#        
#        # Predict data of estimated models
#        line_y_ransac = model_ransac.predict(fit_array)
#
#        plt.figure(3)
#        plt.plot(steps_array,np.log(dof))
#        plt.plot(steps_array,line_y_ransac)
#        plt.plot(steps_array[inlier_mask],np.log(dof[inlier_mask]),'og')
#        
#        plt.figure(4)
#        plt.plot(steps_array,dof)        
#        
########################
#        
#        
#        #for steps in xrange(steps/2+4,steps*2):
#        #    best_F,best_G, _ = \
#        #        clustering.precondition_starting_point_multiple_runs(diff_mat,steps,constants['optimisation_clustering']['r_guess_precondition_runs'])
#        #    plt.scatter(steps,np.vdot(best_F.T,best_G));
#        #    plt.scatter(steps,np.sum(w**steps)*ratio,c='r');
#       
#        plt.figure(4)

#        
#        plt.plot(steps_array,np.log(ratio_ones*dof))
#        plt.plot(steps_array,np.log(ratio_ones*ddof),'r')   
#            #plt.plot(steps_array,ratio*np.maximum(dof[0]+(steps_array-steps_array[0])*ddof[0],0))   
#        
#        plt.show(block=True)
################################        
        
        #val_ratio = np.sort(np.real(w_ones))[-2]/np.sort(np.real(w))[-2]
        #steps = max(int(np.log(1e-3)/np.log(val_ratio)),0)+1 #max(np.argmax(np.nonzero(ratio_ones*ddof<-10)[0] ) ,0)# int(1/(-3*np.real(model_ransac.estimator_.coef_)) )
        steps = max(abs(int( np.log(.95*w_sorted[1])/np.log(w_sorted[1])  )) ,1)
        print("Steps selected",steps)     
        return steps
############################  
#        for steps in xrange(5,200,1):
#            for tries in xrange(3):
#                print("----------------------------------->",steps)
#                best_F,best_G, _ = \
#                    clustering.precondition_starting_point_multiple_runs(diff_mat,steps,constants['optimisation_clustering']['r_guess_precondition_runs'])
#                plt.figure(6)
#                plt.scatter(steps, np.linalg.norm(np.dot(best_G,best_F)-np.eye(best_F.shape[1]))/best_F.shape[1]**2,color='r');
#                plt.figure(7)
#                plt.scatter(steps, best_F.shape[1],color='r' );
#                
#                best_F,best_G, _ = \
#                    clustering.optimisation_clustering_projection(diff_mat,steps=steps,**constants['optimisation_clustering'])
#    
#                plt.figure(6)
#                plt.scatter(steps, np.linalg.norm(np.dot(best_G,best_F)-np.eye(best_F.shape[1])));
#                plt.figure(7)
#                plt.scatter(steps, best_F.shape[1] );
#            plt.pause(.5)
#            
#        plt.show(block=True)
############################
#        for steps in xrange(1,200,1):
#            for tries in xrange(6):
#                print("----------------------------------->",steps,tries)
#                #best_F,best_G, _ = \
#                #    clustering.precondition_starting_point_multiple_runs(diff_mat,steps,constants['optimisation_clustering']['r_guess_precondition_runs'])
#
#                best_F,best_G, _ = \
#                    clustering.optimisation_clustering_projection(diff_mat,steps=steps,**constants['optimisation_clustering'])
#    
#                #good_G = best_G > 0
#                #GlogG = np.zeros_like(best_G)
#                #GlogG[good_G] = best_G[good_G]*np.log2(best_G[good_G])
#    
#                jump_prob = np.dot(best_G,diff_mat.dot(best_F))
#                margin_w,margin_v = np.linalg.eig(jump_prob)    
#                
#                stationary_index = np.argmax(margin_w)
#                
#                statinary = margin_v[:,stationary_index]/np.sum(margin_v[:,stationary_index])
#                
#                joint_prob = jump_prob*statinary[None,:]
#                #good_joint = joint_prob > 1e-5
#                
#                #mutial_info = joint_prob*np.log2(joint_prob/(statinary[:,None]*statinary[None,:]))
#                #stainery_info = np.sum(-statinary*np.log2(statinary))
#                #print(mutial_info)
#                
#                #npmi = np.log2(joint_prob/(statinary[:,None]*statinary[None,:]))/(-np.log2(joint_prob))
#                #npmi[np.logical_not(good_joint)] = 0;
#                #print(npmi)
#                
#                plt.figure(6)
#                plt.scatter(steps,np.trace(joint_prob))# np.sum(mutial_info[good_joint])/stainery_info );
#                #plt.scatter(steps, best_F.shape[1],color='r' );
#            plt.pause(.5)
#            
#        plt.show(block=True)
        
    def get_FG_error(tilemap,discriptor):
        use_gpu = False #discriptor.shape[1] > 15
            
        
        # Do the diffution on the cpu or gpu
        if use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_gpu(discriptor,256)
        if not use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_ann(discriptor,256)

        
        #data_management.print_( ("Ann time (0 if not used): %f"+
        #                         " gpu time (0 if not used): %f") %
        #                                           (tc-tb,tb-ta) );
        

        diff_mat = diffusion_graph.knn_graph_to_diffution_matrix(knn_graph,
                                            normalizer_in=np.linalg.norm(discriptor,axis=1),
                                             **constants['knn_graph_to_diffution'] )        
        steps = get_steps(diff_mat)
#        dist_mat,steps  = diffusion_graph.knn_graph_to_weight_matrix(knn_graph,
#                                            normalizer_in=np.linalg.norm(discriptor,axis=1),
#                                             **constants['knn_graph_to_diffution'])
#    
#    
#        diff_mat= diffusion_graph.weight_matrix_to_diffution_matrix(dist_mat)
#         
        
#        for steps_ in xrange(30,100):
#            F,G,error = clustering.optimisation_clustering_projection(diff_mat,steps=steps_,**constants['optimisation_clustering'])
#            G_max =  np.argmax(G[:,:],axis = 0)
#            
#            texture_indicator = np.zeros(image_lab.shape[:2],np.int32)
#            texture_indicator_map = tile_map_multi_scale.copy_map_for_image(texture_indicator)
#    
#            for textrue_id in xrange(G.shape[0]):
#                for idx in np.nonzero(G_max == textrue_id)[0]:
#                    texture_indicator_map[tilemap.key_from_index(idx) ] = textrue_id
#                   
#            #plt.figure()
#            plt.imshow(texture_indicator)
#            plt.colorbar()
#            plt.title("Steps %d"%steps_)
#            plt.savefig("T:/Steps_%d.jpg"%steps_,bbox_inches='tight')
#            plt.clf()
            
            
        all_calls = [joblib.delayed(clustering.optimisation_clustering_projection)
                        (diff_mat,steps=steps,**constants['optimisation_clustering'])
                            for _ in xrange(constants['options']['number_of_full_trys'])]
        results = joblib.Parallel(n_jobs=2,verbose=10)(all_calls)


#        for F,G,error in results:
#            G_max =  np.argmax(G[:,:],axis = 0)
#            
#            texture_indicator = np.zeros(image_lab.shape[:2],np.int32)
#            texture_indicator_map = tile_map_multi_scale.copy_map_for_image(texture_indicator)
#    
#            for textrue_id in xrange(G.shape[0]):
#                for idx in np.nonzero(G_max == textrue_id)[0]:
#                    texture_indicator_map[tilemap.key_from_index(idx) ] = textrue_id
#                   
#            plt.figure()
#            plt.imshow(texture_indicator)
#            plt.colorbar()
#        
            
        all_Fs,all_Gs,texture_count = zip(* ( (F,G,G.shape[0]) for F,G,error in results ) )
  
        all_Fs = diffusion_graph.weight_matrix_to_diffution_matrix(np.hstack(all_Fs))   
        all_Gs = diffusion_graph.weight_matrix_to_diffution_matrix(np.vstack(all_Gs))
        cumlitive_texture_count = np.cumsum(texture_count)
        
        simular_map = np.dot(all_Gs,all_Fs)
        #simular_map = (simular_map+simular_map.T)/2

        #ap =  sklearn.cluster.AffinityPropagation(affinity='precomputed')
        #labels = ap.fit_predict(simular_map)
        #


        G_diff_mat = diffusion_graph.weight_matrix_to_diffution_matrix(simular_map)
        
        
        best_error = np.inf
        best_F_F = None
        best_G_G = None
        
        for idx in xrange(5):
            F_F,G_G,error = clustering.optimisation_clustering_projection(G_diff_mat,steps=2,r_guess_precondition_runs=600,precondition_runs=500)
        
            if error < best_error:
                best_error = error
                best_F_F = F_F
                best_G_G = G_G
        
            print(idx,error,best_error)
        
        F_F = best_F_F
        G_G = best_G_G
        
        labels = np.argmax(G_G[:,:],axis = 0)
                
        new_texture_count = np.max(labels)+1     
        projection_matrix = np.zeros( (new_texture_count,all_Gs.shape[0]),dtype=np.float32 )
        projection_matrix[(labels,np.arange(all_Gs.shape[0]))] = 1
        print(np.max(labels)+1,np.mean(texture_count))
        
#        best_return = None
#        best_error = 0
#        for result in results:
#            steps_to_use = steps
#            F,G,error = result
#            
#            if error < best_error:
#                best_return = F,G,error
#                best_error = error
#            print(">>>> steps: ",steps_to_use,"error: ",error, "best_error: ", best_error)
#            
#        F,G,error = best_return

        F = np.dot(all_Fs,F_F)#projection_matrix.T)
        G = np.dot(G_G,all_Gs)
        
        clustering_indicator =  np.argmax(G[:,:],axis = 0)
        
        cluster_distance_matrix = np.dot(G,F)
        
        cluster_distance_matrix = (cluster_distance_matrix+cluster_distance_matrix.T)/2

        G_max =  np.argmax(G[:,:],axis = 0)
        
        texture_indicator = np.zeros(image_lab.shape[:2],np.int32)
        texture_indicator_map = tile_map_multi_scale.copy_map_for_image(texture_indicator)

        for textrue_id in xrange(G.shape[0]):
            for idx in np.nonzero(G_max == textrue_id)[0]:
                texture_indicator_map[tilemap.key_from_index(idx) ] = textrue_id
           
           
#        plt.imshow(texture_indicator)
#        plt.colorbar()
#        plt.title("Steps %d"%steps)
#        plt.savefig("T:/size_%d_steps_%d.jpg"%(F.shape[0],steps),bbox_inches='tight')
#        plt.clf()
#        plt.figure()
#        plt.imshow(texture_indicator)
#        plt.title("Final")
#        plt.colorbar()
#        plt.show(block=True)
    
        return diff_mat,steps,clustering_indicator,cluster_distance_matrix
        
    def get_n_clusters(tilemap,discriptor):
        use_gpu = False #discriptor.shape[1] > 15
            
        
        # Do the diffution on the cpu or gpu
        if use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_gpu(discriptor,256)
        if not use_gpu:
            knn_graph =  diffusion_graph.build_knn_graph_ann(discriptor,256)

        
        #data_management.print_( ("Ann time (0 if not used): %f"+
        #                         " gpu time (0 if not used): %f") %
        #                                           (tc-tb,tb-ta) );
        
        
        diff_mat = diffusion_graph.knn_graph_to_diffution_matrix(knn_graph,
                                            normalizer_in=np.linalg.norm(discriptor,axis=1),
                                             **constants['knn_graph_to_diffution'] )   
                                             
        
        
        #steps = int(np.ceil(np.log(diff_mat.shape[0])))
        
      
        steps = get_steps(diff_mat)
        print(diff_mat.shape)
        
        best_F,best_G, _ = \
            clustering.precondition_starting_point_multiple_runs(diff_mat,steps,constants['optimisation_clustering']['r_guess_precondition_runs'])
            
        #print("dof",np.vdot(best_F.T,best_G) )
        return best_F.shape[1],  discriptor, diff_mat, steps      
      
        
    def do_clustering_at_level(texturemap_at_scale,lower_level_texture_indicator=None,old_cluster_distance_matrix=None):
        print("map len",len(texturemap_at_scale))

        if lower_level_texture_indicator is not None and False:
            lower_level_texture_count = np.max(lower_level_texture_indicator)+1
            texture_indicator_map = texturemap_at_scale.copy_map_for_image(texture_indicator)

            current_space = diffution_system.feature_space.BinCountFeatureSpace(lower_level_texture_count)
            discriptor = current_space.create_image_discriptor(texture_indicator_map);
            
            w,v = np.linalg.eigh(old_cluster_distance_matrix)
            discriptor = np.dot(discriptor,np.sqrt(w)[:,None]*v.T)
        else:
            current_space = diffution_system.feature_space.ManyMomentFeatureSpace(5)
            discriptor = current_space.create_image_discriptor(texturemap_at_scale);
        #data_management.print_("Feature space size of %d" % discriptor.shape[1])
        
        
        if constants['options']['useNMF']:
            diff_mat,steps,clustering,new_cluster_distance_matrix\
                                        = get_FG_error(texturemap_at_scale,discriptor)
            n_clusters = np.max(clustering)+1
        else:
            n_clusters,discriptor,diff_mat,steps = get_n_clusters(texturemap_at_scale,discriptor)
            #means = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters,batch_size=max(int(.25*len(texturemap_at_scale)),200),n_init=2000,reassignment_ratio=.2 )
            means = sklearn.cluster.KMeans(n_clusters=n_clusters,max_iter=1000,n_init=100,n_jobs=3 )
            means.fit(discriptor)
            clustering = means.labels_        
            new_cluster_distance_matrix = np.eye(n_clusters)
        
        return clustering, n_clusters, diff_mat, steps,new_cluster_distance_matrix
    
    def get_indicator_at_scale(curent_scale,lower_level_texture_indicator=None,old_cluster_distance_matrix=None):
        texturemap_scale = tile_map_multi_scale.get_single_scale_map(curent_scale)
        
        G_max, n_clusters, diff_mat,steps,new_cluster_distance_matrix =\
            do_clustering_at_level(texturemap_scale,lower_level_texture_indicator,old_cluster_distance_matrix)
        
        texture_indicator = np.zeros(image_lab.shape[:2],np.int32)
        texture_indicator_map = tile_map_multi_scale.copy_map_for_image(texture_indicator)

        for textrue_id in xrange(n_clusters):
            for idx in np.nonzero(G_max == textrue_id)[0]:
                texture_indicator_map[texturemap_scale.key_from_index(idx) ] = textrue_id
            
        return texture_indicator,texture_indicator_map,G_max, diff_mat,steps,new_cluster_distance_matrix

    all_scales =  tile_map_multi_scale.get_scales()  
    min_scale = min(all_scales)
    max_scale = max(all_scales)
    
    
    scale_list = []
    indicator = [] 
    indicator_lookup_list = []
    all_tile_maps = {}
    
    #Auto scale selection
    last_texture_indicator = None
    last_cluster_distance_matrix = np.eye(3)
    current_index = 0
    while(current_index < len(all_scales) ):   
        
        curent_scale = all_scales[current_index]
        print(curent_scale)
        scale_list.append(curent_scale)
        
        texture_indicator,texture_indicator_map,G_max,diff_mat,steps,new_cluster_distance_matrix= \
                            get_indicator_at_scale(curent_scale,last_texture_indicator,last_cluster_distance_matrix)
        indicator.append(texture_indicator)
        indicator_lookup_list.append(texture_indicator_map.get_single_scale_map(curent_scale))
#            textures_per_clusters = np.zeros((len(all_scales),3))*np.nan
#            clusters_with_more_then_min = np.zeros((len(all_scales)))*np.nan 
        
        last_texture_indicator = texture_indicator
        last_cluster_distance_matrix = new_cluster_distance_matrix
        
        
        all_tile_maps[curent_scale] = (texture_indicator_map.get_single_scale_map(curent_scale),G_max,diff_mat,steps)


        print("---------------------->",G_max.shape[0])
        if G_max.shape[0] < 128:
            break 

        
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
                
#                textures_per_clusters[mid_scale_index,0] = np.min(all_num)
#                textures_per_clusters[mid_scale_index,1] = np.mean(all_num)
#                textures_per_clusters[mid_scale_index,2] = np.max(all_num)
#                clusters_with_more_then_min[mid_scale_index] = np.sum(all_num > np.min(all_num) )/float(len(texturemap_at_scale))
            

#            plt.plot(all_scales,textures_per_clusters[:,0],'r*')
#            plt.plot(all_scales,textures_per_clusters[:,1],'g*')
#            plt.plot(all_scales,textures_per_clusters[:,2],'b*')
#            plt.figure()
#            plt.plot(all_scales,clusters_with_more_then_min,'*')
#            plt.show(block=True)
        
        if not(has_set):
            break

    #Add the max scale if needed        
    #if scale_list[-1] < max_scale:
    #    scale_list.append(max_scale)
        
    scale_list.reverse()
    indicator.reverse()
    indicator_lookup_list.reverse()
    print("Scales selected", scale_list)
#    
#    
#    for idx,scale in enumerate(all_scales):
#        tile_map = tile_map_multi_scale.get_single_scale_map(scale)
#        
#        #Create a simple descriptor from the tile map
#        current_space = diffution_system.feature_space.ManyMomentFeatureSpace(5)
#        discriptor = current_space.create_image_discriptor(tile_map);
#        #data_management.print_("Feature space size of %d" % discriptor.shape[1])
#        
#    
#        use_gpu = discriptor.shape[1] > 15
#            
#        
#        # Do the diffution on the cpu or gpu
#        ta = time.time()
#        if use_gpu:
#            knn_graph =  diffusion_graph.build_knn_graph_gpu(discriptor,256)
#        tb = time.time()
#        if not use_gpu:
#            knn_graph =  diffusion_graph.build_knn_graph_ann(discriptor,256)
#        tc = time.time()
#        
#        #data_management.print_( ("Ann time (0 if not used): %f"+
#        #                         " gpu time (0 if not used): %f") %
#        #                                           (tc-tb,tb-ta) );
#        
#        
#        diff_mat = diffusion_graph.knn_graph_to_diffution_matrix(knn_graph,
#                                             **constants['knn_graph_to_diffution'] )
#        #data_management.add_array('diff_mat',diff_mat) steps,precondition_runs,accept_ratio
#                     
#
#        scale_chart[idx] = float(diff_mat.shape[0])/diff_mat.data.shape[0]#np.dot(np.ones(diff_mat.shape[0]), diff_mat.dot(np.ones(diff_mat.shape[1])))/diff_mat.data.shape[0]
#        print('dt',np.dot(np.ones(diff_mat.shape[0]), diff_mat.dot(np.ones(diff_mat.shape[1]))),diff_mat.data.shape[0], diff_mat.shape[0]  )
#        
#        all_tile_maps[scale] = (tile_map,diff_mat)
#    
#    plt.figure()
#    plt.plot(all_scales,np.log(scale_chart),'*')
    

    

    
    class TextureSelector(object):
        
        def __init__(self):
            self.selected_location = None
            self.selected_scale = None
            self.G_max = None
            self.F = None
            self.G = None
            self.diff_mat = None
            
            self.fig = plt.figure()
            self.ax_image = plt.subplot2grid((13,2),(0,0),rowspan=6)
            self.ax_texture = plt.subplot2grid((13,2),(0,1),rowspan=6)
            self.ax_diffution = plt.subplot2grid((13,2),(6,0),rowspan=6)

            
            self.ax_scale = plt.subplot2grid((13,2),(12,0))
            self.ax_button = plt.subplot2grid((13,2),(12,1))
            
            self.initial_image = np.zeros(image_lab.shape,np.float32)
            
            self.image_handel =  self.ax_image.imshow(image.astype(np.float32))
            self.texture_image = self.ax_texture.imshow(self.initial_image)   
            self.diffution_image = self.ax_diffution.imshow(self.initial_image)               
            
            self.clic_ivent = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.cursor = Cursor(self.ax_image, useblit=True, color='red', linewidth=2 )
            
            initial_scale = (np.max(all_scales) + np.min(all_scales))/2
            self.scale_slider = Slider(self.ax_scale, 'Scale', np.min(scale_list), np.max(scale_list), valinit=initial_scale )
                                         
            self.scale_slider.on_changed(self.set_scale)    
            self.set_scale(initial_scale)
            
            self.bsave = Button(self.ax_button, 'Save')
            self.bsave.on_clicked(self.save_indicator)
            
        def show_indicator_for_tile(self,tile):
    #        indicator = np.zeros(diff_mat.shape[0],np.float32)
    #        indicator[tile] = 1
    #        
    #        steps = constants['optimisation_clustering']['steps']
    #        accept_ratio = constants['optimisation_clustering']['accept_ratio']
    #        indicator = diffusion_graph.do_diffusion(diff_mat,indicator,steps)
    #        good_points = np.nonzero(indicator >=  accept_ratio*max(indicator))[0]
                
            texture_index = self.G_max[tile]
            good_points = np.nonzero(self.G_max == texture_index)[0]
                
            #good_tile_set = np.zeros(F.shape[0],np.bool)
            
            rgb_map = self.tile_map.copy_map_for_image(image.astype(np.float32))        
            
            if len(good_points)>0:
                out_image = np.zeros(image_lab.shape,np.float32)
                out_image_map = self.tile_map.copy_map_for_image(out_image) 
                
                #nn_av = 0
                for loc in good_points: # xrange(F.shape[0]):
                    key = self.tile_map.key_from_index(loc)
                    #good_tile_set[loc] = True
                    out_image_map[key] = rgb_map[key]
                
                self.texture_image.set_data(out_image)
                
                
        def show_diffution_indicator_for_tile(self,tile):
            indicator = np.zeros(self.diff_mat.shape[0],np.float32)
            indicator[tile] = 1
            
            steps = self.steps#int(np.ceil(np.log(self.diff_mat.shape[0])/2)) #constants['optimisation_clustering']['steps']
            print("Steps",steps)
            indicator = diffusion_graph.do_diffusion(self.diff_mat,indicator,steps)
            indicator /= np.max(indicator)
            
            rgb_map = self.tile_map.copy_map_for_image(image.astype(np.float32))        
            
            out_image = np.zeros(image_lab.shape,np.float32)
            out_image_map = self.tile_map.copy_map_for_image(out_image) 
            
            #nn_av = 0
            for loc in  xrange(indicator.shape[0]):
                key = self.tile_map.key_from_index(loc)
                out_image_map[key] = rgb_map[key]*indicator[loc]
            
            self.diffution_image.set_data(out_image)
                        
           
       
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
                self.show_indicator_for_tile(selected_tile)
                self.show_diffution_indicator_for_tile(selected_tile)
                self.fig.canvas.draw()
            

        def on_click(self,event):
            global tile_map
            global F,G,error
            global G_max
            global selected_location,selected_scale
    
            if(event.inaxes == self.ax_image and
                    event.button == 1 and 
                        0 <= event.ydata < image.shape[0] and 
                                0 <= event.xdata < image.shape[1]):
                            
                self.selected_location = (int(event.ydata),int(event.xdata))
    
                self.update_image()
            

            
        def set_scale(self,scale):
            global tile_map
            global F,G,error
            global G_max
            global selected_scale
            
            
            self.selected_scale =min(all_tile_maps.keys(), key=lambda x:abs(x-scale)) 
            
            (self.tile_map,self.G_max,self.diff_mat,self.steps) = all_tile_maps[self.selected_scale]
    
            #F,G,error = clustering.optimisation_clustering_projection(diff_mat,**constants['optimisation_clustering'])
            #G_max = np.argmax(G[:,:],axis = 0)
            self.update_image()

    
        def save_indicator(self,event):
                global G_max,tile_map
                
                save_file_name = asksaveasfilename(filetypes=[('jpg','*.jpg')],
                                     defaultextension=".tximg")
    
                if(save_file_name is not None):
                    indicator = np.zeros(image_lab.shape[:2]+(1,),np.int32)
                    indicator_map = self.tile_map.copy_map_for_image(indicator)
                    
                    for loc in xrange(self.G_max.shape[1]):
                        key = self.tile_map.key_from_index(loc)
                        indicator_map[key] = self.G_max[loc]
    
                    indicator_image = np.zeros(indicator.shape[:2]+(3,),np.int32)
                    totalcolors = np.max(indicator)+1
                    a_step = 75
                    b_step = 75
                    c_step = 255.0/(totalcolors)
                    
                    a = 100;
                    b = 0;
                    c = 0;
                    for ii in xrange(totalcolors):
                        a += a_step
                        if a > 255:
                            a = 0
                            b+= b_step
                        if b > 255:
                            b = 0
                            c+= c_step
                        
                        indicator_image[indicator[:,:,0] == ii,0] = a
                        indicator_image[indicator[:,:,0] == ii,1] = b
                        indicator_image[indicator[:,:,0] == ii,2] = c
                    
    
                    skimage.io.imsave(save_file_name,indicator_image)
                    self.texture_image.set_data(indicator_image)
                    self.fig.canvas.draw()


    
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