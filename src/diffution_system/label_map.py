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
Created on Fri Aug 19 14:37:43 2016

@author: Yitzchak David Lockerman
"""

import feature_space
import stochastic_NMF_gpu as NMF
import diffusion_graph
import region_map

import numpy as np

import sklearn.cluster
                    
defualt_feature_space_constants = {
                                    'feature_space_name' : 'histogram_moments',
                                    'moment_count':  5
                                  }
defualt_knn_graph_to_diffution_constants = {
                                'smothing_factor' : "intrinsic dimensionality",
                                'norm_type' : 'local'
                            }
                                  
defualt_stochastic_NMF_constants = {
                            'precondition_runs' : 55,
                            'max_itter' : 40000,
                            'cutt_off' : 1e-20,
                            'total_NMF_boost_runs': 6,
                            'boosting_calulation_runs' : 5
                        }
default_NMF_boosted_constants = {
                            'total_NMF_boost_runs': 6,
                            'boosting_calulation_runs' : 5
                        }                 
    
def build_label_map(superpixels_at_scale,
                    curent_scale,
                    clustering_algorithum = 'kmeans',
                    knn =128, 
                    steps=None,
                    r_guess_precondition_runs = 2,
                    feature_space_constants = None,
                    knn_graph_to_diffution_constants = None,
                    stochastic_NMF_constants=None,
                    NMF_boosted_constants = None,
                    make_debug_display_vector=None):
    """
    Builds a single scale label-map from a single scale superpixel map. 
    
    Parameters
    ----------
    superpixels_at_scale: SLIC or HSLIC 
        The mapping of superpixels for the scale to compute the label map. 
    curent_scale: float
        The current scale in pixels.  
    clustering_algorithum: str, optional 
        The clustering algorithm to use.  Can be one of:  'NMF', 'NMF-boosted', 
        'kmeans', or 'spectral'
    knn: int , optional 
        The number of k nearest neighbors to use when computing the similarity 
        graph. 
    steps: int , optional
        The power of the diffusion matrix to use for factorization or guessing 
        the number of labels. If not provided, it will be estimated. 
    r_guess_precondition_runs: int , optional
        The number of times to run the precondition algorithm without an 
        improving error before stopping when estimating r. This only effects if 
        projection is used, not the final number of labels. 
    feature_space_constants: dict 
        The keyword parameters to pass to feature_space.create_feature_space
    knn_graph_to_diffution_constants: dict
        The keyword parameters to pass to 
        diffusion_graph.knn_graph_to_diffution_matrix
    stochastic_NMF_constants: dict
        The keyword parameters to pass to NMF.stochastic_NMF_projection
    NMF_boosted_constants: dict
        The keyword parameters to pass to NMF.NMF_boosted (in addition to 
        stochastic_NMF_constants)        
    make_debug_display_vector: func
        A function that returns a function that takes (F,G,tital) and displays 
        it for debugging. 

    """
    
    
    if feature_space_constants is None:
        feature_space_constants = defualt_feature_space_constants
                                   
    if knn_graph_to_diffution_constants is None:
        knn_graph_to_diffution_constants = defualt_knn_graph_to_diffution_constants
                            
    if stochastic_NMF_constants is None:
        stochastic_NMF_constants = defualt_stochastic_NMF_constants 
        
    if NMF_boosted_constants is None:
        NMF_boosted_constants = default_NMF_boosted_constants
        
    aditinal_info = {}
    
    current_space = feature_space.create_feature_space(curent_scale=curent_scale,
                                                          **feature_space_constants)
    discriptor = current_space.create_image_discriptor(superpixels_at_scale);
    

    
    use_gpu = False #discriptor.shape[1] > 15
        
    
    # Do the diffution on the cpu or gpu
    if use_gpu:
        knn_graph =  diffusion_graph.build_knn_graph_gpu(discriptor,knn)
    if not use_gpu:
        knn_graph =  diffusion_graph.build_knn_graph_ann(discriptor,knn)


    diff_mat = diffusion_graph.knn_graph_to_diffution_matrix(knn_graph,
                                         **knn_graph_to_diffution_constants )        
    
    
    if steps is None:
        steps = diffusion_graph.estimate_steps(diff_mat)  
        
    if make_debug_display_vector:
        debug_display_vector=make_debug_display_vector(superpixels_at_scale)
    else:
        debug_display_vector=None
        
    if (clustering_algorithum == 'NMF' or 
                        clustering_algorithum == 'NMF-boosted'):
        if clustering_algorithum == 'NMF':
            F,G,error = NMF.stochastic_NMF_projection(diff_mat,steps=steps,
                                    debug_display_vector=debug_display_vector, 
                                    **stochastic_NMF_constants)        
        else:
            #http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
            kwargs = stochastic_NMF_constants.copy()
            kwargs.update(NMF_boosted_constants)
            F,G = NMF.NMF_boosted(diff_mat,NMF.stochastic_NMF_projection,
                                  steps=steps,
                                  debug_display_vector=debug_display_vector, 
                                  **kwargs)
                         
        n_labels = G.shape[0]
        label_map_mapping = np.argmax(G[:,:],axis = 0)
        
        aditinal_info['F'] = F
        aditinal_info['G'] = G
    else:
        best_F,_, _ = \
            NMF.estimate_starting_point(diff_mat,
                                        steps,
                                        r_guess_precondition_runs,
                                     debug_display_vector=debug_display_vector)

        n_labels = best_F.shape[1]
        
        if clustering_algorithum == 'kmeans':
            means = sklearn.cluster.KMeans(n_clusters=n_labels,
                                           max_iter=1000,n_init=100,n_jobs=3 )
            means.fit(discriptor)
            label_map_mapping = means.labels_ 

        elif clustering_algorithum == 'spectral':
            spectral = sklearn.cluster.SpectralClustering(n_clusters=n_labels,
                                            affinity='precomputed',n_init=100 )
            spectral.fit(diff_mat)
            label_map_mapping = spectral.labels_ 
        else:
            raise Exception("Unknown clustering algorithum %s"%clustering_algorithum)
    
    label_map = region_map.build_compound_region(superpixels_at_scale,label_map_mapping)
    return label_map, aditinal_info


def build_label_map_stack(hierarchical_superpixels,
                                     scale_list = None,
                                     build_label_map_prams={}):
                                         
    """
    Builds a stack of label maps from a set of hierarchical superpixels. This 
    method can automatically detect the scales of interest. 
    
    At is heart, this method will call build_label_map at a number of scales 
    and will return a map from scales to the different maps. 
    
    Parameters
    ----------
    hierarchical_superpixels: HierarchicalRegionMap
        The hierarchical superpixels to turn into labels. 
    scale_list: list of floats , optional
        The list of scales to use. By default, the list of scales will be 
        automatically calculated.
    build_label_map_prams: dict
        Parameters to pass to build_label_map.   
        
    Returns 
    -------
    map_dict: dict
        A mapping from scale to the label map at that scale. 
    aditinal_info_dict: dict
        A mapping from scale to any additional information returned by 
        build_label_map
    """
    atomic_map = hierarchical_superpixels.get_atomic_map()

    all_slic_scales =  hierarchical_superpixels.get_scales()  


    map_dict = {} 
    aditinal_info_dict = {}
    
    if (scale_list is None):
        #Auto scale selection
    
        current_index = 0
        while(current_index < len(all_slic_scales) ):   
            
            curent_scale = all_slic_scales[current_index]
            print(curent_scale)
            
            superpixels_at_scale =\
                hierarchical_superpixels.get_single_scale_map(curent_scale)
            label_map, aditinal_info = build_label_map(superpixels_at_scale,
                                                       curent_scale,
                                                       **build_label_map_prams)
            
            map_dict[curent_scale] = label_map
            aditinal_info_dict[curent_scale] = aditinal_info

            #Create a table stating which label each atomic superpixel belongs 
            atomic_label_table = np.empty(len(atomic_map),dtype=np.int32)
            for label_idx,label in enumerate(label_map):
                atomic_indexes = np.array(label_map.get_atomic_indexes(label) )
                atomic_label_table[atomic_indexes] = label_idx

            #Use a binery search to find the jump point
            min_scale_index = current_index+1
            max_scale_index = len(all_slic_scales) -1
            
            has_set = False
            while min_scale_index + 1 < max_scale_index:
                mid_scale_index = (max_scale_index+min_scale_index) /2
                scale_at_mid = all_slic_scales[mid_scale_index]
                
                scale_region_map =\
                    hierarchical_superpixels.get_single_scale_map(scale_at_mid)
                
                
                all_num = np.zeros(len(scale_region_map))
                
                for tid,super_pixel in enumerate(scale_region_map):
                    atomic_indexes = np.array(
                            scale_region_map.get_atomic_indexes(super_pixel))
                    all_num[tid] = len(np.unique(
                                        atomic_label_table[atomic_indexes]))
                    
                if np.min(all_num) >= 2 and not has_set:
                    max_scale_index = mid_scale_index
                else:
                    min_scale_index = mid_scale_index
                    
                
                if max_scale_index == min_scale_index + 1:
                    current_index = max_scale_index
                    has_set = True
                    
            if not(has_set):
                break
    else:
        for curent_scale in scale_list:
            superpixels_at_scale = \
                hierarchical_superpixels.get_single_scale_map(curent_scale)
            label_map,aditinal_info =\
                    build_label_map(superpixels_at_scale,
                                    curent_scale,**build_label_map_prams) 
            
            map_dict[curent_scale] = label_map
            aditinal_info_dict[curent_scale] = aditinal_info

    return map_dict, aditinal_info_dict