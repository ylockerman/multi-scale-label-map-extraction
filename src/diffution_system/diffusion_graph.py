# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:20:32 2013

@author: Yitzchak David Lockerman
"""

#import knn_finder

import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg


import scipy.optimize as spopt

    
def build_knn_graph_gpu(features,k):
    """
        Create a knn graph using the gpu
    """
    print "Creating a %d-nn graph on the gpu" % (k);
    
    import gpu.knn_finder
    with   gpu.knn_finder.KnnFinder(features) as ktree:
        nearist,dist = ktree.get_knn(features,k)

    return nearist,dist*dist
    
def build_knn_graph_ann(features,k,eps=.001):
    """
        Create a knn graph using the ann library 
    """
    if k > features.shape[0]:
        k = features.shape[0]
        
    print "Creating a %d-nn graph on the cpu with an error of 1+%f" % (k,eps);
    
    import scikits.ann as ann

    knn_finder = ann.kdtree(features);
    nearist, dist_sqr= knn_finder.knn(features,k,eps);
    
    return nearist,dist_sqr


def knn_graph_to_weight_matrix(graph,smothing_factor='smothing_factor',
                                       norm_type = 'global',normalizer_in=None):
    
    nearist,dist_sqr = graph
    n_points = dist_sqr.shape[0]
    k = dist_sqr.shape[1]
    
    #We need to construct a sparce matrix quicly, to do that we prealocate 
    #an ij matrix and a data matrix, fill them, then convert to csr
    data = np.zeros( (2*k*n_points) )
    ij = np.zeros((2,2*k*n_points),np.int)
    insertion_pouint = 0;
    
    non_equal_indexes = np.nonzero(np.all(dist_sqr>0,axis=0))[0]
    
    if len(non_equal_indexes) > 0:
        first_good_index = non_equal_indexes[0]
    else:
        first_good_index = k
        
    local_offset = 6
    
    if norm_type == 'local' and first_good_index + local_offset >= k:
        #norm_type = 'local'
        local_offset = k - first_good_index - 1
        print "Not enugh index for local norm, falling back to offset %d" % local_offset;
        
    if norm_type == 'global':
        normalizer = np.zeros((n_points))
        good_vals = np.isfinite(dist_sqr[:,1])
        normalizer[:] = np.sqrt( np.mean(dist_sqr[good_vals,1]) );
        
    elif norm_type == 'local':
        normalizer = np.sqrt(dist_sqr[:,first_good_index+local_offset])
        normalizer[np.isinf(normalizer)] = 0
        
    elif norm_type == 'input':
        if normalizer_in is None:
            raise ValueError('No normalizer inputed')
 
        if isinstance(normalizer_in, np.ndarray):
            if normalizer_in.shape != (n_points,):
                raise ValueError('Normalizer has the wrong shape')   
            normalizer = normalizer_in
        else:
            normalizer = np.empty((n_points))           
            normalizer[:] = normalizer_in
            
    elif norm_type == 'none':
        normalizer = np.ones((n_points))
        
    else:
        raise ValueError('Unknown norm_type :' + norm_type)
    #import matplotlib.pyplot as plt
    #plt.errorbar(np.arange(dist_sqr.shape[1]),
    #                 np.mean(dist_sqr,axis=0),yerr=np.std(dist_sqr,axis=0) );
    #plt.show()
        
    print "smothing_factor",smothing_factor
    if smothing_factor == "intrinsic dimensionality":
        
        
            
        #obj_function(r) := exp(-f/(exp(r)));
        #diff(obj_function(r),r,1) = f*%e^(-f*%e^(-r)-r)
        #diff(obj_function(r),r,2) = f*(f*%e^(-r)-1)*%e^(-f*%e^(-r)-r)
        def weight(r):
            weight_val = 0
            dweight_val = 0
            #ddweight_val = 0
            exp_nr = np.exp(-r)
            
            for pid in xrange(n_points):
                good_vals = np.nonzero(nearist[pid,:]>=0)
                f = dist_sqr[pid,:]/(normalizer[pid]*normalizer[nearist[pid,:]] )
                weight_val += np.sum(np.exp(-f[good_vals]*exp_nr))
                dweight_val += np.sum(f*np.exp(-f*exp_nr - r ))
                #ddweight_val += f*(f*exp_nr-1)*np.exp(-f*exp_nr-r)

            return np.log(weight_val),dweight_val/weight_val

        def neq_dweight(r):
            _,dweight = weight(r)
            return -dweight

        max_r  = spopt.brute(neq_dweight,(slice(-10, 10, 0.25),),
                                                         finish=spopt.fmin)
        #max_r = max_r_opt_res.x
        
        smothing_factor = np.exp(-max_r)
        d = -2*neq_dweight(max_r)
    else:
        d = 1
        
        
        print "New epsilon factor val", smothing_factor ,"d", d, "steps (Jianye)", d*np.power(n_points,1.0/d)
#        
#            
#        r_list = np.linspace(-6*np.log(10),6*np.log(10),50)
#        w_val,dirv_val = zip(*[ weight(r) for r in r_list ])
#        
#        plt.figure(2)
#        #delta = (r_list/np.log(10))[1] - (r_list/np.log(10))[0]
#        plt.plot( (r_list/np.log(10)) - np.log(2)/np.log(10),w_val)        
#        plt.plot( (r_list/np.log(10)) - np.log(2)/np.log(10),dirv_val)
#        plt.plot( (max_r/np.log(10)) - np.log(2)/np.log(10),-neq_dweight(max_r),'*')        
    
            
    for pid in xrange(n_points):
        good_vals = np.nonzero(nearist[pid,:]>=0)
        num_good_vals = good_vals[0].shape[0]
        
        f = (dist_sqr[pid,:])/(normalizer[pid]*normalizer[nearist[pid,:]] )
        weight_val = np.exp(- smothing_factor*f )

        #dist_mat[r,nearist[r,:]]= weight_val;
        insertion_pouint_end =insertion_pouint + num_good_vals
        ij[0,insertion_pouint:insertion_pouint_end] = pid
        ij[1,insertion_pouint:insertion_pouint_end] = nearist[pid,good_vals]
        data[insertion_pouint:insertion_pouint_end] = weight_val[good_vals]
        insertion_pouint = insertion_pouint_end
        
        #dist_mat[nearist[r,:],r]= weight_val;
        insertion_pouint_end =insertion_pouint + num_good_vals
        ij[0,insertion_pouint:insertion_pouint_end] = nearist[pid,good_vals]
        ij[1,insertion_pouint:insertion_pouint_end] = pid
        data[insertion_pouint:insertion_pouint_end] = weight_val[good_vals]
        insertion_pouint = insertion_pouint_end    
        
        #Normalize the matrix so that energy is conserved
    print data.shape, ij.shape
    dist_mat = sparse.csc_matrix(
                    (data[:insertion_pouint],ij[:,:insertion_pouint]),
                                                    shape=(n_points,n_points));
        
    #if np.isnan(d):
    #    print d  
                                                 
    return dist_mat, int(d*np.power(n_points,1.0/d))
    
def weight_matrix_to_diffution_matrix(dist_mat):
    n_points = dist_mat.shape[0]
    
    #dist_mat.setdiag(np.ones(n_points,np.float32))
    nrom_cnst = dist_mat.transpose().dot(np.ones(dist_mat.shape[0]))  
    full_matrix = dist_mat*sparse.diags(np.nan_to_num(1/nrom_cnst),0);



    return full_matrix.astype(np.float64);    

def knn_graph_to_diffution_matrix(graph,smothing_factor, norm_type = 'global',normalizer_in=None):
    """
        Given a knn graph, create the diffution matrix
    """
    print ("Building a fiddution matrix with a smothing factor of %s." % 
                                                                (smothing_factor));

    dist_mat, _ = knn_graph_to_weight_matrix(graph,smothing_factor,norm_type,normalizer_in)
    
    
    return weight_matrix_to_diffution_matrix(dist_mat)
         
    
    
def do_diffusion(diff_mat,heat_map,steps=1):
    """
    Preform diffusion on a heat map
    """
    heat_map_new = heat_map.reshape((-1,)).astype(np.float64)

    while(steps >= 1):
        heat_map_new = diff_mat.dot(heat_map_new)
        steps -= 1
        
    return heat_map_new.reshape(heat_map.shape)
    
def estimate_steps(diff_mat):
    """
    Estimate a good number of steps to use for diffusion. The estimate is the 
    number of steps needed for the first decaying eigenvector to fall to 95% of
    its original energy, rounded down. 
    """
    w,v = splinalg.eigs(diff_mat,min(500,diff_mat.shape[0]-2) )
    w_sorted = np.real(np.sort(w)[::-1])
    
    first_good_index = np.nonzero(w_sorted < .9999)[0][0]
    steps = max(abs(int( np.log(.95*w_sorted[first_good_index])/np.log(w_sorted[first_good_index])  )) ,1)
  
    return steps 

    
