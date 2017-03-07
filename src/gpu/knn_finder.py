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

Created on Thu Jan 17 21:58:57 2013

@author: Yitzchak David Lockerman

"""
import numpy as np

import opencl_tools
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array;


import time


import bottleneck as bn
import utils


timing_constant = 200 
            

_OpenCL_code="""
        #ifdef WARPSPEED
            #define NOWARPBLOCK
        #else
            #define NOWARPBLOCK barrier(CLK_LOCAL_MEM_FENCE);
        #endif

        //Bases on NVIDA oclReduction's code and 
        // http://www.nvidia.com/content/cudazone/download/OpenCL/NVIDIA_OpenCL_ProgrammingOverview.pdf
        __kernel void calc_dists(__global const float *points,
                          __global const float *query,
                          int queryindex,
                          __global float *out_dists,
                          __global float *point_offset,
                          __global float *querry_offset,
                          __local float* sdata)
        {
            const uint global_index = get_global_id(0);
            const uint point_index = get_global_id(2);
            const uint local_offset = get_local_id(0)*get_local_size(1); 
            
            sdata += local_offset;
            queryindex += point_index;
            
            
            const uint dim = get_local_id(1);

            const float diff = dim < dims && global_index < rows ?
                                points[global_index*dims + dim ] - 
                                            query[queryindex*dims + dim] : 0;
                            
            sdata[dim] = diff*diff;

            barrier(CLK_LOCAL_MEM_FENCE);  
"""+opencl_tools.get_inkernal_reduction('sdata','blockSize','dim')+"""      
            
            //And return the sqrt
            if (dim == 0 && global_index < rows)
            {
                out_dists[global_index*get_global_size(2) + point_index] =
                        point_offset[global_index]
                                +sqrt(sdata[0])
                                +querry_offset[queryindex]; 
            }

        }

"""

class KnnFinder(object):
    
    def __init__(self, points,point_offset=None):
        self._ctx = opencl_tools.get_a_context()
        self._queue = cl.CommandQueue(self._ctx,properties=opencl_tools.profile_properties);
        self._gpu_points = cl_array.to_device(self._queue,points.astype(np.float32))
        
        if(point_offset!=None):
            self._point_offset \
                = cl_array.to_device(self._queue,point_offset.astype(np.float32))
        else:
            self._point_offset \
                = cl_array.zeros(self._queue,points.shape[0],np.float32)
                
    def __enter__ (self):
        return self;
    
    def __exit__ (self, type, value, tb):
        try:
            self._queue.finish()
            self._gpu_points.data.release();
            self._point_offset.data.release();
        except:
            pass;
        
        
    #@profile
    def get_knn(self,querys,k,query_offset=None):
        float32_size = int(np.float32(0).nbytes)
        


        #####Figure out sizes
        point_length=int(self._gpu_points.shape[0]);
        dims = int(self._gpu_points.shape[1])
        dims_po2 = 2**dims.bit_length()
        device = self._ctx.devices[0]
        
        querys_per_run = (timing_constant*3200)/point_length
                
        querys_per_run = min(querys_per_run,\
                    (2**(device.get_info(cl.device_info.ADDRESS_BITS))-1)\
                                                    /(2*point_length*dims_po2))                                            
        querys_per_run = min(querys_per_run,querys.shape[0])
        querys_per_run = min(querys_per_run,\
                            device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)\
                                /(self._gpu_points.shape[0]*float32_size*5))
                                
        querys_per_run = int(max(querys_per_run,1));
        
        ###Create buffers
        true_k = min(k,point_length)
        dists = np.empty((querys.shape[0],true_k),dtype=np.float32);
        indexes = np.empty((querys.shape[0],true_k),dtype=np.int32);
        
        dists_gpu = \
            cl_array.empty(self._queue,(self._gpu_points.shape[0],querys_per_run),np.float32);
        query_gpu = \
                cl_array.to_device(self._queue,querys.astype(np.float32)) 
            
        if(query_offset!=None):
            gpu_query_offset \
                = cl_array.to_device(self._queue,query_offset.astype(np.float32))
        else:
            gpu_query_offset \
                    = cl_array.zeros(self._queue,querys.shape[0],np.float32)        
        
        
        ###Create kernal
        preamble="""
        #define blockSize %d
        #define dims %d
        #define rows %d
        
        """ % (dims_po2,dims,self._gpu_points.shape[0])

        preamble = opencl_tools.build_preamble_for_context(self._ctx,preamble)
        prg = cl.Program(self._ctx,preamble+_OpenCL_code).build();
        
        distkernal = prg.calc_dists;
        distkernal.set_scalar_arg_dtypes\
                        ([None,None,np.int32,None,None,None,None])  

        ##calulate run size
        max_run_items=distkernal.get_work_group_info\
                            (cl.kernel_work_group_info.WORK_GROUP_SIZE,device)                             
        
        work_item_sizes = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES);
        
        
        point_per_run=min([work_item_sizes[0]/4,\
                            (max_run_items)/(8*dims_po2),\
                            point_length])       
                            
        if(point_per_run < 1):
            point_per_run = 1              
        local_shape =  (point_per_run ,dims_po2,1)
        
        global_shape = (point_length,dims_po2,querys_per_run)
        overshoot = global_shape[0] % local_shape[0]
        if overshoot > 0:
            global_shape = (global_shape[0]+ (local_shape[0]-overshoot)\
                                                            ,global_shape[1]\
                                                            ,global_shape[2])
                                                            
        #selector = gpu_quick_selection.GPUQuickSelection\
        #                                    (ctx,self._queue,dists_gpu.shape)      

        dist_calc = None;
        def make_calc(stop_point,ii) : 
                def calc():
                    if(k<dist_local.shape[0]):
                        index_local  = utils.argpartsort(dist_local,k,axis=0)[:k,:]
                        
                        for r in xrange(stop_point-ii):
                            dists_cpu_buffer_local = dist_local[index_local[:,r],r];
                            indexes_cpu_buffer_local = index_local[:,r]
                         
                
                            index_local2  = np.argsort(dists_cpu_buffer_local,axis=0)
                        
                            dists[ii+r,:] = dists_cpu_buffer_local[index_local2]
                            indexes[ii+r,:] = indexes_cpu_buffer_local[index_local2]
                            #print ii+r,indexes_cpu_buffer_local[index_local2[0]]
                    else:
                        for r in xrange(stop_point-ii):
                            dists_cpu_buffer_local = dist_local[:,r];

                            index_local2  = np.argsort(dists_cpu_buffer_local,axis=0)
                        
                            dists[ii+r,:] = dists_cpu_buffer_local[index_local2]
                            indexes[ii+r,:] = index_local2                        
                return calc
                                              
        for ii in xrange(0,querys.shape[0],querys_per_run):
            
            #For the last step, we need to make shure not to go over
            if (querys.shape[0]-ii) < querys_per_run:
                global_shape = (global_shape[0],
                                    global_shape[1],
                                     (querys.shape[0]-ii))
            
            distkernal(self._queue,global_shape,local_shape,\
               self._gpu_points.data,\
               query_gpu.data,\
               ii,\
               dists_gpu.data,\
               self._point_offset.data,\
               gpu_query_offset.data,\
               cl.LocalMemory(float32_size*local_shape[0]*local_shape[1])).wait()
            
            stop_point = min(ii+querys_per_run,querys.shape[0])
            
            if(dist_calc != None):
                dist_calc();

            
#            newdists, newindexs = \
#                    selector.gpu_quick_selection(self._queue,dists_gpu,k,stop_point-ii)
#                    
#            dists[ii:stop_point,:] = newdists.T
#            indexes[ii:stop_point,:] = newindexs.T
            

            dist_local = dists_gpu.get(self._queue);
            dist_calc = make_calc(stop_point,ii) 

        if(dist_calc != None):
            dist_calc();
            
        dists_gpu.data.release();
        query_gpu.data.release();
        gpu_query_offset.data.release();
        
        return indexes,dists


if __name__ == '__main__': 
    
    import scikits.ann as ann
    #np.set_printoptions(threshold = np.nan)
    k=200;
    points = np.random.rand(500000, 64).astype(np.float32);
    queries = np.random.rand(500, 64).astype(np.float32);
    
    t0 =time.time()    
    gpu_knn = KnnFinder(points)
    our_indexies,our_distances = gpu_knn.get_knn(queries,k)
    
    t1 =time.time()   
    print t1-t0
    
    truth_tree = ann.kdtree(points);
    truth_points, truth_distances= truth_tree.knn(queries,k);
    truth_distances=np.sqrt(truth_distances)
    
    t2 =time.time()   
    
    print  t2-t1
    #print truth_points
    
    error = np.abs(our_distances- truth_distances)/truth_distances;

    index_error = our_indexies-truth_points
    print "----"
    print np.nansum(error)/error.size
    print np.sum(np.abs(index_error)>0)/index_error.size
    