# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:20:32 2013

@author: Yitzchak David Lockerman
"""

if __name__ == '__main__':
    import sys
    sys.path.append('.')
    

import numpy as np
import time
import math

import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.optimize

import scipy.optimize

from gpu import opencl_tools
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
elementwise = opencl_tools.cl_elementwise
reduction = opencl_tools.cl_reduction
from gpu import gpu_sparse_matrix
from gpu import gpu_matrix_dot
from gpu import gpu_algorithms


import matplotlib.pyplot as plt

import numexpr

import diffusion_graph

def sparce_matrix_power(D,v,s):
    result = v.copy();

    
    for ii in xrange(s):
        result = D*result;
    
    return result;

def random_stocastic(n,m):
    d = 1
    
    mat = np.random.rand(n,m)
    mat /= np.sum(mat,axis=0)
    
    return mat


def full_random_samples(matrix,a):
    full_sam

#code for first kernal: Get F,G from gamma, delta
_safe_normalization_kernal = """
    //if the matrix is in row major order, must define ROW_MAJOR_ORDER
    
    #ifdef ROW_MAJOR_ORDER
        #define from_matrix(mat,r,c) mat[r*cols+c]
    #else
        #define from_matrix(mat,r,c) mat[r+rows*c]
    #endif
                
    __kernel void safe_normilization_kernal(__global const float* matrix,
                               __global const float* norm,
                                         __global float * out)
    {

        const int rows = get_global_size(0); 
        const int cols = get_global_size(1); 
        const int r = get_global_id(0); 
        const int c = get_global_id(1); 
        
        const float norm_const = norm[c];      
        
        if(norm_const < 1e-6 && norm_const > -1e-6)
            from_matrix(out,r,c) = 0;
        else
            from_matrix(out,r,c) = from_matrix(matrix,r,c)/norm_const;
    }
        
"""

class SafeNormalizationKernal:
    
    def __init__(self,matrix,queue):
        self._queue =  queue
            
        self._block_size = 16
        self._matrix = matrix
        
        self._matrix_order = gpu_algorithms._get_matrix_order(matrix)
        
        preamble = ""
        if(self._matrix_order == 'C' ):
            preamble += "#define ROW_MAJOR_ORDER\n"
            
        prg = cl.Program(self._queue.context,preamble+_safe_normalization_kernal).build();
                         
        self.kernel = prg.safe_normilization_kernal
        
    def __call__(self,out,norm_vector,matrix=None):
        if matrix is None:
            matrix = self._matrix
        else:
            assert self._matrix_order == gpu_algorithms._get_matrix_order(matrix)
            
        global_size = matrix.shape
        self.kernel(self._queue,global_size,None,
                            matrix.data,norm_vector.data,out.data)

           
        cl.enqueue_barrier(self._queue)
                
        return out
        
            

#code for second kernal: get and use gradent from F,G and the calulated values
_gradent_fixer = """
    //if the matrix is in row major order, must define ROW_MAJOR_ORDER
    
    #ifdef ROW_MAJOR_ORDER
        #define from_matrix(mat,r,c) mat[r*cols+c]
    #else
        #define from_matrix(mat,r,c) mat[r+rows*c]
    #endif
                
    __kernel void gradent_fixer(__global const float* grade_in,
                                __global const float* norm,
                                __global const float* grade_fix,
                                __global float * grade_out,
                                __global float *value_in,
                                __global float *value_out,
                                float di)
    {

        const int rows = get_global_size(0); 
        const int cols = get_global_size(1); 
        const int r = get_global_id(0); 
        const int c = get_global_id(1); 
        
        float invers_norm = 0.0;
        if(norm[c]> 1e-6 || norm[c]< -1e-6 )
            invers_norm = 1/norm[c];
            
        //F_grade*inv_norm_gamma - gamm_grade_fix*np.square(inv_norm_gamma)
         float gradeent = 
            from_matrix(grade_in,r,c)*invers_norm - 
                                        grade_fix[c]*invers_norm*invers_norm;
        
        from_matrix(grade_out,r,c) = gradeent;
        
        float new_val =from_matrix(value_in,r,c)  + di*gradeent;
        if(new_val < 0) new_val =0;
        
        from_matrix(value_out,r,c) = new_val;
    }
        
"""
class GradentFixer:
    
    def __init__(self,grade_in,norm,grade_fix,queue):
        self._queue =  queue
            
        self._block_size = 16
        self._grade_in = grade_in
        self._norm = norm
        self._grade_fix = grade_fix
        
        self._order = gpu_algorithms._get_matrix_order(grade_in)

        
        preamble = ""
        if(self._order == 'C' ):
            preamble += "#define ROW_MAJOR_ORDER\n"
            
        prg = cl.Program(self._queue.context,preamble+_gradent_fixer).build();
                         
        self.kernel = prg.gradent_fixer
        
        self.kernel.set_scalar_arg_dtypes([ 
                                        None,#__global const float* grade_in,
                                        None,#__global const float* norm,
                                        None,#__global const float* grade_fix,
                                        None,#__global float * grade_out,
                                        None,#__global float *value_in,
                                        None,#__global float *value_out,
                                        np.float32#__global float di)
                                  ])
        
    def __call__(self,value_out,gradent_out,value_in,di):
        assert gpu_algorithms._get_matrix_order(gradent_out) == self._order
        assert gpu_algorithms._get_matrix_order(value_in) == self._order
        assert gpu_algorithms._get_matrix_order(value_out) == self._order
        
        assert self._grade_in.shape == gradent_out.shape;
        assert self._grade_in.shape == value_in.shape;
        assert self._grade_in.shape == value_out.shape;
        
        global_size = self._grade_in.shape

        self.kernel(self._queue,global_size,None,
                            self._grade_in.data,
                            self._norm.data,
                            self._grade_fix.data,
                            gradent_out.data,
                            value_in.data,
                            value_out.data,
                            np.float32(di))

        cl.enqueue_barrier(self._queue)
        
#This kernal caluates the error, ignoring the constent term with only D
_error_kernal_code = """
    //Must define: ROWS,COLS
    //ALSO, must define an access for each matrix
    
    __kernel void error_kernal(__global float *FT,
                               __global float *G,
                               __global float *left_power_T,
                               __global float *right_power,
                               __global float *FTF,
                               __global float *GGT,
                               __global float *out,
                               int r_value)
    {
        const int r = get_global_id(0); 
        const int c = get_global_id(1); 
        
        float first_val = 0;
        if(r < r_value && c < r_value)
            first_val = FTF_at(r,c,r_value,r_value)*GGT_at(r,c,r_value,r_value);
            
        out_at(r,c,ROWS,COLS) = 
            first_val
                -FT_at(r,c,ROWS,COLS)*left_power_T_at(r,c,ROWS,COLS)
                -right_power_at(r,c,ROWS,COLS)*G_at(r,c,ROWS,COLS);
        
    }
"""

class ErrorCalulator:
    
    def __init__(self,FT,G,left_power_T,right_power,FTF,GGT,out,queue):
        self._queue =  queue
        
        all_names = ['FT','G','left_power_T','right_power','FTF','GGT','out']
        all_matrixes = [FT,G,left_power_T,right_power,FTF,GGT,out]
        normal_shaped_maps = [FT,G,left_power_T,right_power,out]
        self._orders = map( gpu_algorithms._get_matrix_order, all_matrixes)
        self._shapes = [x.shape for x in all_matrixes]
        self._mat_shape = FT.shape
        
        
        for mat in normal_shaped_maps:
            assert self._mat_shape == mat.shape
        
        assert FTF.shape[0] == FTF.shape[1]
        assert FTF.shape == GGT.shape
        
        preamble = "#define ROWS %d\n#define COLS %d" % self._mat_shape
        for name,mat in zip(all_names,all_matrixes):
            preamble += gpu_algorithms.add_matrix_axses_for(name,mat)
            
        prg = cl.Program(self._queue.context,preamble+_error_kernal_code).build();
                         
        self.kernel = prg.error_kernal
        
        self.kernel.set_scalar_arg_dtypes([ 
                                           None,# __global float *FT,
                                           None,#__global float *G,
                                           None,#__global float *left_power_T,
                                           None,#__global float *right_power,
                                           None,#__global float *FTF,
                                           None,#__global float *GGT,
                                           None,#__global flaot *out,
                                           np.int32,#float r_value
                                  ])
        
    def __call__(self,FT,G,left_power_T,right_power,FTF,GGT,out):
        all_matrixes = [FT,G,left_power_T,right_power,FTF,GGT,out]
        for order,shape,mat in zip(self._orders,self._shapes,all_matrixes):
            assert shape == mat.shape 
            assert order == gpu_algorithms._get_matrix_order(mat)

        global_size = self._mat_shape

        self.kernel(self._queue,global_size,None,
                            FT.data,
                            G.data,
                            left_power_T.data,
                            right_power.data,
                            FTF.data,
                            GGT.data,
                            out.data,
                            FTF.shape[0])

        cl.enqueue_barrier(self._queue)    
        
#dose a transpose on a matrix by chagning the metadata
def _gpu_transpose(gpu_mat):
    other_order = gpu_algorithms._get_matrix_order(gpu_mat)
    if(other_order == "C"):
        our_order = "F"
    else:
        our_order = "C"

    return cl_array.Array(gpu_mat.queue,
                          gpu_mat.shape[::-1],
                          gpu_mat.dtype,
                          our_order,
                          data = gpu_mat.data)
__T = _gpu_transpose

#@profile
def optimisation_clustering_gradent_decent_probablity_sparce(ctx,diff_mat, r,steps=1, max_itter = 50000,gamma=None,delta=None,cutt_off = 1e-4,display_size=None,**kwargs):
    diff_mat=diff_mat.astype(np.float32)
    
    if ctx is None:
        ctx = opencl_tools.get_a_context()
        
    n = diff_mat.shape[0]
    #a list of bufferse we need to free at the end
    to_release = []
    
    #Since most operations requre only one of F and G we can have two seperate queues
    queue_gamma_F = cl.CommandQueue(ctx,properties=opencl_tools.profile_properties);
    queue_delta_G = cl.CommandQueue(ctx,properties=opencl_tools.profile_properties);
    
    if(sparse.issparse(diff_mat)):
        diff_mat_gpu = gpu_sparse_matrix.OpenclCsrMatrix(diff_mat,queue=queue_delta_G,dtype=np.float32); #For Dv 
        diff_mat_T_gpu = gpu_sparse_matrix.OpenclCsrMatrix(diff_mat.T,queue=queue_gamma_F,dtype=np.float32);#For vD
        multiple_prod = True
    else:
        mat_power = np.linalg.matrix_power(diff_mat,steps)
        diff_mat_gpu = gpu_matrix_dot.DotKernal(mat_power,(n,r),b_order="F",queue=queue_delta_G)
        diff_mat_T_gpu = gpu_matrix_dot.DotKernal(mat_power.T,(n,r),b_order="C",c_order="F",queue=queue_gamma_F)
        
        multiple_prod = False
        
    
    
    to_release += [diff_mat_gpu,diff_mat_T_gpu]
    #Begin Temp line
#    diff_mat = np.asarray(diff_mat.todense())
#    #our_error = []
#    #gt_error = []
    #end Temp
    
    
     
    if gamma is None:
        gamma = np.random.uniform(size=(n,r)).astype(np.float32)        
    else:
        assert gamma.shape == (n,r)
        assert np.all(gamma >= 0)
        gamma = gamma.astype(np.float32,order="C")
        
        
    if delta is None:
        delta = np.random.uniform(size=(r,n)).astype(np.float32)
    else:
        assert delta.shape == (r,n)
        assert np.all(delta >= 0)
        delta = delta.astype(np.float32,order="C")        
        

    gamma = cl_array.to_device(queue_gamma_F,gamma)
    delta = cl_array.to_device(queue_delta_G,delta)
    to_release += [gamma.data,delta.data]
    
    
    #other varubles, preallocated for speed
    F = cl_array.empty_like(gamma);
    G = cl_array.empty_like(delta) 
    F_grade= cl_array.empty_like(gamma) 
    G_grade= cl_array.empty_like(delta) 
    to_release += [F.data,G.data,F_grade.data,G_grade.data]
    
    #summation tools, to calulatet he sum of gamma and delta
    gamma_sum = gpu_algorithms.SumKernal(gamma,axis=0,queue=queue_gamma_F)
    norm_gamma = cl_array.empty(queue_gamma_F,(r),dtype=np.float32)
    to_release += [norm_gamma.data]
    
    delta_sum = gpu_algorithms.SumKernal(delta,axis=0,queue=queue_delta_G)
    norm_delta = cl_array.empty(queue_delta_G,(n),dtype=np.float32)
    to_release += [norm_delta.data]
    
    #normilisation tools, used to normalize the gamma and delta    
    gamma_to_F = SafeNormalizationKernal(gamma,queue=queue_gamma_F)  
    delta_to_G = SafeNormalizationKernal(delta,queue=queue_delta_G)   

    
    #multiplcation tools and matrixes
    GGT = cl_array.empty(ctx,(r,r),dtype=np.float32)
    G_mult_GT = gpu_matrix_dot.DotKernal(G,__T(G).shape,__T(G),GGT,queue=queue_delta_G)
    
    FGGT = cl_array.empty_like(gamma)
    #we use the F queue here because that is where it will be used,
    #This is a point we must use events to sync
    F_mult_GGT = gpu_matrix_dot.DotKernal(F,GGT.shape,GGT,FGGT,queue=queue_gamma_F)
    
    FTF = cl_array.empty(ctx,(r,r),dtype=np.float32)
    FT_mult_F = gpu_matrix_dot.DotKernal(__T(F),F.shape,F,FTF,queue=queue_gamma_F)
    
    FTFG = cl_array.empty_like(delta)
    #This is used for G so we need to put it in G queue, see above
    FTF_mult_G = gpu_matrix_dot.DotKernal(FTF,G.shape,FTF,FTFG,queue=queue_delta_G)
    
    to_release += [GGT.data, FGGT.data, FTF.data, FTFG.data]
    #to_release += [G_mult_GT,F_mult_GGT,FT_mult_F,FTF_mult_G ]
    
    #matrix vector power of sparce matrixes
    left_power = cl_array.empty_like(gamma)
    left_power_temp = cl_array.empty_like(gamma)
    right_power = cl_array.empty_like(delta)
    right_power_temp = cl_array.empty_like(delta)
    
    to_release += [left_power.data,right_power.data,
                       left_power_temp.data,right_power_temp.data]
    
    #gradent matrixes
    F_grade = cl_array.empty_like(gamma)
    G_grade = cl_array.empty_like(delta)
    
    to_release += [F_grade.data,G_grade.data]
    
    #The gradernt factor
    gamm_grade_fix = cl_array.empty(queue_gamma_F,(r),dtype=np.float32)
    gamm_grade_fix_calulate = gpu_algorithms.SumProductKernal(F_grade,axis=0,queue=queue_gamma_F)
    
    delta_grade_fix = cl_array.empty(queue_delta_G,(n),dtype=np.float32)
    delta_grade_fix_calulate = gpu_algorithms.SumProductKernal(G_grade,axis=0,queue=queue_delta_G)
    
    to_release += [gamm_grade_fix.data,delta_grade_fix.data]
    
    #Kernal to fix the gradent
    gamma_gradent_calulater = GradentFixer(F_grade,norm_gamma,gamm_grade_fix,queue=queue_gamma_F)
    gamma_grade = cl_array.empty_like(gamma)
    
    delta_gradent_calulater = GradentFixer(G_grade,norm_delta,delta_grade_fix,queue=queue_delta_G)
    delta_grade = cl_array.empty_like(delta)

    to_release += [gamma_grade.data,delta_grade.data]
    
    #The output values for each itteration
    gamma_new = cl_array.empty_like(gamma)
    delta_new = cl_array.empty_like(delta)
    
    to_release += [gamma_new.data,delta_new.data]

    #The reduction to calulate the gradent
    change_calulate = reduction.ReductionKernel(ctx,np.float32,neutral="0",
                                reduce_expr="a+b",
                                map_expr="(new_val[i]-old_val[i])*gradient[i]",
                                arguments="__global float *old_val,"
                                          "__global float *new_val,"
                                          "__global float *gradient")
                       

    #kernal and storage to calulate the gradent
    error_temp_stor = cl_array.empty(queue_gamma_F,(r,n),dtype=np.float32)
    error_calulator = ErrorCalulator(__T(F),G,__T(left_power),right_power,FTF,GGT,error_temp_stor,queue_gamma_F)
                  
    to_release += [error_temp_stor.data]
    
    gamma_change = np.ones((1),np.float32)
    delta_change = np.ones((1),np.float32)
    itter = 0;
    di = 2000;
    change_target = 5;
    di_decay = .5
    
#    #bgin temp code
#    diff_mat = np.asarray(diff_mat.todense())
#    last_change_gamma = None
#    last_change_delta = None
#    real_change = []
#    total_change = []
#    D2R = np.linalg.matrix_power(diff_mat,2*steps)
#    trDTD = np.linalg.norm(D2R)**2
#    plt.figure(2)
#    plt.clf()
#    
#    def nn_fix(val):
#        val=val.copy()
#        val[val<0] = 0
#        
#        return val
#    def error_value(gamma_cpu,delta_cpu):
#        gamma_cpu = nn_fix(gamma_cpu)
#        delta_cpu = nn_fix(delta_cpu)
#        
#        norm_gamma_cpu = np.sum(gamma_cpu,axis=0)
#        norm_delta_cpu = np.sum(delta_cpu,axis=0)
#        
#        F_cpu = np.zeros_like(gamma_cpu)
#        G_cpu = np.zeros_like(delta_cpu)
#        
#        good_norm_gamma_cpu = norm_gamma_cpu >0
#        good_norm_delta_cpu = norm_delta_cpu >0
#        
#        F_cpu[:,good_norm_gamma_cpu] = gamma_cpu[:,good_norm_gamma_cpu]/norm_gamma_cpu[good_norm_gamma_cpu]
#        G_cpu[:,good_norm_delta_cpu] = delta_cpu[:,good_norm_delta_cpu]/norm_delta_cpu[good_norm_delta_cpu]
#    
#        DmFG = D2R-np.dot(F_cpu,G_cpu);   
#        return np.linalg.norm(DmFG)**2
#    #end temp code
    
    #code to calulate F and G from gamma and delta, note that since
    #all calulations is done in place, the infomation travels to outsidet 
    #the function
    
    #We also calulate FTF and GGT
    def calulate_F_G(gamma_to_use,delta_to_use):
        gamma_sum(norm_gamma,gamma_to_use)
        delta_sum(norm_delta,delta_to_use)

        
        gamma_to_F(F,norm_gamma,gamma_to_use)
        delta_to_G(G,norm_delta,delta_to_use)       
        
        #F_grade = -2*np.dot(DmFG,G.T)
        #G_grade = -2*np.dot(F.T,DmFG)
        #The -2 constent dose not matter, so we remove it for performance,
        #Note the pluss below
        
        #Calulate the matrix multiplcations

        G_mult_GT.dot(__T(G),GGT)   
        FT_mult_F.dot(F,FTF)          


    #this function senqrenises all the the queues
    #it must be called before varubles belonging to one queue are 
    #used by another
    def sync_queues():
        gamma_F_event = cl.enqueue_barrier(queue_gamma_F)
        delta_G_event = cl.enqueue_barrier(queue_delta_G)
        cl.enqueue_barrier(queue_gamma_F,[delta_G_event])
        cl.enqueue_barrier(queue_delta_G,[gamma_F_event])      

    #These functions calulate the actual diffution
    if multiple_prod:
        def calulate_left_power():
            if (steps == 1):
                diff_mat_gpu.dot(__T(G),left_power)
            else:
                diff_mat_gpu.dot(__T(G),left_power_temp)
                diff_mat_gpu.dot(left_power_temp,left_power)
                for i in xrange(steps/2-1): 
                    diff_mat_gpu.dot(left_power,left_power_temp)
                    diff_mat_gpu.dot(left_power_temp,left_power)   
                
        def calulate_right_power():
            if(steps == 1):
                 right_power_T = diff_mat_T_gpu.dot(F,__T(right_power))
            else:
                right_power_temp_T = diff_mat_T_gpu.dot(F,__T(right_power_temp))
                right_power_T = diff_mat_T_gpu.dot(right_power_temp_T,__T(right_power))
                for i in xrange(steps/2-1):  
                    diff_mat_T_gpu.dot(F,right_power_temp_T)
                    diff_mat_T_gpu.dot(right_power_temp_T,right_power_T) 
    else:
        # We compute the matrix power above, no need to do it here
        def calulate_left_power():
                old_G = G.get(queue_delta_G)
                diff_mat_gpu.dot(__T(G),left_power)
    
        def calulate_right_power():
                 right_power_T = diff_mat_T_gpu.dot(F,__T(right_power))     
                 

        
    def calulate_error_value():
        sync_queues()
      
        error_calulator(__T(F),G,__T(left_power),right_power,FTF,GGT,error_temp_stor)
        return float(cl_array.sum(error_temp_stor,np.float32,queue_gamma_F).get(queue_gamma_F))

        

    t0=time.time()
    #precalulate F and G, and the eror
    calulate_F_G(gamma,delta) #precalulate F,G to use for the erorr
    calulate_left_power() #precalulate the left power to use in the error
    calulate_right_power();
    current_error = calulate_error_value()
    while itter < max_itter:
        #di *= di_decay
        #calulate_F_G(gamma,delta) #should be already done
    
        #Since the operatons below need both F and G, we need to syncrinise them
        sync_queues()

        
        #FGGT = np.dot(F,np.dot(G,G.T))        
        FTFG =FTF_mult_G.dot(G,FTFG)
        #FTFG = np.dot(np.dot(F.T,F),G)
        FGGT =F_mult_GGT.dot(GGT,FGGT)

        #left_power = diff_mat_gpu*G.T #do this r times
        #calulate_left_power(); #calulated elseware

        #right_power = (diff_mat_T_gpu*F).T #Do this r times before the T
        #calulate_right_power(); #calulated elseware
        
        #Since the operatons below need both F and G, we need to syncrinise them again
        sync_queues()

        #F_grade = (left_power - FGGT)
        cl_array.Array._axpbyz(F_grade, 1, left_power, -1, FGGT, queue=queue_gamma_F)
        
        #G_grade = (right_power -  FTFG)
        cl_array.Array._axpbyz(G_grade, 1, right_power, -1, FTFG, queue=queue_delta_G)
        
        

        

        #gamm_grade_fix = np.tile(np.sum(F_grade*gamma,axis=0),(gamma.shape[0],1))
        gamm_grade_fix_calulate(gamm_grade_fix,gamma,F_grade)
        #delta_grade_fix = np.tile(np.sum(G_grade*delta,axis=0),(delta.shape[0],1))
        delta_grade_fix_calulate(delta_grade_fix,delta,G_grade)


        #gamma_grade = F_grade*inv_norm_gamma - gamm_grade_fix*np.square(inv_norm_gamma)
        #gamma_new = gamma + di*gamma_grade
        #bad_gamma = gamma_new<0;
        #bad_delta = delta_new<0;
        gamma_gradent_calulater(gamma_new,gamma_grade,gamma,di)
        
        #delta_grade = G_grade*inv_norm_delta - delta_grade_fix*np.square(inv_norm_delta)
        #delta_new = delta + di*delta_grade   
        #gamma_new[bad_gamma] = 0;
        #delta_new[bad_delta] = 0;
        delta_gradent_calulater(delta_new,delta_grade,delta,di)
        

#        #Calulate the error polynomal
#        trFGGtdFt = cl_array.vdot(FGGT,gamma_grade)        
#        trFTFGdGT = cl_array.vdot(FTFG,delta_grade)
#        trFTDdGT  = cl_array.vdot(right_power,delta_grade)
#        trDGTdFT  = cl_array.vdot(left_power,gamma_grade)
#        
#        p0 = -2*(trFGGtdFt+trFTFGdGT-trFTDdGT-trDGTdFT)
        
        
        #Calulate the max change for stoping
        change_calulate(gamma,gamma_new,gamma_grade,queue=queue_gamma_F).get(queue_gamma_F,gamma_change,True)
        change_calulate(delta,delta_new,delta_grade,queue=queue_delta_G).get(queue_delta_G,delta_change,True)
        
        cl.enqueue_barrier(queue_gamma_F).wait()
        cl.enqueue_barrier(queue_delta_G).wait()


        #Calulate F and G using the new values
        calulate_F_G(gamma_new,delta_new) #calulate the new F,G
        calulate_left_power() #and left power.
        calulate_right_power();
        new_error = calulate_error_value()


##        #Begin temp output

#        
#        a_values = np.linspace(0,4*di,50)
#        error_values = np.zeros_like(a_values)
#        dot_values = np.zeros_like(a_values)
#        
#        gamma_cpu = gamma.get(queue_gamma_F)
#        delta_cpu = delta.get(queue_delta_G)
#        
#        gamma_grade_cpu = gamma_grade.get(queue_gamma_F)
#        delta_grade_cpu = delta_grade.get(queue_delta_G)
#        
#        c_error = error_value(gamma_cpu,delta_cpu)
#        di_error = error_value(gamma_cpu + di*gamma_grade_cpu,
#                                             delta_cpu + di*delta_grade_cpu)
#                                             
#        for ii in xrange(a_values.size):
#            a = a_values[ii]
#            error_values[ii] = error_value(gamma_cpu + a*gamma_grade_cpu,
#                                             delta_cpu + a*delta_grade_cpu)
#            dot_values[ii] = c_error -2*(np.vdot(gamma_grade_cpu,nn_fix(a*gamma_grade_cpu+gamma_cpu)-gamma_cpu)+
#                                            np.vdot(delta_grade_cpu,nn_fix(a*delta_grade_cpu+delta_cpu)-delta_cpu))
#
#        #polynomial_error_values =c_error- a_values*p0.get()
#        #print p0
#        plt.figure(1)
#        plt.clf();
#        plt.plot(a_values,error_values)
#        #plt.plot(a_values,polynomial_error_values,'red')
#        plt.plot(a_values,dot_values,'red')
#
#        dp_gamma = 0
#        dp_delta = 0
#        
#        if last_change_gamma is not None:
#            dp_gamma = np.vdot(last_change_gamma,gamma_grade_cpu)/(np.linalg.norm(last_change_gamma)*np.linalg.norm(gamma_grade_cpu))
#            #dp_gamma = np.linalg.norm(nn_fix(gamma_cpu + di*gamma_grade_cpu)-gamma_new.get(queue_gamma_F))
#            
#        if last_change_delta is not None:
#            dp_delta = np.vdot(last_change_delta,delta_grade_cpu)/(np.linalg.norm(last_change_delta)*np.linalg.norm(delta_grade_cpu))
#            #dp_delta = np.linalg.norm(nn_fix(delta_cpu + di*delta_grade_cpu)-delta_new.get(queue_delta_G))
#            
#        last_change_gamma = gamma_grade_cpu
#        last_change_delta = delta_grade_cpu
#        
#        
#        print c_error-trDTD-current_error,di_error-trDTD-new_error #total_change,c_error-di_error,(current_error-new_error),dp_gamma,dp_delta
#        plt.plot(di,di_error,'*')
#        plt.plot(di,c_error-(current_error-new_error),'o')
#
#        #plt.xlim(xmin=0)
#        #plt.ylim((np.min(error_values),c_error))
#        plt.pause(.02)
#        
#        plt.figure(2)
#        plt.plot(current_error-new_error,c_error - di_error,'o')
#        plt.pause(.02)
#        #end temp output   
        
        
        delta_error = current_error-new_error
        #Prevent incresing the error, or a blowup
        if(delta_error < 0 or delta_error> change_target):
            if(delta_error < 0):
                di = di*di_decay
            else:
                di = di*change_target/delta_error

            #since we are not progresing, we need to recalulate the old F and G and left power
            calulate_F_G(gamma,delta)
            calulate_left_power()
            calulate_right_power();

        else:
            cl.enqueue_copy(queue_gamma_F,gamma.data,gamma_new.data)
            cl.enqueue_copy(queue_delta_G,delta.data,delta_new.data)     
            current_error = new_error
        
        #if we move less then the cutoff, we can stop
        if(np.abs(delta_error) < cutt_off):
            break;
         #Begin temp output
#        if itter%5000==0:
#            cl.enqueue_barrier(queue).wait()
#            
#            gamma_cpu = gamma.get(queue)
#            delta_cpu = delta.get(queue)
#            #print np.sum(F,axis=0), np.sum(G,axis=0)
#            norm_gamma_cpu = np.sum(gamma_cpu,axis=0)
#            norm_delta_cpu = np.sum(delta_cpu,axis=0)
#            
#            F_cpu = gamma_cpu/norm_gamma_cpu
#            G_cpu = delta_cpu/norm_delta_cpu
#        
#            DmFG = diff_mat-np.dot(F_cpu,G_cpu);   
#            error_val = np.linalg.norm(DmFG)
#
#            print itter, error_val, (last_error-error_val)/error_val, total_change , (time.time()-t0)/(itter+1)
#            last_error = error_val
#            
#            if display_size is not None:
#               
#                plt.figure(1)
#                plt.clf()
#                plt.imshow(np.argmax(G_cpu,axis=0).reshape(display_size))
#                plt.pause(1)
#                plt.figure(2)
#                plt.clf()
#                plt.imshow(np.argmax(F_cpu,axis=1).reshape(display_size))             
#                plt.pause(1)
#            

#       end temp output        
        

        
        itter += 1
    t1=time.time();
    

    #begin Temp
#    import scipy.spatial.distance
#    print "Error corrilation of :",scipy.spatial.distance.correlation(our_error[2:],gt_error[2:])
#    
#    import matplotlib.pyplot as plt
#    plt.plot(our_error,gt_error)
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.show()
    #end temp


    
    F_cpu = F.get(queue_gamma_F)#gamma_cpu/norm_gamma_cpu
    G_cpu = G.get(queue_delta_G)#delta_cpu/norm_delta_cpu
    
  
    #Temp line:
    print "Total Time: ",t1-t0,"Itters ",itter, "Time per itteration :", (t1-t0)/itter if itter > 0 else 'N/A', "Error value:", current_error

    
    for to_free in to_release:
        to_free.release()

    return F_cpu,G_cpu,current_error


def sparce_mat_power(sp,steps,v):
    v = v.copy();
    
    for _ in xrange(steps):
        v = sp.dot(v);     
        
    return v;
    
def error_term(D,steps,F,G):
#    return (np.vdot(np.dot(F.T,F),np.dot(G,G.T)) 
#                - np.vdot(F,D.dot(G.T))
#                - np.vdot(D.transpose().dot(F).T,G ) )

    return (np.vdot(np.dot(F.T,F),np.dot(G,G.T)) 
                - np.vdot(F,sparce_mat_power(D,steps,G.T))
                - np.vdot(sparce_mat_power(D.transpose(),steps,F).T,G ) )

def F_to_G(diff_mat,steps,F):

    if (F.shape[0] == 0 or F.shape[1] == 0 
            or diff_mat.shape[0] == 0 or diff_mat.shape[1] == 0):
                    return np.zeros((F.shape[1],diff_mat.shape[1]))
    
    DTF = F;
    diff_mat_trans = diff_mat.transpose();
    
    for _ in xrange(steps):
        DTF = diff_mat_trans.dot(DTF);  
        
    sum_targets = np.ones(diff_mat.shape[1])  
    
    FTD2 = DTF.T*2
    FTF2 = np.dot(F.T,F)*2
    one = np.ones((F.shape[1],1))
    multiplyers = np.zeros((diff_mat.shape[1],1))
    
    val = np.inf
    
    while True:
        G = nnls(FTF2,FTD2-np.dot(one,multiplyers.T))
        G *= sum_targets*np.nan_to_num(1/np.sum(G,axis=0))
        multiplyers = np.linalg.lstsq(one,FTD2-np.dot(FTF2,G))[0].T

        nv = np.linalg.norm(np.dot(FTF2,G)-FTD2+np.dot(one,multiplyers.T))
        
        if(val-nv < 1e-8):
            break
        val = nv
        
    return G
                     
                     
def precondition_starting_point(diff_mat,steps):#,accept_ratio=1e-1):
        F = np.zeros((diff_mat.shape[0],0),dtype=np.float32)
        G = np.zeros((0,diff_mat.shape[0]),dtype=np.float32)
        
        unused_points = np.ones(diff_mat.shape[0],np.bool)

        while np.sum(unused_points) > 1:
            #print np.sum(unused_points);
            posible_locations = np.nonzero(unused_points)[0];
            chosen = posible_locations[np.random.randint(posible_locations.size)]
            
            indicator = np.zeros( diff_mat.shape[0],np.float32);
            indicator[chosen] = 1;
            #indicator = np.random.exponential(1.0,F.shape[0])*unused_points;

            for _ in xrange(steps):
                indicator = diff_mat.dot(indicator);
            indicator = indicator/np.sum(indicator)
            
            

#            G = F_to_G(diff_mat,steps,F)
#            new_error = error_term(diff_mat,steps,F,G);
#            if(new_error>last_eror):
#                F = F[:,:-1]
#                G = G[:-1,:]
#                break;
#            else:
#                last_eror = new_error 
#                all_error_valus.append(new_error);
                
            #if(debug_display_vector is not None):
            #    debug_display_vector(F[:,-1]*diff_mat.shape[0])
            #    debug_display_vector(G[-1,:]) 
            #    debug_display_vector(G_tmp[-1,:])
            next_indicator = diff_mat.dot(indicator)
            used_locations = next_indicator < indicator
            
            #used_locations = indicator >= accept_ratio*np.max(indicator)#1/float(diff_mat.shape[0]);# np.logical_and(indicator >= 1/float(diff_mat.shape[0]),unused_points);
            unused_points[used_locations] = 0;
            unused_points[chosen] = 0;
            
            F = np.hstack( (F, indicator.reshape(-1,1) ) );
#            plt.subplot(2,2,1)
#            dv = debug_display_vector(np.log10(indicator));
#            plt.imshow(dv);plt.colorbar();
#            
#            plt.subplot(2,2,2)
#            dv = debug_display_vector(used_locations);
#            plt.imshow(dv);plt.colorbar();  
#            
#            plt.subplot(2,2,3)
#            dv = debug_display_vector(unused_points);
#            plt.imshow(dv);plt.colorbar();          
#            
#
#
#           
#           
#            
#            q = np.arange(0,100);
#            qutail = np.percentile(indicator,q.tolist())
#            
#            import scipy.stats
#            slope, intercept, r_value, p_value, std_err =scipy.stats.linregress(q[90:],np.log(qutail[90:]))
#
#            
#            
#            ax = plt.subplot(2,2,4)
#            #plt.plot(qutail[:-1],np.diff(qutail)/q[:-1])
#            plt.plot(q,qutail,'r')
#            plt.plot(q,np.exp(intercept + slope*q) )
#            ax.set_yscale('log')
#            plt.show();
               
            #plt.hist(indicator*diff_mat.shape[0],500);
            #plt.show();
        G = F_to_G(diff_mat,steps,F)
        new_val = error_term(diff_mat,steps,F,G)
        return F,G,new_val
            
def precondition_starting_point_multiple_runs(diff_mat,steps,precondition_runs,debug_display_vector=None):
    best_val = np.inf
    #last_eror = np.inf

    
    time_since_reset = 0;
    
    while time_since_reset < precondition_runs:
        try:
            #print time_since_reset , precondition_runs , best_val
            F,G,new_val = precondition_starting_point(diff_mat,steps)
            #Run for two steps:
            #print new_val
            #F,G,new_val = \
            #  optimisation_clustering_gradent_decent_probablity_sparce(
            #                 ctx=None,diff_mat=diff_mat,r=int(F.shape[1]),
            #                 steps=steps,max_itter=5,gamma=F,delta=G,
            #                 cutt_off=cutt_off)  
                             
                             
            
            #print(new_val,F.shape[1],time_since_reset)
            if new_val < best_val:
                print best_val, new_val, F.shape[1]
                
                if(debug_display_vector is not None):
                    debug_display_vector(F,G, "new val: %f old val: %f"% (new_val,best_val));
                                   
                best_val = new_val
                best_F = F;
                best_G = G;
                time_since_reset = 0;
                


            else:
                time_since_reset = time_since_reset + 1
            #plt.plot(all_error_valus)
        except Exception,e:
            print "Exception",e
            import traceback
            traceback.print_exc()
            
    if(debug_display_vector is not None):        
        plt.show(block=True); 
    return best_F,best_G, best_val                      
                     
#Whcihc one to use by default
def optimisation_clustering(diff_mat,precondition_runs = 25,
                            steps=5, max_itter =4000,gamma=None,
                            delta=None,cutt_off = 1e-16,
                                            debug_display_vector=None):

    best_F,best_G, best_val = \
        precondition_starting_point_multiple_runs(diff_mat,steps,precondition_runs,debug_display_vector=debug_display_vector)
    print "result ",(best_val,best_F.shape[1])
    

    r  = int(best_F.shape[1]);
    best_gamma,best_delta,best_error = \
      optimisation_clustering_gradent_decent_probablity_sparce(
                     ctx=None,diff_mat=diff_mat,r=r,steps=steps,
                     max_itter=max_itter,gamma=best_F,delta=best_G,
                     cutt_off=cutt_off)
        
        
    

    
    
    return best_gamma,best_delta,best_error
    
    

    
def nnls(A,b):
    if len(b.shape) <= 1:
        return scipy.optimize.nnls(A,b)[0]
        
    x = np.zeros((A.shape[1],b.shape[1]))
    for ii in xrange(b.shape[1]):
        x[:,ii], _ = scipy.optimize.nnls(A,b[:,ii])
    return x
    
#Solves xA=b
def nnls_T(A,b):
    if len(b.shape) <= 1:
        return scipy.optimize.nnls(A.T,b.T)[0].T
        
    x = np.zeros((b.shape[0],A.shape[0]))
    for ii in xrange(b.shape[0]):
        x[ii,:] = scipy.optimize.nnls(A.T,b[ii,:].T)[0].T
    return x    
    
def nnls_F(G,D,sum_targets=None):

    
    if G.shape[0] == 0 or D.shape[0]==0:
        return np.zeros((D.shape[0],G.shape[0]))
    
    if sum_targets is None:
        sum_targets = np.ones(G.shape[0])    

    DG2 = np.dot(D,G.T)*2
    GGT2 = np.dot(G,G.T)*2
    one = np.ones((D.shape[0],1))
    multiplyers = np.zeros((G.shape[0],1))
    
    val = np.inf
    
    while True:
        F = nnls_T(GGT2,DG2-np.dot(one,multiplyers.T))
        F *= np.nan_to_num(sum_targets/np.sum(F,axis=0))
        multiplyers = np.linalg.lstsq(one,DG2-np.dot(F,GGT2))[0].T

        nv = np.linalg.norm(np.dot(F,GGT2)-DG2+np.dot(one,multiplyers.T))

        if((val-nv) < 1e-8):
            break
        val = nv
    
#    va =  np.linalg.norm(np.dot(F,G)-D)
#    othr = np.dot(D,np.linalg.pinv(G));
#    othr[othr < 0] = 0
#    othr *= sum_targets/np.sum(othr,axis=0)
#
#    vb = np.linalg.norm(np.dot(othr,G)-D)
#    
#    print np.sum(othr,axis=0)-sum_targets, np.sum(F,axis=0)-sum_targets
#    if va < vb:
#        print 'ours better', va, vb
#    else:
#        print 'other better', va, vb             
#    
#    print '---------------------------->',np.linalg.norm(np.dot(F,G)-D)
    return F; 

  
def nnls_G(F,D,sum_targets=None):
    
    if sum_targets is None:
        sum_targets = np.ones(D.shape[1])  
        
    if F.shape[1] == 0 or D.shape[1]==0:
        return np.zeros((F.shape[1],D.shape[1]))
    
    FTD2 = np.dot(F.T,D)*2
    FTF2 = np.dot(F.T,F)*2
    one = np.ones((F.shape[1],1))
    multiplyers = np.zeros((D.shape[1],1))
    
    val = np.inf
    
    while True:
        G = nnls(FTF2,FTD2-np.dot(one,multiplyers.T))
        G *= np.nan_to_num(sum_targets/np.sum(G,axis=0))
        multiplyers = np.linalg.lstsq(one,FTD2-np.dot(FTF2,G))[0].T

        nv = np.linalg.norm(np.dot(FTF2,G)-FTD2+np.dot(one,multiplyers.T))
        
        if((val-nv) < 1e-8):
            break
        val = nv
        
    return G 
    
#@profile    
def optimisation_clustering_projection(diff_mat,projection_runs = 1,
                            steps=5,r_guess_precondition_runs=1,
                            debug_display_vector=None, **kwargs):
    best_err = np.Inf

    n = diff_mat.shape[0]
    if(not sparse.issparse(diff_mat)):
        #return optimisation_clustering_gradent_decent_probablity(diff_mat, r, max_itter, cutt_off)
        diff_mat = sparse.csr_matrix(diff_mat)
        

        
    #Guess a projection count 
    best_F,_, _ = \
        precondition_starting_point_multiple_runs(diff_mat,steps,r_guess_precondition_runs,
                                                  debug_display_vector=debug_display_vector)
    r_first_guess=best_F.shape[1]
    del best_F
    proj_count = 10*3*r_first_guess
    
    if proj_count > n:
        print "not enugh points for prjection, falling back to reguler code"
        return optimisation_clustering(diff_mat,steps=steps,debug_display_vector=debug_display_vector,**kwargs)
    
    
    print "Guessing an r of %d with %d projected points" % (r_first_guess,proj_count)
    F_proj_best = None
    G_proj_best = None
    last_exeption = None
    for i in xrange(projection_runs):
        try:
                
            our_purm = np.random.permutation(n)
            selected = our_purm[:proj_count]
            other = our_purm[proj_count:]
    
            D_sel_all = diff_mat[selected,:]
            D_all_sel = diff_mat[:,selected]
            
            ctx = opencl_tools.get_a_context()
            queue = cl.CommandQueue(ctx,properties=opencl_tools.profile_properties);
            to_remove = []
            
            D_sel = D_sel_all[:,selected]
            D_sel_gpu = gpu_sparse_matrix.OpenclCsrMatrix(D_sel,queue=queue);
            to_remove += [D_sel_gpu]
            
            D_so = D_sel_all[:,other]
            D_so_gpu = gpu_sparse_matrix.OpenclCsrMatrix(D_so,queue=queue);
            D_so_T_gpu = gpu_sparse_matrix.OpenclCsrMatrix(D_so.T,queue=queue);
            to_remove += [D_so_gpu, D_so_T_gpu]
            
            D_os = D_all_sel[other,:]
            D_os.sort_indices()
            D_os_gpu = gpu_sparse_matrix.OpenclCsrMatrix(D_os,queue=queue);
            to_remove += [D_os_gpu]
            
            D_othr = diff_mat[other,:][:,other]
            D_othr.sort_indices()
            D_othr_gpu = gpu_sparse_matrix.OpenclCsrMatrix(D_othr,queue=queue);
            D_othr_T_gpu = gpu_sparse_matrix.OpenclCsrMatrix(D_othr.T,queue=queue);
            to_remove += [D_othr_gpu, D_othr_T_gpu]
            
            #Calulate the power of D to the steps for the projected part
            A_sel = np.eye(proj_count,proj_count,dtype=np.float32)
            A_sel_gpu = cl_array.to_device(queue,A_sel)
            A_so = np.zeros((proj_count,n-proj_count),dtype=np.float32) 
            A_so_gpu = cl_array.to_device(queue,A_so)
            A_os = np.zeros((n-proj_count,proj_count),dtype=np.float32) 
            A_os_gpu = cl_array.to_device(queue,A_os)
            to_remove += [A_sel_gpu.data, A_so_gpu.data, A_os_gpu.data]
            
            D_sel_x_A_sel = cl_array.empty(queue,(proj_count,proj_count),np.float32)
            D_so_x_A_os   = cl_array.empty(queue,(proj_count,proj_count),np.float32)
            to_remove += [D_sel_x_A_sel.data, D_so_x_A_os.data]        
            
            D_so_T_x_A_sel_T = cl_array.empty(queue,(proj_count,n-proj_count),np.float32)
            D_othr_T_x_A_so_T = cl_array.empty(queue,(proj_count,n-proj_count),np.float32)
            to_remove += [D_so_T_x_A_sel_T.data, D_othr_T_x_A_so_T.data]  
            
            D_os_x_A_sel = cl_array.empty(queue,(n-proj_count,proj_count),np.float32)
            D_othr_x_A_os = cl_array.empty(queue,(n-proj_count,proj_count),np.float32)
            to_remove += [D_os_x_A_sel.data, D_othr_x_A_os.data]          
            
            for _ in xrange(steps):
                #A_sel_new = D_sel.dot(A_sel) + D_so.dot(A_os)
                #cl.enqueue_barrier(queue)
                #time.sleep(1)
                D_sel_gpu.dot(A_sel_gpu,D_sel_x_A_sel)
                D_so_gpu.dot(A_os_gpu,D_so_x_A_os)
                
                #A_so = (D_so.T.dot(A_sel.T) + D_othr.T.dot(A_so.T) ).T
                #time.sleep(1)
                D_so_T_gpu.dot(__T(A_sel_gpu),__T(D_so_T_x_A_sel_T))
                D_othr_T_gpu.dot(__T(A_so_gpu),__T(D_othr_T_x_A_so_T))
                
                #A_os = D_os.dot(A_sel) + D_othr.dot(A_os)
                #time.sleep(1)
                D_os_gpu.dot(A_sel_gpu,D_os_x_A_sel)
                D_othr_gpu.dot(A_os_gpu,D_othr_x_A_os)
    
                #A_sel = A_sel_new
                #time.sleep(1)
                cl_array.Array._axpbyz(A_sel_gpu,1,D_sel_x_A_sel,1,D_so_x_A_os,queue=queue)
                cl_array.Array._axpbyz(A_so_gpu,1,D_so_T_x_A_sel_T,1,D_othr_T_x_A_so_T,queue=queue)
                cl_array.Array._axpbyz(A_os_gpu,1,D_os_x_A_sel,1,D_othr_x_A_os,queue=queue)
                
                #print A_sel , A_sel_gpu.get(queue)
                #assert np.allclose(A_os,A_os_gpu.get(queue))
                #assert np.allclose(A_sel,A_sel_gpu.get(queue))
                #assert np.allclose(A_so,A_so_gpu.get(queue))            
                
                
            A_sel = A_sel_gpu.get(queue)
            A_so  = A_so_gpu.get(queue)
            A_os  = A_os_gpu.get(queue)
            
            for x in to_remove:
                x.release()
            
            A_consts = np.sum(A_sel,axis=0)
            A_sel /= A_consts
        
            #We have already premultiplyed, so steps is 1
            F_sel,G_sel,_ = optimisation_clustering(A_sel,steps=1,**kwargs)
            
            
            F_proj = np.zeros((n,F_sel.shape[1]))
            G_proj = np.zeros((F_sel.shape[1],n))              
            F_proj[selected,:],G_proj[:,selected] = F_sel, G_sel
            
            F_proj[selected,:] = np.dot(np.diag(A_consts),F_proj[selected,:])
            
            F_proj[other,:] = nnls_F(G_proj[:,selected],A_os,1-np.sum(F_proj[selected,:],axis=0))
            G_proj[:,other] = nnls_G(F_proj[selected,:],A_so)
    
            err = error_term(diff_mat,steps,F_proj,G_proj)
            if err < best_err:
                F_proj_best,G_proj_best,best_err = F_proj,G_proj,err
        except Exception, e:
            last_exeption = e
            import traceback
            traceback.print_exc()

    if F_proj_best is None:
        print "Unsable to find a projected matrix", last_exeption
        return optimisation_clustering(diff_mat,steps=steps,**kwargs)
        
    #Do a final correction        
    #return   F_proj_best,G_proj_best,err
    return optimisation_clustering_gradent_decent_probablity_sparce(None,
                        diff_mat=diff_mat,r=F_proj_best.shape[1],steps=steps,
                        gamma=F_proj_best,delta=G_proj_best,**kwargs)
            
def compare_to_random_base(f,diff_mat,steps=5, max_itter =4000,
                           cutt_off = 1e-16,random_runs = 200,
                           debug_display_vector=None,**kwargs):
    """
        Compares the result of a NMF to ones preformed by a random start
    """                           
    import joblib
    import operator
             
    t0_start = time.time();
    best_F,best_G,best_error = f(diff_mat=diff_mat,steps=steps,
                                 max_itter=max_itter,cutt_off=cutt_off,**kwargs)
    t0_end = time.time();
    
    r= best_F.shape[1]
    

    
    t1_start = time.time();
    all_trys = [];
    preerrors = [];
    for _ in xrange(random_runs):    
        F_rand = np.random.uniform(size=best_F.shape).astype(np.float32)   
        #G_rand = np.random.uniform(size=best_G.shape).astype(np.float32)   #
        G_rand = F_to_G(diff_mat,steps,F_rand)
        
        all_trys.append(
            joblib.delayed
                (optimisation_clustering_gradent_decent_probablity_sparce)
                        (ctx=None,diff_mat=diff_mat,r=r,steps=steps,
                         max_itter=max_itter,gamma=F_rand,delta=G_rand,
                         cutt_off=cutt_off)
            )
        #preerrors.append(np.linalg.norm(diff_mat-F_rand.dot(G_rand) ))
        preerrors.append(error_term(diff_mat,steps,F_rand,G_rand))
    
    all_results = joblib.Parallel(n_jobs=1,verbose=-1)(all_trys)
    error_values =  np.array(map(operator.itemgetter(2),all_results))
    
    #hist, bin_edges = np.histogram(error_values,10);
    #bin_center = bin_edges[1:] - np.diff(bin_edges)/2;
    t1_end = time.time();
    
    print 'Compare of', diff_mat.shape[0], r
    print "Time of new attempt: ", (t0_end-t0_start)
    print "Time of old attempt: ", (t1_end-t1_start)
    
    
    better_values = np.sum(error_values<best_error);
    total_values = error_values.size;
    print "We got ",better_values," better by random out of ", total_values, "trys";
    p =better_values/float(total_values)
    
    #Calulate the a bounds of p, from http://www.sigmazone.com/binomial_confidence_interval.htm
    #http://arxiv.org/pdf/1303.1288.pdf
    a = .05
    import scipy.stats
    
    if better_values == 0:
        p_lb = 0.0
        p_ub = 1 - np.power(a/2,1.0/total_values)
    elif better_values == total_values:
        p_lb = np.power(a/2,1.0/total_values)
        p_ub = 1.0
    else:
        p_lb = scipy.stats.beta.ppf(a/2,better_values,total_values-better_values+1)
        p_ub = scipy.stats.beta.ppf(1-a/2,better_values+1,total_values-better_values)
    
    
    print "chance of being lower estamate of %f with a ci of [%f,%f]" %\
                                            (p,np.float(p_lb),np.float(p_ub))
                                            
    times_per_run = np.float((t1_end-t1_start)/(random_runs))
    t_p = times_per_run/p if p > 0 else np.inf;
    t_p_lb = times_per_run/p_lb if p_lb > 0 else np.inf;
    t_p_ub = times_per_run/p_ub if p_ub > 0 else np.inf;
    print "adjusted time old (for expectation of better hit):",  t_p,\
                "bounds:", (t_p_lb,t_p_ub)

    #print( np.sum(error_values<current_error) )
    #plt.plot(bin_center,hist)
    plt.figure();
    #hist, _, _ = plt.hist(error_values,20,normed=True,histtype='step')
    hist, _, _ = plt.hist(error_values,60,normed=True,histtype='step')
    #hist, _, _ = plt.hist(error_values_new,20,normed=True,histtype='step',color='red')
    plt.vlines(best_error,0,np.max(hist))
    plt.title('Probablity of NMF error')
    plt.xlabel('Relitive error')
    
    
    
    if(debug_display_vector is not None):
        plt.figure();
        out_im = debug_display_vector(np.argmax(best_G,0));
        plt.imshow(out_im);
        plt.colorbar();
        plt.title("new output: %f EC: %f"% (best_error,np.vdot(best_F,best_G.T)) ) 
        
        for test_F,test_G,test_errro in all_results:
            plt.figure();
            debug_display_vector(test_F,test_G, "old output: %f EC: %f"% (test_errro,np.vdot(test_G,test_F.T)));

    
    plt.show(block=True);

    #Pass thrugh result for compatablity 
    return best_F,best_G,best_error
    #print "old best:" ,best_error,"new best:",best_error 
    
    
    

    
if __name__ == '__main__':
        n = 5000
        r = 10
        steps = 2;
        
        def make_random(n,r):
            wight_values = np.random.normal(2*r,1,size=(n,))
            wight_values[wight_values<0] = 0;

            wight_values2 = np.random.normal(r-5,1,size=(n,))
            wight_values2[wight_values2<0] = 0;
            wight_values2[np.random.uniform(size=(n,)) > .1 ] = 0;
            
            placement = np.random.randint(0,r,size=(n,));
            placement2 = np.random.randint(0,r,size=(n,));
            
            gamma = np.random.uniform(size=(n,r)).astype(np.float32)
            gamma[xrange(0,n),placement] += wight_values
            gamma[xrange(0,n),placement2] += wight_values2
            
            delta = np.random.uniform(size=(r,n)).astype(np.float32)
            delta[placement,xrange(0,n)] += wight_values  
            delta[placement2,xrange(0,n)] += wight_values2 
            
            F = gamma/np.sum(gamma,axis=0)
            G = delta/np.sum(delta,axis=0)
            
            return F,G

        our_time_list = []       
        time_list = []
        time_list_corrected = []
        
        our_error_list = []
        error_list = []
        error_list_corrected = []
        gt_error = []
        
        n_values = np.linspace(40*r,n,10,dtype=np.int32)
        
        for n in n_values:
            print "------------------------->",n
            #crate a random sample
            F,G = make_random(n,r)
            D= np.dot(F,G);
            D[D<1e-4] = 0 
            D_pow = np.linalg.matrix_power(D,steps)
            D = sparse.csr.csr_matrix(D)
            
            def error(D,F,G):
                return np.linalg.norm(D-np.dot(F,G))/np.linalg.norm(D)
                
            gt_error.append(error(D_pow,F, G))
            
            #Simple run
            t0 =time.time()
            F_calc,G_calc,_ =  optimisation_clustering(D,steps=steps)
            t1 =time.time()   
                
                

                
            #our_time = t1-t0
            #our_error = error(D,F_calc,G_calc)
                
    
            #Using projection
            
            our_time_list.append(t1-t0)
            our_error_list.append(error(D_pow,F_calc,G_calc))
            
            t2 =time.time()
            F_proj_best,G_proj_best,_=optimisation_clustering_projection(D,steps=steps)
            t3 =time.time()
            time_list.append(t3-t2)
            error_list.append(error(D_pow,F_proj_best, G_proj_best))
            
    
    
            F_corr, G_corr,_ = optimisation_clustering_gradent_decent_probablity_sparce(
                                 ctx=None,diff_mat=D,r=F_proj_best.shape[1],steps=steps,
                                 max_itter=4000,gamma=F_proj_best,delta=G_proj_best,
                                 cutt_off=1e-16)
    
            t4 =time.time()
            time_list_corrected.append(t4-t2)
            error_list_corrected.append(error(D_pow,F_corr, G_corr))
            
            
    #        F_ran, G_ran,_ = optimisation_clustering_gradent_decent_probablity_sparce(
    #                             ctx=None,diff_mat=D,r=F_proj.shape[1],steps=steps,
    #                             max_itter=4000,
    #                             cutt_off=1e-16)        
    #        
    #        t5 =time.time()
        
        
        #Show out put
        plt.figure()
        plt.plot(n_values,time_list,label='Projected')
        plt.plot(n_values,time_list_corrected,label='Corrected')
        plt.plot(n_values,our_time_list,'--',label='our') 
        plt.xlabel('n',fontsize='x-large')
        plt.ylabel('Time (s)',fontsize='x-large')
        plt.legend()
        
        plt.figure()
        plt.plot(n_values,error_list,label='Projected')
        plt.plot(n_values,error_list_corrected,label='Corrected') 
        plt.plot(n_values,our_error_list,'--',label='our') 
        if steps==1: #gt error is only meeningful when steps are 1
            plt.plot(n_values,gt_error,'-',label='ground_truth')
        
        plt.xlabel('n',fontsize='x-large')
        plt.ylabel('Error',fontsize='x-large')        
        plt.legend()
        
        plt.figure()
        plt.plot(time_list,error_list,'o',label='Projected')  
        plt.plot(time_list_corrected,error_list_corrected,'o',label='Corrected') 
        plt.plot(our_time_list,our_error_list,'*',label='our')          
        plt.xlabel('Time (s)',fontsize='x-large')
        plt.ylabel('Error',fontsize='x-large') 
        plt.legend()

          
        plt.show()          
#        print r
#        print "our                  ",error(D,F_calc,G_calc),  "\t", (t1-t0),"s\t r of" ,  F_calc.shape[1];
#        print "projected            ",error(D,F_proj,G_proj),  "\t", (t3-t2),"s\t r of" , F_proj.shape[1];
#        print "projected - corrected",error(D,F_corr, G_corr), "\t", (t4-t2),"s\t r of" ,  F_corr.shape[1];    
##        print "random               ",error(D,F_ran, G_ran),   "\t", (t5-t4),"s\t r of" ,  F_ran.shape[1];
#        
#        #As a base line, do a random sample
#        F_test,G_test = make_random()
#        print "gigo",error(D,F_test,G_test);
#        print "exact",error(D,F,G);
#        #print np.dot(G,F)
