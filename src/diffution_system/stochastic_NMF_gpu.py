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

import joblib

from gpu import opencl_tools
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
elementwise = opencl_tools.cl_elementwise
reduction = opencl_tools.cl_reduction
from gpu import gpu_sparse_matrix
from gpu import gpu_matrix_dot
from gpu import gpu_algorithms

import diffusion_graph

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
    """
    This class runs a GPU kernel that normalizes a dense matrix into a 
    stochastic matrix. It safely handles the case where a column is all zeros. 
    We use it to convert gamma and delta to F and G.  
    
    The size of the normalization vectors must be calculated separately, and 
    passed as an argument. 
    """
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
    """
    This class calculates the gradients of F and G from the gradients of gamma 
    and delta. 
    """
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
    """
    This function calculates the error of a FG segmentation. The value is 
    relative to the value of |D^s|^2, which is not calculated. 
    
    The class takes the following parameters: 
        FT:               The transpose of F
        G:                The G matrix from the factorisation
        left_power_T:     The value (D^sG^T)^T
        right_power:      The value F^TD^s
        FTG:              The value F^TF
        GGT:              The value GG^T
        out:              The matrix to store the output
        queue:            The GPU queue
    """
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
        
def _gpu_transpose(gpu_mat):
    """
    Calulates the transpose of a dense GPU array by changing its metadata. 
    """
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
                          
#A shorthand to make calculating  the transpose of a gpu matrix                          
__T = _gpu_transpose

#Note that steps must be even
def stochastic_NMF_gradient_descent(ctx,diff_mat, r,steps=1, max_itter = 50000,
                                    gamma=None,delta=None,cutt_off = 1e-4,**kwargs):
    """
    This solves the problem D^s=FG for stochastic matrixes F and G, with sizes 
    (n,r) and (r,n) respectively. It takes an n by n matrix D, number of 
    steps s, and rank r as inputs. This method uses a form of gradient descent 
    to find a locally optimal solution from a starting point gamma and delta. 
    (It is Algorithm 5 in supplement 2 of Lockerman et. al. 2016)
    
    Normally, this function would not be called directly. Instead use 
    stochastic_NMF or stochastic_NMF_projection.
    
    Parameters
    ----------
    
    ctx : cl.Context
        The GPU context. Can be NULL, in which case get_a_context() from 
         opencl_tools will be used.
    diff_mat : {matrix, ndarray,sparse matrix}
        An n by n matix who’s power will be factorized
    r : int
        The rank of the factorization. The outputted F will be of size n by r 
        and outputted G will be r by n.  
    steps : int , optional
        The power of diff_mat to be used for the factorization.
    max_itter : int , optional
        The maximum number of descent iterations to use. 
    gamma : n by r matrix , optional
        An initial gamma matrix (that is a non-normalized F matrix). If not 
        provided a random matrix will be used.
    delta : r by n matrix , optional    
        An initial delta matrix (that is a non-normalized G matrix). If not 
        provided a random matrix will be used.  
    cutt_off : float , optional
        The change in error at which we stop the decent.
        
    Returns
    -------
    F : n by r matrix:
        The first matrix of the factorization
    G : r by n matrix:
        The seccond matrix of the factorization   
    relative_error: float
        The error relative to |D^s|^2, which is not calculated.
    """
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
                G.get(queue_delta_G)
                diff_mat_gpu.dot(__T(G),left_power)
    
        def calulate_right_power():
                 diff_mat_T_gpu.dot(F,__T(right_power))     
                 

        
    def calulate_error_value():
        sync_queues()
      
        error_calulator(__T(F),G,__T(left_power),right_power,FTF,GGT,error_temp_stor)
        return float(cl_array.sum(error_temp_stor,np.dtype(np.float32),queue_gamma_F).get(queue_gamma_F))

        

    #t0=time.time()
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


        
        itter += 1
    #t1=time.time();

    F_cpu = F.get(queue_gamma_F)#gamma_cpu/norm_gamma_cpu
    G_cpu = G.get(queue_delta_G)#delta_cpu/norm_delta_cpu
    
  
    #Temp line:
    #print "Total Time: ",t1-t0,"Itters ",itter, "Time per itteration :", (t1-t0)/itter if itter > 0 else 'N/A', "Error value:", current_error

    
    for to_free in to_release:
        to_free.release()

    return F_cpu,G_cpu,current_error




def nnls(A,b):
    """
    Solves Ax=b in a nonnegative least squares sense
    """
    if len(b.shape) <= 1:
        return scipy.optimize.nnls(A,b)[0]
        
    x = np.zeros((A.shape[1],b.shape[1]))
    for ii in xrange(b.shape[1]):
        x[:,ii], _ = scipy.optimize.nnls(A,b[:,ii])
    return x
    
def nnls_T(A,b):
    """
    Solves xA=b nonnegative least squares sense
    """
    if len(b.shape) <= 1:
        return scipy.optimize.nnls(A.T,b.T)[0].T
        
    x = np.zeros((b.shape[0],A.shape[0]))
    for ii in xrange(b.shape[0]):
        x[ii,:] = scipy.optimize.nnls(A.T,b[ii,:].T)[0].T
    return x    
    

def nnls_F(G,D,sum_targets=None):
    """
    Solves D=FG for F given D and G in a nonnegative least squares sense with 
    the constraint that all will be stochastic.
    (This is Algorithm 2 from supplement 2 of Lockerman et. al. 2016.)
    """
    
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

    return F; 


def nnls_G(F,D,sum_targets=None):
    """
    Solves D=FG for G given D and F in a nonnegative least squares sense with 
    the constraint that all will be stochastic.
    (This is Algorithm 3 from supplement 2 of Lockerman et. al. 2016.)
    """
    
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
    
    

def nnls_G_power(diff_mat,steps,F):
    """
    Solves D^steps=FG for G given D and F in a nonnegative least squares sense 
    with the constraint that all will be stochastic.
    
    (This is Algorithm 3 from supplement 2 of Lockerman et. al. 2016.)
    """
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
    
    
def matrix_power(sp,steps,v):
    """
    Calulates sp^steps*v
    """
    v = v.copy();
    
    for _ in xrange(steps):
        v = sp.dot(v);     
        
    return v;
        
    
    
def error_term(D,steps,F,G):
    """
    This function calculates the error of a FG segmentation. The value is 
    relative to the value of D^s, which is not calculated.
    
    Unlike ErrorCalulator, this works on the GPU not CPU
    """
    return (np.vdot(np.dot(F.T,F),np.dot(G,G.T)) 
                - np.vdot(F,matrix_power(D,steps,G.T))
                - np.vdot(matrix_power(D.transpose(),steps,F).T,G ) )

                     
                     
def estimate_single_starting_point(diff_mat,steps):
        """
        Attempts to find a good starting point for the gradient descent algorithm. 
        (This uses Lockerman et. al. 2016 supplement 2 Algorithm 2.)   
        
        Normally, this function is not called directly. Instead 
        estimate_starting_point used to compare multiple starting points. 
        """
        F = np.zeros((diff_mat.shape[0],0),dtype=np.float32)
        G = np.zeros((0,diff_mat.shape[0]),dtype=np.float32)
        
        unused_points = np.ones(diff_mat.shape[0],np.bool)

        while np.sum(unused_points) > 1:
            #print np.sum(unused_points);
            posible_locations = np.nonzero(unused_points)[0];
            chosen = posible_locations[np.random.randint(posible_locations.size)]
            
            indicator = np.zeros( diff_mat.shape[0],np.float32);
            indicator[chosen] = 1;

            for _ in xrange(steps):
                indicator = diff_mat.dot(indicator);
            indicator = indicator/np.sum(indicator)
            

            next_indicator = diff_mat.dot(indicator)
            used_locations = next_indicator < indicator
            
            unused_points[used_locations] = 0;
            unused_points[chosen] = 0;
            
            F = np.hstack( (F, indicator.reshape(-1,1) ) );

        G = nnls_G_power(diff_mat,steps,F)
        new_val = error_term(diff_mat,steps,F,G)
        return F,G,new_val
            
def estimate_starting_point(diff_mat,steps,precondition_runs,
                                        debug_display_vector=None):
    """
     Attempts to find a good starting point for the gradient descent algorithm. 
     (This uses Lockerman et. al. 2016 supplement 2 Algorithm 1.)   
     
    Parameters
    ----------
    diff_mat : {matrix, ndarray,sparse matrix}
        An n by n matix who’s power will be factorized
    steps : int , optional
        The power of diff_mat to be used for the factorization.
    precondition_runs: int
        The number of times to calculate a starting point, without finding an 
        improvement, before accepting the best answer. 
    debug_display_vector: function , optimal 
        A function that takes F,G, and a title. It is called on every vector 
        starting point generated. Can be used for debugging to display 
        starting points.  
        
    Returns
    -------
    F : n by r matrix:
        The first matrix of the factorization
    G : r by n matrix:
        The seccond matrix of the factorization   
    relative_error: float
        The error relative to |D^s|^2, which is not calculated.  
    """
    best_val = np.inf
    #last_eror = np.inf

    time_since_reset = 0;
    
    while time_since_reset < precondition_runs:
        try:
            F,G,new_val = estimate_single_starting_point(diff_mat,steps)

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

        except Exception,e:
            print "Exception",e
            import traceback
            traceback.print_exc()
            
    if(debug_display_vector is not None):
        try:
            import matplotlib.pyplot as plt        
            plt.show(block=True); 
        except:
            print("Could not display debug output, do you have matplotlib?")
    return best_F,best_G, best_val                      
                     
#Whcihc one to use by default
def stochastic_NMF(diff_mat,precondition_runs = 25,
                            steps=5, max_itter =4000,cutt_off = 1e-16,
                                            debug_display_vector=None):
    """
    This solves the problem D^s=FG for stochastic matrixes F and G, with sizes 
    (n,r) and (r,n) respectively. It takes an n by n matrix, D, and the number 
    of steps, s, as inputs. This function uses estimate_starting_point to find 
    a starting point and r. Aftwewords it uses stochastic_NMF_gradient_descent 
    to find the local minimum.  
 
 
    (It is Algorithm 4 in supplement 2 of Lockerman et. al. 2016)
    
    Normally, this function would not be called directly. Instead use 
    stochastic_NMF or stochastic_NMF_projection.
    
    Parameters
    ----------
    diff_mat : {matrix, ndarray,sparse matrix}
        An n by n matix who’s power will be factorized
    precondition_runs: int
        The number of times to calculate a starting point, without finding an 
        improvement, before accepting the best answer.         
    steps : int , optional
        The power of diff_mat to be used for the factorization.
    max_itter : int , optional
        The maximum number of descent iterations to use. 
    cutt_off : float , optional
        The change in error at which we stop the decent.  
    debug_display_vector: function , optimal 
        A function that takes F,G, and a title. It is called on every vector 
        starting point generated. Can be used for debugging to display 
        starting points.  
        
    Returns
    -------
    F : n by r matrix:
        The first matrix of the factorization
    G : r by n matrix:
        The seccond matrix of the factorization   
    relative_error: float
        The error relative to |D^s|^2, which is not calculated.
    """                       

    best_F,best_G, best_val = \
        estimate_starting_point(diff_mat,steps,precondition_runs,debug_display_vector=debug_display_vector)
    print "result ",(best_val,best_F.shape[1])
    

    r  = int(best_F.shape[1]);
    best_gamma,best_delta,best_error =  stochastic_NMF_gradient_descent(
                                         ctx=None,diff_mat=diff_mat,r=r,
                                         steps=steps,max_itter=max_itter,
                                         gamma=best_F,delta=best_G,
                                         cutt_off=cutt_off)
                            

    return best_gamma,best_delta,best_error
    
    

def stochastic_NMF_projection(diff_mat,projection_runs = 1,steps=5,
                            r_guess_precondition_runs=1,final_correction=True,
                            debug_display_vector=None, **kwargs):
    """
    This solves the problem D^s=FG for stochastic matrixes F and G, with sizes 
    (n,r) and (r,n) respectively. It takes an n by n matrix, D, and the number 
    of steps, s, as inputs. 
    
    If n is larger then 10*3*r (for a value of r estimated through 
    estimate_starting_point), this method will try to project the structure of 
    D to a smaller matrix. It will then attempt to solve that simpler problem 
    and then use it to find the larger factorization. Finally, 
    stochastic_NMF_gradient_descent is used to find the local minimum 
    near the solution. 
 
    (It is Algorithm 6 in supplement 2 of Lockerman et. al. 2016)
    Parameters
    ----------
    diff_mat : {matrix, ndarray,sparse matrix}
        An n by n matix who’s power will be factorized
    precondition_runs: int
        The number of times to calculate a starting point, without finding an 
        improvement, before accepting the best answer.         
    steps : int , optional
        The power of diff_mat to be used for the factorization.
    r_guess_precondition_runs : int , optional
        This is passed to estimate_starting_point as precondition_runs when 
        attempting to estimate r. This estimated r is only used to decide when 
        to use the projection. It is not the final value of r returned. 
    final_correction bool , optinal
        If true, this function will use stochastic_NMF_gradient_descent to 
        attempt to find the local minimum by the projected results. 
    debug_display_vector: function , optimal 
        A function that takes F,G, and a title. It is called on every vector 
        starting point generated. Can be used for debugging to display 
        starting points.  
    
    Additional keyword arguments are accepted. They are passed to 
    stochastic_NMF. See that function for details. 
        
    Returns
    -------
    F : n by r matrix:
        The first matrix of the factorization
    G : r by n matrix:
        The seccond matrix of the factorization   
    relative_error: float
        The error relative to |D^s|^2, which is not calculated.
    """
    best_err = np.Inf

    n = diff_mat.shape[0]
    if(not sparse.issparse(diff_mat)):
        diff_mat = sparse.csr_matrix(diff_mat)
        

        
    #Guess a projection count 
    best_F,_, _ = \
        estimate_starting_point(diff_mat,steps,r_guess_precondition_runs,
                                                  debug_display_vector=debug_display_vector)
    r_first_guess=best_F.shape[1]
    del best_F
    proj_count = 10*3*r_first_guess
    
    if proj_count > n:
        print "not enugh points for prjection, falling back to reguler code"
        return stochastic_NMF(diff_mat,steps=steps,debug_display_vector=debug_display_vector,**kwargs)
    
    
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
            F_sel,G_sel,_ = stochastic_NMF(A_sel,steps=1,**kwargs)
            
            
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
        return stochastic_NMF(diff_mat,steps=steps,**kwargs)

    if final_correction:
        return stochastic_NMF_gradient_descent(None,diff_mat=diff_mat,
                                               r=F_proj_best.shape[1],steps=steps,
                                                gamma=F_proj_best,delta=G_proj_best,
                                                **kwargs)
    else:
        return F_proj_best,G_proj_best,err
            
            
            
def NMF_boosted(diff_mat,main_NMF_algorithum,total_NMF_boost_runs = 6,
                     boosting_calulation_runs = 5, debug_display_vector=None, **kwargs):
    """
    This solves the problem D^s=FG for stochastic matrixes F and G, with sizes 
    (n,r) and (r,n) respectively. It takes an n by n matrix, D, and the number 
    of steps, s, as inputs. 
    
    This algorithm will run main_NMF_algorithum multiple times and then attempt 
    to combine the answers to produce a consistent result. 
    
 
    (It is Algorithm 6 in supplement 8 of Lockerman et. al. 2016)
    Parameters
    ----------
    diff_mat : {matrix, ndarray,sparse matrix}
        An n by n matix who’s power will be factorized
    main_NMF_algorithum : function 
        The clustering algorithm to use. It can be stochastic_NMF or
        stochastic_NMF_projection
    total_NMF_boost_runs : int
        The total number of times to run main_NMF_algorithum
    boosting_calulation_runs : int    
        The total number of times to attempt to combine the different results. 
        The answer with the minimum error is returned. 
    debug_display_vector: function , optimal 
        A function that takes F,G, and a title. It is called on every vector 
        starting point generated. Can be used for debugging to display 
        starting points.  
    
    Additional keyword arguments are accepted. They are passed to 
    main_NMF_algorithum. 
        
    Returns
    -------
    F : n by r matrix:
        The first matrix of the factorization
    G : r by n matrix:
        The seccond matrix of the factorization   
    """
    all_calls = [joblib.delayed(main_NMF_algorithum)
                    (diff_mat,
                      debug_display_vector=debug_display_vector,
                      **kwargs)
                             for _ in xrange(total_NMF_boost_runs)]
                            
    n_jobs = 1 if debug_display_vector is None else 1
    results = joblib.Parallel(n_jobs=n_jobs,verbose=10)(all_calls)
        
    if boosting_calulation_runs > 0:
        all_Fs,all_Gs,texture_count = zip(* ( (F,G,G.shape[0]) for F,G,error in results ) )
  
        all_Fs = diffusion_graph.weight_matrix_to_diffution_matrix(np.hstack(all_Fs))   
        all_Gs = diffusion_graph.weight_matrix_to_diffution_matrix(np.vstack(all_Gs))
        
        simular_map = np.dot(all_Gs,all_Fs)

        G_diff_mat = diffusion_graph.weight_matrix_to_diffution_matrix(simular_map)
        
        
        best_error = np.inf
        best_F_F = None
        best_G_G = None
        
        for idx in xrange(boosting_calulation_runs):
            F_F,G_G,error = stochastic_NMF_projection(G_diff_mat,
                                  steps=2,r_guess_precondition_runs=600,
                                  precondition_runs=500,
                                  debug_display_vector=debug_display_vector)
        
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
            
            
            
            
def compare_to_random_base(f,diff_mat,steps=5, max_itter =4000,
                           cutt_off = 1e-16,random_runs = 200,
                           debug_display_vector=None,**kwargs):
    """
        Compares the result of a NMF to ones preformed by a random start
        This function is used for debuging
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
        G_rand = nnls_G_power(diff_mat,steps,F_rand)
        
        all_trys.append(
            joblib.delayed
                (stochastic_NMF_gradient_descent)
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

    try:
        import matplotlib.pyplot as plt
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
    except:
        print("Could not display output, do you have matplotlib?")

    #Pass thrugh result for compatablity 
    return best_F,best_G,best_error
    #print "old best:" ,best_error,"new best:",best_error 
    
    
    


if __name__ == '__main__':
        n = 3000
        r = 10
        steps = 8;
        
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

        simple_time_list = []       
        time_list = []
        time_list_corrected = []
        time_list_corrected_boosted = []
        
        simple_error_list = []
        error_list = []
        error_list_corrected = []
        error_list_corrected_boosted = []
        gt_error = []
        
        n_values = np.linspace(40*r,n,6,dtype=np.int32)
        
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
            F_calc,G_calc,_ =  stochastic_NMF(D,steps=steps)
            t1 =time.time()   
                
            
            simple_time_list.append(t1-t0)
            simple_error_list.append(error(D_pow,F_calc,G_calc))
            
            t2 =time.time()
            F_proj_best,G_proj_best,_=stochastic_NMF_projection(D,steps=steps,final_correction=False)
            t3 =time.time()
            time_list.append(t3-t2)
            error_list.append(error(D_pow,F_proj_best, G_proj_best))
            
    
    
            F_corr, G_corr,_ = stochastic_NMF_gradient_descent(
                                 ctx=None,diff_mat=D,r=F_proj_best.shape[1],steps=steps,
                                 max_itter=4000,gamma=F_proj_best,delta=G_proj_best,
                                 cutt_off=1e-16)
    
            t4 =time.time()
            time_list_corrected.append(t4-t2)
            error_list_corrected.append(error(D_pow,F_corr, G_corr))
            
            t5 =time.time()
            F_boo, G_boo =  NMF_boosted(D_pow,stochastic_NMF_projection,steps=steps)
            #F_boo, G_boo,_ = stochastic_NMF_gradient_descent(
            #                     ctx=None,diff_mat=D,r=F_boo.shape[1],steps=steps,
            #                     max_itter=4000,gamma=F_boo,delta=G_boo,
            #                     cutt_off=1e-16)
            t6 =time.time()
            time_list_corrected_boosted.append(t4-t2)
            error_list_corrected_boosted.append(error(D_pow,F_boo, G_boo))


        
        #Show output
        try:
            import matplotlib.pyplot as plt
            
            plt.figure()
            plt.plot(n_values,time_list,label='Projected')
            plt.plot(n_values,time_list_corrected,label='Corrected')
            plt.plot(n_values,time_list_corrected_boosted,label='Corrected Boosted')            
            plt.plot(n_values,simple_time_list,'--',label='simple') 
            plt.xlabel('n',fontsize='x-large')
            plt.ylabel('Time (s)',fontsize='x-large')
            plt.legend()
            
            plt.figure()
            plt.plot(n_values,error_list,label='Projected')
            plt.plot(n_values,error_list_corrected,label='Corrected') 
            plt.plot(n_values,error_list_corrected_boosted,label='Corrected Boosted') 
            plt.plot(n_values,simple_error_list,'--',label='Simple') 
            if steps==1: #gt error is only meeningful when steps are 1
                plt.plot(n_values,gt_error,'-',label='ground_truth')
            
            plt.xlabel('n',fontsize='x-large')
            plt.ylabel('Error',fontsize='x-large')        
            plt.legend()
            
            plt.figure()
            plt.plot(time_list,error_list,'o',label='Projected')  
            plt.plot(time_list_corrected,error_list_corrected,'o',label='Corrected') 
            plt.plot(time_list_corrected_boosted,error_list_corrected_boosted,label='Corrected Boosted')             
            plt.plot(simple_time_list,simple_error_list,'*',label='Simple')          
            plt.xlabel('Time (s)',fontsize='x-large')
            plt.ylabel('Error',fontsize='x-large') 
            plt.legend()
    
              
            plt.show()          
        except:
            print("Unable to show gui. Do you have matplotlib?")
            raise