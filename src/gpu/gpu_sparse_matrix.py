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

Created on Sun Feb 24 13:23:39 2013

@author: Yitzchak David Lockerman
"""

import numpy as np

import opencl_tools
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_reduction = opencl_tools.cl_reduction

import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

import gpu_algorithms

_csr_code ="""
    //Must define:
        //num_rows, num_cols,blockSize
        

    //Adapted from Efficient Sparse Matrix-Vector Multiplication on CUDA
    //By Nathan Bell and Michael Garlandy
    #define out_rows num_cols
    __kernel void spmv_csr_vector_kernel(__global const uint * ptr,
                                         __global const uint * indices,
                                         __global const float * data,
                                         __global const float * in,
                                         __global float * out,
                                         const uint row_offset,
                                         const uint col_offset,
                                         const uint max_cols,
                                         __local float* sdata)
    {
        const int offset = get_local_id(0);
        const int local_size = get_local_size(0);
        
        const int row = get_global_id(1)+row_offset; 
        
        
        const int out_col = get_global_id(2) + col_offset;  
        
        const int col_start = ptr[row];
        const int col_end = ptr[row+1];
            
        //Add the offset so we don't overide other attemps to run
        sdata += get_local_id(1)*get_local_size(0);
            
        //Do the innital addition untill we obtain at most local_size ellements
        sdata[offset] = 0.0;
            
        if(row < num_rows && out_col < max_cols)
        {

            for(uint c=col_start+offset;c<col_end;c+=local_size)
            {
                const uint column_index = indices[c];
                #ifdef INPUT_COLUMN_MAJOR
                    const float in_value = in[column_index+out_col*out_rows];
                #else
                    const float in_value = in[column_index*max_cols+out_col];
                #endif
                sdata[offset] += data[c]*in_value;
            }
        }
        
        #ifdef WARPSPEED
        if (blockSize > 32)
        #endif        
            barrier(CLK_LOCAL_MEM_FENCE);  
"""+opencl_tools.get_inkernal_reduction('sdata','blockSize','offset')+ """
        //Return the output
        if (offset == 0 && row < num_rows && out_col < max_cols)
        {
            #ifdef OUTPUT_COLUMN_MAJOR
                out[row+out_col*num_rows] = sdata[0];
            #else
                out[row*max_cols+out_col] = sdata[0];
            #endif
        }
      
    }
"""


    
class OpenclCsrMatrix:

    def __init__(self, arg1, shape=None, dtype=None, copy=False,queue=None):
        """Orginaly taken from scipi's implementation"""
        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue
            
        self.size =-1
        #Default running sizes
        self.rows_per_run = 1000
        self.cols_per_run = 100
        self.local_size =  (16,16,1);
        

        if isinstance(arg1,self.__class__) or sp.isspmatrix(arg1):
            self._set_self( arg1,copy )

        elif isinstance(arg1, tuple):
            if sp.sputils.isshape(arg1):
                # It's a tuple of matrix dimensions (M, N)
                # create empty matrix
                self.shape = arg1   #spmatrix checks for errors here
                M, N = self.shape
                self.data    = cl_array.zeros(self._queue,0, sp.sputils.getdtype(dtype, default=np.float32))
                self.indices = cl_array.zeros(self._queue,0, np.intc)
                self.indptr  = cl_array.zeros(self._queue,self._swap((M,N))[0] + 1, dtype=np.intc)
            else:
                if len(arg1) == 2:
                    # (data, ij) format
                    other = sp.coo_matrix(arg1, shape=shape)
                    self._set_self( other )
                elif len(arg1) == 3:
                    # (data, indices, indptr) format
                    (data, indices, indptr) = arg1
                    self.indices = self._make_gpu_array(indices, copy=copy)
                    self.indptr  = self._make_gpu_array(indptr, copy=copy)
                    self.data    = self._make_gpu_array(data, copy=copy, dtype=sp.sputils.getdtype(dtype, data))
                else:
                    raise ValueError("unrecognized %s_matrix constructor usage" %
                            self.format)

        else:
            #must be dense
            try:
                arg1 = np.asarray(arg1)
            except:
                raise ValueError("unrecognized %s_matrix constructor usage" %
                        self.format)
                        
            if dtype == None:
                dtype = np.float32
                
            self._set_self(sp.coo_matrix(arg1, dtype=dtype) )

        # Read matrix dimensions given, if any
        if shape is not None:
            self.shape = shape   # spmatrix will check for errors
        else:
            if self.shape is None:
                # shape not already set, try to infer dimensions
                try:
                    major_dim = len(self.indptr) - 1
                    minor_dim = self.indices.max() + 1
                except:
                    raise ValueError('unable to infer matrix dimensions')
                else:
                    self.shape = self._swap((major_dim,minor_dim))

        if dtype is not None:
            if(self.data.dtype != dtype):
                old_data = self.data
                self.data = self.data.astype(dtype)
                old_data.data.release()

        #self.check_format(full_check=False)
        self._old_shape = (-1,-1)
        self._expected_order = ("","")   
        
        self.dtype = self.data.dtype
        
        self._kernal_table = dict()
    
    def _make_gpu_array(self,array,copy,dtype=None):
        if isinstance(array, cl_array.Array):
            if(copy):
                ret=array._copy()
            else:
                ret=array
        else:
            ret = cl_array.to_device(self._queue,array)
            
        if(dtype!=None):
            ret = ret.astype(dtype,self._queue)
        
        return ret
        
    def _set_self(self,matrix,copy=False):
        if sp.isspmatrix(matrix):
            csr_mat = matrix.tocsr()
            self.indices = self._make_gpu_array(csr_mat.indices, copy=copy)
            self.indptr  = self._make_gpu_array(csr_mat.indptr, copy=copy)
            self.data    = self._make_gpu_array(csr_mat.data.astype(np.float32), copy=copy)
            self.shape = csr_mat.shape 
        elif isinstance(matrix,self.__class__):
            self.indices = self._make_gpu_array(matrix.indices, copy=copy)
            self.indptr  = self._make_gpu_array(matrix.indptr, copy=copy)
            self.data    = self._make_gpu_array(matrix.data, copy=copy)
            self.shape = matrix.shape   
        else:
           raise ValueError("unrecognized matrix given to _set_self")           
    
    #rebulds the kernal
    def _rebuild(self):
        key = self._expected_order+self._old_shape        
        if key in self._kernal_table:
            self._spmv_csr_vector_kernel = self._kernal_table[key]
            return
            
        preamble="""
        #define blockSize %d
        #define num_rows %d
        #define num_cols %d 
        """ % (self.local_size[:1]+self.shape)

        
        preamble = opencl_tools.build_preamble_for_context\
                                        (self._queue.context,preamble)
                              
        if(self._expected_order[0] == 'F'):
             preamble += "#define INPUT_COLUMN_MAJOR\n"
        if(self._expected_order[1] == 'F'):
             preamble += "#define OUTPUT_COLUMN_MAJOR\n"     
             
        prg = cl.Program(self._queue.context,preamble+_csr_code).build();
        self._spmv_csr_vector_kernel = prg.spmv_csr_vector_kernel
        self._spmv_csr_vector_kernel.set_scalar_arg_dtypes([
                                    None, #__global const uint * ptr,
                                    None, #__global const uint * indices,
                                    None, #__global const float * data,
                                    None, #__global const float * in,
                                    None, #__global float * out,
                                    np.uint32, #const uint row_offset,
                                    np.uint32,#const uint col_offset,
                                    np.uint32,#const uint max_cols
                                    None #__local float* sdata)
                                ])
        self._kernal_table[key] = self._spmv_csr_vector_kernel
        
        
    def release(self):
        self.indices.data.release()
        self.indptr.data.release()
        self.data.data.release()
    
    def copy(self):
        return OpenclCsrMatrix((self.indices,self.indptr,self.data),
                                                         self.shape,copy=True);
        

    def mul(self,other,ret_stor):
        ret_shape = ret_stor.shape
        float32_size = int(np.float32(0).nbytes)

        local_size = self.local_size;
        
        rows_per_run = min(self.rows_per_run,ret_shape[0])        
        cols_per_run = min(self.cols_per_run,ret_shape[1])

        overshoot = rows_per_run % local_size[1]
        rows_per_run = rows_per_run + (local_size[1]-overshoot)
        
        overshoot = cols_per_run % local_size[2]
        cols_per_run = cols_per_run + (local_size[2]-overshoot)        
        
        global_size = (local_size[0],rows_per_run,cols_per_run)

        order_of_mats = (gpu_algorithms._get_matrix_order(other),gpu_algorithms._get_matrix_order(ret_stor))
        
        #We need to recompile the kernal if the size changed
        if(self.shape != self._old_shape or self._expected_order != order_of_mats):
            #Note that we will change back
            self._expected_order = order_of_mats 
            self._old_shape = self.shape 
            self._rebuild() #and rebuild
            
        

        for row in xrange(0,ret_shape[0],rows_per_run):
            for col in xrange(0,ret_shape[1],cols_per_run):
                self._spmv_csr_vector_kernel(self._queue,global_size,local_size,
                                    self.indptr.data,  #__global const uint * ptr,
                                    self.indices.data, #__global const uint * indices,
                                    self.data.data,    # __global const float * data,
                                    other.data,        # __global const float * in,
                                    ret_stor.data,     # __global const float * out,
                                    row,               #const uint row_offset,
                                    col,               #const uint col_offset,
                                    ret_shape[1],#const uint max_cols
                                    cl.LocalMemory(float32_size*local_size[0]*local_size[1]))
        cl.enqueue_barrier(self._queue)
    
    def dot(self,other,ret=None):
        ret_type=np.float32
        

        
        if other.dtype != np.float32:
            other = other.astype(np.float32) 
            
        con_back = False
        if not isinstance(other,cl_array.Array):        
            con_back = True
            if(len(other.shape)<2):
                other = np.reshape(other,(other.shape[0],-1))
                resthape_at_end = True;
            else:
                resthape_at_end = False;
            other = cl_array.to_device(self._queue,other)
            
                
        
        if(ret is None or not isinstance(ret,cl_array.Array)):
            ret_shape = (self.shape[0],other.shape[1])
            ret_stor = cl_array.empty(self._queue, ret_shape, 
                                      dtype=ret_type)
        else:
            ret_stor = ret
            
        self.mul(other,ret_stor)
        
        if ret is None: #Was not given a return location, create one
            if con_back:
                ret=ret_stor.get(self._queue)
                if resthape_at_end:
                    ret=np.reshape(ret,(-1))
                return ret
            return ret_stor
        elif(not isinstance(ret,cl_array.Array)): #set the cpu return location
            ret[:] = ret_stor.get(self._queue)
            return ret
        else: #we were asked for a gpu location, return it
            return ret_stor
        
    
    def __mul__(self,other):
        return self.dot(other)
    
    def get(self):
        data = self.data.get(self._queue)
        indices = self.indices.get(self._queue)
        indptr = self.indptr.get(self._queue)
        
        return sp.csr_matrix((data, indices, indptr),self.shape)        
    def conj(self):
        return OpenclCsrMatrix(self.get().conj())

class GPUMatrixLinearOperator(splinalg.LinearOperator):
    def __init__(self, A):
        splinalg.LinearOperator.__init__(self, shape=A.shape, dtype=A.dtype,
                                matvec=None, rmatvec=self.rmatvec)
        self.matvec = A.dot
        self.matmat = A.dot
        self.__mul__ = A.dot
        self.A = A
        self.A_conj = None
        self.size =-1
        
    def rmatvec(self, x):
        if self.A_conj is None:
            self.A_conj = self.A.conj()
        return self.A_conj.dot(x)  


if __name__ == '__main__': 
    import time
    #import matplotlib.pyplot as plt 
    
    test_cpu=sp.rand(32000,32000,density=.15,format="csr",dtype=np.float32)
    dense_cpu = np.random.rand(test_cpu.shape[1],1).astype(np.float32)
    


    t0 =time.time()  
    
    true_val = test_cpu*dense_cpu
    
    t1 =time.time()  
    print "Their time:", t1-t0
    test_gpu=OpenclCsrMatrix(test_cpu)
    
    for in_order in ['F','C']:
        for out_order in ['F','C']:
            print "Testing: In order",in_order, "with out order",out_order
            
        
            dense_gpu = cl_array.to_device(test_gpu._queue,
                                               np.asarray(dense_cpu,
                                                          order=in_order))
            gpu_val = test_gpu*dense_gpu
            gpu_val.data.release()
            gpu_val = cl_array.zeros(test_gpu._queue, gpu_val.shape,
                                        dtype=np.float32,order=out_order)

            t2 =time.time()
            test_gpu.mul(dense_gpu,gpu_val)
            gpuv=gpu_val.get()
        
            t3 =time.time()  
            
            print "\tOur time:", t3-t2
            err = true_val-gpuv
            print "\tMax error:",np.max(np.abs(err)/true_val)            
            dense_gpu.data.release(); del dense_gpu
            gpu_val.data.release(); del gpu_val
    
