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

Created on Mon Dec 30 13:56:44 2013

@author: Yitzchak David Lockerman

This file contains algorthums that are too spicific for opencl_tools
"""

import numpy as np

import opencl_tools
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_reduction = opencl_tools.cl_reduction




def _get_matrix_order(mat):

    """
    Returns the order of a matrix or throws if it is incontinues
    """
    if(isinstance(mat,str)):
        if(mat == "C" or mat == "F"):
            return mat
        else:
            raise ValueError("Unknown matrix order type : %s" % mat)



    if(mat.flags.c_contiguous):
        return "C"
    elif(mat.flags.f_contiguous):
        return "F"
    else:
        raise ValueError("Only contiguous arrays are supported")
        
        
def add_matrix_axses_for(name,mat):
    if _get_matrix_order(mat) == 'C':
        return "\n#define %s_at(r,c,rows,cols) %s[mad_sat((int)(r),(int)(cols),(int)(c))]\n" % (name,name)
    else:
        return "\n#define %s_at(r,c,rows,cols) %s[mad_sat((int)(c),(int)(rows),(int)(r))]\n" % (name,name)

#Sum aglorithum (sums around a row or collom of a matrix)
_sum_code = """
    //Must define: blockSize, DTYPE
    //Must define sum_size,other_size OR SIZE_AS_ARGUMENT
    //ALSO, if sum is over the slow collomn must define SUM_OVER_SLOW_CHANGING
    // if SIZE_AS_ARGUMENT is not defnined
    
    #ifndef SIZE_AS_ARGUMENT
        #ifdef SUM_OVER_SLOW_CHANGING
            #define sum_small_change 1
        #else
            #define sum_small_change 0
        #endif
    #endif
        
    #define from_matrix(mat,sum,other) (sum_small_change ? mat[sum+sum_size*other] : mat[other+other_size*sum])   
    __attribute__((reqd_work_group_size(blockSize, 1, 1)))
    __kernel void sum_per_axis(__global const DTYPE * in,
                                         __global DTYPE * out
                              #ifdef SIZE_AS_ARGUMENT
                                         , int sum_size, int other_size,int sum_small_change
                              #endif
                              )
    {
        __local DTYPE sdata[blockSize];
        const int our_index = get_global_id(1); //The index of the vector that we will sum
        const int offset = get_local_id(0);       

        sdata[offset] = 0;        
        if(our_index < other_size)
        {
            //Load the values
            for(int i = offset; i<sum_size;i+=blockSize )
            {
               sdata[offset] += from_matrix(in,i,our_index);
            }
        }
        
        NOWARPBLOCK
"""+opencl_tools.get_inkernal_reduction('sdata','blockSize','offset')+ """

        if(our_index < other_size && offset == 0)
        {
            out[our_index] = sdata[0];
        }
        
        
    }
"""

class SumKernal(object):
    
    def __init__(self,matrix,axis,queue=None):
        assert axis >= 0 and axis <= 1
        
        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue
           
        self._block_size = 32
        self._axis = axis
        if np.issubdtype(matrix,type):
            self._sizes_given_as_arguments = True
            
            self._dtype = matrix
            self._ctype = cl.tools.dtype_to_ctype(self._dtype)
            
            preamble="""
                #define blockSize %d 
                #define DTYPE %s
                """ % (self._block_size,
                       self._ctype)   
            preamble = opencl_tools.build_preamble_for_context\
                                            (self._queue.context,preamble)
            preamble += "#define SIZE_AS_ARGUMENT\n"
        else:
            self._sizes_given_as_arguments = False
            self._matrix = matrix
            self._matrix_order = _get_matrix_order(matrix)


            #if we are using C major order, we see the sum order as oposit, so
            #exchange the order
            if(self._matrix_order == 'C'):
                self._effective_axis = 1 - axis
            else:
                self._effective_axis = axis
            
            self._dtype = matrix.dtype
            self._ctype = cl.tools.dtype_to_ctype(self._dtype)
            
            preamble="""
                #define sum_size %d
                #define other_size %d
                #define blockSize %d 
                #define DTYPE %s
                """ % (matrix.shape[axis],
                       matrix.shape[1-axis],
                       self._block_size,
                       self._ctype)   
            preamble = opencl_tools.build_preamble_for_context\
                                            (self._queue.context,preamble)
            if(self._effective_axis == 0 ):
                preamble += "#define SUM_OVER_SLOW_CHANGING\n"
            
        prg = cl.Program(self._queue.context,preamble+_sum_code).build();
                         
        self.kernel = prg.sum_per_axis
        
        if self._sizes_given_as_arguments:
            self.kernel.set_scalar_arg_dtypes([None,None,
                                                np.int32,np.int32,np.int32])

    def __call__(self,out=None,matrix=None):

        
        assert matrix is not None or not self._sizes_given_as_arguments
        
        if matrix is None:
            matrix = self._matrix
        elif not self._sizes_given_as_arguments:
            assert matrix.shape == self._matrix.shape
            assert _get_matrix_order(matrix) == self._matrix_order
            assert matrix.dtype == self._dtype
        
        out_size = matrix.shape[1-self._axis]
        
        if(out is not None):
            assert out.size == out_size;
            assert out.dtype == self._dtype
            
            
            
        if(out is None or not isinstance(out,cl_array.Array) ):
            out_tmp = cl_array.empty(self._queue,(out_size),dtype=self._dtype)
        else:
           out_tmp = out
           
        local_size = (self._block_size,1)
        global_size = (self._block_size,int(out_size))   

        if self._sizes_given_as_arguments:
            matrix_order = _get_matrix_order(matrix)
            if(matrix_order == 'C'):
                effective_axis = 1 - self._axis
            else:
                effective_axis = self._axis
            sum_size = matrix.shape[self._axis]
            other_size = matrix.shape[1 - self._axis]
            sum_over_slow_changing = 1 if effective_axis == 0 else 0
        
            self.kernel(self._queue,global_size,local_size,
                           matrix.data,out_tmp.data,sum_size,other_size,sum_over_slow_changing)
        else:
            self.kernel(self._queue,global_size,local_size,matrix.data,out_tmp.data)

           
        cl.enqueue_barrier(self._queue)
        if(out is None):
            return out_tmp
        else:
            if(not isinstance(out,cl_array.Array) ):
                cl.enqueue_copy(self._queue,out,out_tmp.data)
                
        return out
    
    @staticmethod
    def test():
        import time
        queue = cl.CommandQueue(opencl_tools.get_a_context(),properties=opencl_tools.profile_properties)
        
        #test_sum
        cpu_array = np.random.rand(7546*5,6425).astype(np.float32)
        
        true_val = []
        
        for axs in xrange(2):
            t0 =time.time()  
            true_val.append(np.sum(cpu_array,axis = axs))
            t1 =time.time()  
            print "Their time for axis ",axs,":", t1-t0
            
        
        
        
        for in_order in ['F','C']:
            for axs in xrange(2):
                precompute_test_sum = SumKernal(np.float32,axis = axs,queue=queue)
                for size_type in ['size at construction','size at run']:
                    print "Testing: In order",in_order, "with axis",axs, 'and', size_type
                    
                    t1 =time.time()
                    
                    
                    
                    gpu_array = cl_array.to_device(queue,
                                                       np.asarray(cpu_array,
                                                                  order=in_order))

                    if size_type == 'size at construction' :                                     
                        test_sum = SumKernal(gpu_array,axis = axs,queue=queue)
                    else:
                        test_sum = precompute_test_sum
                    t2 =time.time()
                    gpu_val = test_sum(matrix=gpu_array)
                    cl.enqueue_barrier(queue).wait()
                    t3 =time.time()  
                    sum_val=gpu_val.get()
                    t4 = time.time()  
                    
                    
                    print "\tOur time:", t3-t2
                    print "\tOur time with transfers\\construction:", t4-t1
                    err = true_val[axs]-sum_val
                    print "\tMax error:",np.max(np.abs(err)/true_val[axs])            
                    gpu_array.data.release(); del gpu_array
                    gpu_val.data.release(); del gpu_val
                    
      
#Sum-product aglorithum (sums around a row or collom of the products of two matrixes)
_sum_product_code = """
    //Must define: sum_size,other_size,blockSize
    //ALSO, if sum is over the slow collomn must define SUM_OVER_SLOW_CHANGING
    
    #ifdef SUM_OVER_SLOW_CHANGING
        #define from_matrix(mat,sum,other) (mat[sum+sum_size*other])
    #else
        #define from_matrix(mat,sum,other) (mat[other+other_size*sum])
    #endif
                
    __kernel void sum_product_per_axis(__global const float * in1,
                                       __global const float * in2,
                                         __global float * out)
    {
        __local float sdata[blockSize];
        const int our_index = get_global_id(1); //The index of the vector that we will sum
        const int offset = get_local_id(0);       
        
        if(our_index < other_size)
        {
            //Load the values
            sdata[offset] = 0;
            for(int i = offset; i<sum_size;i+=blockSize )
            {
                sdata[offset] += (from_matrix(in1,i,our_index) * from_matrix(in2,i,our_index));
            }
        }
        NOWARPBLOCK
"""+opencl_tools.get_inkernal_reduction('sdata','blockSize','offset')+ """

        if(our_index < other_size && offset == 0)
            out[our_index] = sdata[0];
        
        
    }
"""

class SumProductKernal(object):
    
    def __init__(self,matrix,axis,queue=None):
        assert axis >= 0 and axis <= 1
        
        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue
            
        self._block_size = 32
        self._matrix = matrix
        self._axis = axis


        self._matrix_order = _get_matrix_order(matrix)
        #if we are using C major order, we see the sum order as oposit, so
        #exchange the order
        if(self._matrix_order == 'C'):
            self._effective_axis = 1 - axis
        else:
            self._effective_axis = axis
        
        preamble="""
            #define sum_size %d
            #define other_size %d
            #define blockSize %d 
            """ % (matrix.shape[axis],
                   matrix.shape[1-axis],
                   self._block_size)   
            
        preamble = opencl_tools.build_preamble_for_context\
                                        (self._queue.context,preamble)
        if(self._effective_axis == 0 ):
            preamble += "#define SUM_OVER_SLOW_CHANGING\n"
            
        prg = cl.Program(self._queue.context,preamble+_sum_product_code).build();
                         
        self.kernel = prg.sum_product_per_axis

    def __call__(self,out,other,matrix):

             
        local_size = (self._block_size,1)
        global_size = (self._block_size,self._matrix.shape[1-self._axis])
        assert self._matrix_order == _get_matrix_order(other)         
        assert self._matrix.shape == other.shape
        


        
        if matrix is None:
            matrix = self._matrix
        else:
            assert matrix.shape == self._matrix.shape
            assert _get_matrix_order(matrix) == self._matrix_order
            
        if(out is None or not isinstance(out,cl_array.Array) ):
            out_tmp = cl_array.empty(self._queue,(global_size[1]),dtype=np.float32)
        else:
           out_tmp = out
           
        self.kernel(self._queue,global_size,local_size,
                                matrix.data,
                                other.data,out_tmp.data)

           
        cl.enqueue_barrier(self._queue)
        if(out is None):
            return out_tmp
        else:
            if(not isinstance(out,cl_array.Array) ):
                cl.enqueue_copy(self._queue,out,out_tmp.data)
                
        return out
    
    @staticmethod
    def test():
        import time
        queue = cl.CommandQueue(opencl_tools.get_a_context(),properties=opencl_tools.profile_properties)
        
        #test_sum
        cpu_array1 = np.random.rand(1000,1000).astype(np.float32)
        cpu_array2 = np.random.rand(1000,1000).astype(np.float32)
        
        true_val = []
        
        for axs in xrange(2):
            t0 =time.time()  
            true_val.append(np.sum(cpu_array1*cpu_array2,axis = axs))
            t1 =time.time()  
            print "Their time for axis ",axs,":", t1-t0
            
        
        
        
        for in_order in ['F','C']:
            for axs in xrange(2):
                print "Testing: In order",in_order, "with axis",axs
                
                t1 =time.time()
                
                
                
                gpu_array1 = cl_array.to_device(queue,
                                                   np.asarray(cpu_array1,
                                                              order=in_order))
                gpu_array2 = cl_array.to_device(queue,
                                                   np.asarray(cpu_array2,
                                                              order=in_order))                                                              
                test_sum=SumProductKernal(gpu_array1,axis = axs,queue=queue)
    
                t2 =time.time()
                gpu_val = test_sum(None,gpu_array2,gpu_array1)
                cl.enqueue_barrier(queue).wait()
                t3 =time.time()  
                sum_val=gpu_val.get()
                t4 = time.time()  
                
                
                print "\tOur time:", t3-t2
                print "\tOur time with transfers:", t4-t1
                err = true_val[axs]-sum_val
                print "\tMax error:",np.max(np.abs(err)/true_val[axs])            
                gpu_array1.data.release(); del gpu_array1
                gpu_array2.data.release(); del gpu_array2
                gpu_val.data.release(); del gpu_val
        
#Test the algorithums
if __name__ == '__main__': 
    print "TESTING SumKernal"
    SumKernal.test()
    print "TESTING SumProductKernal"
    SumProductKernal.test()