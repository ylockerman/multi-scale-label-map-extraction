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


Created on Thu Feb 28 16:34:00 2013

This code for matrix multiplication is from the pyopencl examples
"""
from __future__ import division
import time 

import opencl_tools
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_reduction = opencl_tools.cl_reduction

import gpu_algorithms
import numpy as np

KERNEL_CODE = """

// Thread block size
#define BLOCK_SIZE %(block_size)d

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width (columns)
#define HA %(h_a)d // Matrix A height (rows)
#define WB %(w_b)d // Matrix B width (columns)
#define HB WA  // Matrix B height (rows)
#define WC WB  // Matrix C width (columns)
#define HC HA  // Matrix C height (rows)

//index 0 is column
//index 1 is row

/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *

Read more at: http://docs.nvidia.com/cuda/eula/index.html#ixzz4FSviVEda 
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook


This has been modifyed by Yitzchak David Lockerman
Included under CUDA Samples End User License Agreement, Section 2.1.1:
 
 Developer shall have the right to modify and create derivative works with the 
 Source Code. Developer shall own any derivative works ("Derivatives") it 
 creates to the Source Code, provided that Developer uses the Materials in 
 accordance with the terms and conditions of this Agreement. Developer may 
 distribute the Derivatives, provided that all NVIDIA copyright notices and 
 trademarks are propagated and used properly and the Derivatives include the 
 following statement:
 "This software contains source code provided by NVIDIA Corporation."


 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */


#define AS(j, i) As[i + j * BLOCK_SIZE]
#define BS(j, i) Bs[i + j * BLOCK_SIZE]

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! WA is A's width and WB is B's width
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1))) 
void
matrixMul( __global float* C, 
          __global float* A, 
          __global float* B,
          uint x_offset,
          uint y_offset)
{
    __local float As[BLOCK_SIZE*BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE*BLOCK_SIZE];

    // Block index
    int bx = get_group_id(0)*BLOCK_SIZE + x_offset;
    int by = get_group_id(1)*BLOCK_SIZE + y_offset;

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed by the block
    #ifdef A_ROW_MAJOR_ORDER
        int aBegin = WA * by;
    #else
        int aBegin = by;
    #endif
        
    // Index of the first sub-matrix of B processed by the block
    #ifdef B_ROW_MAJOR_ORDER
        int bBegin =  bx;
    #else
        int bBegin =  HB*bx;
    #endif
    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int bloc=0;bloc<WA;bloc+=BLOCK_SIZE) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        #ifdef A_ROW_MAJOR_ORDER
            AS(ty, tx) = (bloc+tx < WA && by+ty < HA) ? 
                            A[aBegin + WA * ty + bloc+tx] : 0.0f;
        #else
            AS(ty, tx) = (bloc+tx < WA && by+ty < HA) ? 
                            A[aBegin + ty + HA*(bloc+tx)] : 0.0f;
        #endif
               
        #ifdef B_ROW_MAJOR_ORDER
            BS(ty, tx) = (bx+tx < WB && bloc+ty < HB) ? 
                            B[bBegin + WB * (bloc+ty) + tx] : 0.0f;
        #else
            BS(ty, tx) = (bx+tx < WB && bloc+ty < HB) ? 
                        B[bBegin + bloc+ty + HB*tx] : 0.0f;              
        #endif
        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    if((bx+tx)<WC && (by+ty) < HC)
        #ifdef C_ROW_MAJOR_ORDER
            C[(by+ty)*WC +(bx+tx)] = Csub;
        #else
            C[(by+ty) +(bx+tx)*HC] = Csub;
        #endif

}

"""

class DotKernal(object):
    
    #creates a kernal to multiply a matrix (gpu, or cpu) of a size by another size
    def __init__(self,a,b_shape,b_order='C',c_order='C',queue=None):
        assert a.shape[1] == b_shape[0]
        assert a.dtype == np.float32
        
        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue
            
        self._block_size = 16

        self.shape = a.shape        
        
        if min(a.shape[0],a.shape[1],b_shape[0],b_shape[1]) < self._block_size:
            self._block_size = min(a.shape[0],a.shape[1],b_shape[0],b_shape[1])
            
        self._b_shape = b_shape
        self._kernel_params  = {"block_size": self._block_size,
                                "w_a":a.shape[1], 
                                "h_a":a.shape[0], 
                                "w_b":b_shape[1]}
        
                      
        self._a_order = gpu_algorithms._get_matrix_order(a)
        self._b_order = gpu_algorithms._get_matrix_order(b_order)
        self._c_order = gpu_algorithms._get_matrix_order(c_order)
        
        
        preamble = ""
        if(self._a_order == 'C'):
            preamble+= "#define A_ROW_MAJOR_ORDER\n"
        if(self._b_order == 'C'):
            preamble+= "#define B_ROW_MAJOR_ORDER\n"
        if(self._c_order == 'C'):
            preamble+= "#define C_ROW_MAJOR_ORDER\n"
            
        full_kernal = preamble + (KERNEL_CODE % self._kernel_params)
        prg = cl.Program(self._queue.context, full_kernal ).build()
                         
        self.kernel = prg.matrixMul
        self.kernel.set_scalar_arg_dtypes([ 
                                            None,#__global float* C, 
                                            None,#__global float* A, 
                                            None,#__global float* B,
                                            np.uint32,#uint x_offset,
                                            np.uint32,#uint y_offset
                                          ])
        #Transfer the matrix to both the gpu, if it is not there already
        if isinstance(a,cl_array.Array):
            self.mat = a;
        else:
            self.mat = cl_array.to_device(self._queue,a)
        
        self.max_batch_size = (2048,2048)
        
    def release(self):
        self.mat.data.release()
        
    def __mul__(self,h_c):
        return self.dot(h_c)
    
    def dot(self,h_b,out=None):
        assert h_b.shape == self._b_shape
        assert h_b.dtype == np.float32
        assert gpu_algorithms._get_matrix_order(h_b) == self._b_order
        
        local_size = (self._block_size,self._block_size)
        if out is not None:
            assert gpu_algorithms._get_matrix_order(out) == self._c_order
            assert out.dtype == np.float32

        if(out is None or not isinstance(out,cl_array.Array) ):
            out_tmp = cl_array.empty(self._queue,
                                 (self._kernel_params["w_a"],h_b.shape[1]),
                                  order = self._c_order,
                                  dtype=np.float32)
        else:
           out_tmp = out
            

        if(not isinstance(h_b,cl_array.Array) ):
            h_b_tmp = cl_array.to_device(self._queue,h_b)
        else:
            h_b_tmp = h_b
        
        global_size = out_tmp.shape[::-1]  
        global_size = map(min,global_size,self.max_batch_size)
        global_size = map(opencl_tools.pad_overshoot,global_size,local_size)
        


        for x in xrange(0,out_tmp.shape[1],global_size[0]):
            for y in xrange(0,out_tmp.shape[0],global_size[1]):
                self.kernel(self._queue,global_size,local_size,
                                out_tmp.data,self.mat.data,h_b_tmp.data,x,y)#.wait();
        cl.enqueue_barrier(self._queue)
        if(out is None):
            return out_tmp
        else:
            if(not isinstance(out,cl_array.Array) ):
                cl.enqueue_copy(self._queue,out,out_tmp.data)
                
        return out
            
if __name__ == '__main__':                          
    a_height =1200L
    a_width = 1200L
    b_width = 1200L
    for a_order in ['F','C']:
        for b_order in ['F','C']:
            for c_order in ['F','C']:
                print "Testing: a order",a_order, "with b order",b_order, "to c order", c_order
                
                b_height=a_width
                c_height = a_height
                c_width = b_width
                
                h_a = np.random.rand(a_height, a_width).astype(np.float32,order=a_order)
                h_b = np.random.rand(b_height, b_width).astype(np.float32,order=b_order)
                
                t0=time.time()
                true_c = np.dot(h_a,h_b)
                t1=time.time()
                
                gpu_dot = DotKernal(h_a,h_b.shape[::-1],
                                        b_order=b_order,
                                        c_order = c_order)
                out = cl_array.empty(gpu_dot._queue,
                                             (gpu_dot._kernel_params["h_a"],
                                              h_b.shape[1]),
                                              order = c_order,
                                              dtype=np.float32)
                h_b_gpu = cl_array.to_device(gpu_dot._queue,h_b)
                t2=time.time()
                
            
                our_c = (gpu_dot.dot(h_b_gpu,out))
                assert our_c is out
                                            
                
                t3=time.time()
                
                our_c=our_c.get(gpu_dot._queue)
                t4=time.time()
                
                
            
                print "\tour time:", t3-t2
                print "\tour time (with transfers):", t4-t1
                print "\tcpu time:", t1-t0
                
            #    t5=time.time()
            #    their_out = cl_array.dot(gpu_dot.mat,h_b_gpu,queue=gpu_dot._queue)
            #    cl.enqueue_barrier(gpu_dot._queue).wait()
            #    t6=time.time()
            #    print "their time:",t6-t5
                
                print "\terror ours:", np.max(np.abs(true_c-our_c)/np.abs(true_c))
            #    there_c=their_out.get(gpu_dot._queue)
            #    print "error there:", np.max(np.abs(true_c-there_c)/np.abs(true_c))