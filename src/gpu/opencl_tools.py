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

Created on Tue Feb 26 13:51:00 2013

@author: Yitzchak David Lockerman
"""
profiler_enabled = False
import os

#first, try to load the config file
try:
    import ConfigParser

    config = ConfigParser.RawConfigParser()
    config.read('gpu_conf.ini')

    openc_platform_index = config.getint('DEFAULT', 'OPENCL_PLATFORM_INDEX')
    openc_device_index = config.getint('DEFAULT', 'OPENCL_DEVICE_INDEX')
    
except:
    import traceback
    traceback.print_exc()
    try:
        openc_platform_index = int(os.getenv('OPENCL_PLATFORM_INDEX', 0))
    except:
        print "Invalid opencl_index selected"
        opencl_index = 0
    profiler_enabled = False
    
    
    try:
        openc_device_index = int(os.getenv('OPENCL_DEVICE_INDEX', 0))
    except:
        print "Invalid opencl_index selected"
        opencl_index = 0
profiler_enabled = False

from string import Template

if not profiler_enabled:
    try:
        import pyopencl as cl;
    except:
        import tkMessageBox
        tkMessageBox.showinfo(title="Needs OpenCL 1.2", message="Unabe to load OpenCL. Please install drivers with Opencl 1.2 suport")
        raise
    import pyopencl.array as cl_array;
    import pyopencl.reduction as cl_reduction
    import pyopencl.elementwise as cl_elementwise
    import pyopencl.algorithm as cl_algorithm
    import pyopencl.clrandom as cl_random
    import pyopencl.scan as cl_scan    
    profile_properties = 0
else:
    import opencl_profile
    cl = opencl_profile.cl
    cl_array = opencl_profile.cl_array
    cl_reduction = opencl_profile.cl_reduction
    cl_elementwise = opencl_profile.cl_elementwise
    cl_scan = opencl_profile.cl_scan    
    cl_random = opencl_profile.cl_random    
    profile_properties = int(cl.command_queue_properties.PROFILING_ENABLE)

import pyopencl.characterize as clchar;


platform = cl.get_platforms()[openc_platform_index]
device = platform.get_devices()[openc_device_index];

print "Selecting platform: %s. device: %s" % (platform,device)

_ctx = cl.Context(platform.get_devices()); 

fast_clargs = clchar.get_fast_inaccurate_build_options(device)
work_item_sizes = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES);


def get_a_context(cpu=False):
    if cpu:
        return cl.Context(cl.get_platforms()[1].get_devices())
        
    return _ctx;
    
_inkernal_reduction_template = Template("""


        if ($blockSize >= 1024) 
        { 
            if ($offset < 512) { $sdata[$offset] += $sdata[$offset + 512*($stepsize)]; } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if ($blockSize >= 512) 
        { 
            if ($offset < 256) { $sdata[$offset] += $sdata[$offset + 256*($stepsize)]; } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if ($blockSize >= 256) 
        { 
            if ($offset < 128) { $sdata[$offset] += $sdata[$offset + 128*($stepsize)]; } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if ($blockSize >= 128) 
        { 
            if ($offset <  64) { $sdata[$offset] += $sdata[$offset +  64*($stepsize)]; } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        
        #ifdef WARPSPEED
        if ($offset < 32)
        #endif
        {
            if ($blockSize >=  64) { if ($offset < 32) {$sdata[$offset] += $sdata[$offset + 32*($stepsize)];} NOWARPBLOCK  }
            if ($blockSize >=  32) { if ($offset < 16) {$sdata[$offset] += $sdata[$offset + 16*($stepsize)];} NOWARPBLOCK  }
            if ($blockSize >=  16) { if ($offset < 8)  {$sdata[$offset] += $sdata[$offset +  8*($stepsize)];} NOWARPBLOCK  }
            if ($blockSize >=   8) { if ($offset < 4)  {$sdata[$offset] += $sdata[$offset +  4*($stepsize)];} NOWARPBLOCK  }
            if ($blockSize >=   4) { if ($offset < 2)  {$sdata[$offset] += $sdata[$offset +  2*($stepsize)];} NOWARPBLOCK  }
            if ($blockSize >=   2) { if ($offset < 1)  {$sdata[$offset] += $sdata[$offset +  1*($stepsize)];} NOWARPBLOCK  }
        } 
""")

def get_inkernal_reduction(array,block_size,index,stepsize="1"):
    return _inkernal_reduction_template.safe_substitute(sdata=array,
                                                        blockSize=block_size,
                                                        offset=index,
                                                        stepsize=stepsize)
                   
_inkernal_dual_reduction_template = Template("""


        if ($blockSize >= 1024) 
        { 
            if ($offset < 512) 
            { 
                $array1[$offset] += $array1[$offset + 512*($stepsize1)]; 
                $array2[$offset] += $array2[$offset + 512*($stepsize2)]; 
            } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if ($blockSize >= 512) 
        { 
            if ($offset < 256) 
            { 
                $array1[$offset] += $array1[$offset + 256*($stepsize1)]; 
                $array2[$offset] += $array2[$offset + 256*($stepsize2)]; 
            } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if ($blockSize >= 256) 
        { 
            if ($offset < 128) 
            { 
                $array1[$offset] += $array1[$offset + 128*($stepsize1)]; 
                $array2[$offset] += $array2[$offset + 128*($stepsize2)]; 
            } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if ($blockSize >= 128) 
        { 
            if ($offset <  64) 
            { 
                $array1[$offset] += $array1[$offset +  64*($stepsize1)]; 
                $array2[$offset] += $array2[$offset +  64*($stepsize2)]; 
            } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        
        #ifdef WARPSPEED
        if ($offset < 32)
        #endif
        {
            if ($blockSize >=  64) 
            { 
                if ($offset < 32) 
                {
                    $array1[$offset] += $array1[$offset + 32*($stepsize1)];
                    $array2[$offset] += $array2[$offset + 32*($stepsize2)];
                } 
                NOWARPBLOCK  
            }
            if ($blockSize >=  32) 
            { 
                if ($offset < 16) 
                {
                    $array1[$offset] += $array1[$offset + 16*($stepsize1)];
                    $array2[$offset] += $array2[$offset + 16*($stepsize2)];
                } 
                    NOWARPBLOCK  
            }
            if ($blockSize >=  16) 
            { 
                if ($offset < 8)  
                {
                    $array1[$offset] += $array1[$offset +  8*($stepsize1)];
                    $array2[$offset] += $array2[$offset +  8*($stepsize2)];
                } 
                NOWARPBLOCK 
            }
            if ($blockSize >=   8) 
            { 
                if ($offset < 4)  
                {
                    $array1[$offset] += $array1[$offset +  4*($stepsize1)];
                    $array2[$offset] += $array2[$offset +  4*($stepsize2)];
                } 
                NOWARPBLOCK  
            }
            if ($blockSize >=   4) 
            { 
                if ($offset < 2)  
                {
                    $array1[$offset] += $array1[$offset +  2*($stepsize1)];
                    $array2[$offset] += $array2[$offset +  2*($stepsize2)];
                } 
                NOWARPBLOCK  
            }
            if ($blockSize >=   2) 
            { 
                if ($offset < 1)  
                {
                    $array1[$offset] += $array1[$offset +  1*($stepsize1)];
                    $array2[$offset] += $array2[$offset +  1*($stepsize2)];
                } 
                NOWARPBLOCK  
            }
        } 
""")

def get_inkernal_dual_reduction(array1,array2,block_size,index,stepsize1="1",stepsize2="1"):
    return _inkernal_dual_reduction_template.safe_substitute(array1=array1,
                                                        array2=array2,
                                                        blockSize=block_size,
                                                        offset=index,
                                                        stepsize1=stepsize1,
                                                        stepsize2=stepsize2)
                                                        
def build_preamble_for_context(ctx,preamble=""):
    warp_size=clchar.get_simd_group_size(ctx.devices[0],0)
    
    if(warp_size >= 32 and False):
        preamble += """
                    #define WARPSPEED
                    #define NOWARPBLOCK
                    """
    else:
        preamble +=  """
                    #define NOWARPBLOCK barrier(CLK_LOCAL_MEM_FENCE);
                     """

    return preamble

def pad_overshoot(global_val,local_val):
    overshoot = global_val % local_val;
    if(overshoot > 0):
        global_val = global_val + (local_val - overshoot)
    return global_val
    
#Arg min reduction kernal
#based on https://github.com/inducer/pyopencl/blob/master/test/test_algorithm.py#L366
def argmin_kernal(context):

    import numpy as np
    mmc_dtype = np.dtype([
        ("cur_min", np.float32),
        ("cur_index", np.int32),
        ("pad", np.int32),
        ])

    name = "argmin_collector"
    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

    mmc_dtype, mmc_c_decl = match_dtype_to_c_struct(device, name, mmc_dtype)
    mmc_dtype = get_or_register_dtype(name, mmc_dtype)


    preamble = mmc_c_decl + r"""//CL//

    argmin_collector mmc_neutral()
    {
        // FIXME: needs infinity literal in real use, ok here
        argmin_collector result;
        result.cur_min = INFINITY;
        result.cur_index = -1;
        return result;
    }

    argmin_collector mmc_from_scalar(float x,int index)
    {
        argmin_collector result;
        result.cur_min = x;
        result.cur_index = index;
        return result;
    }

    argmin_collector agg_mmc(argmin_collector a, argmin_collector b)
    {
        argmin_collector result = a;
        if (b.cur_min < result.cur_min)
        {
            result.cur_min = b.cur_min;
            result.cur_index = b.cur_index;
        }
        return result;
    }

    """

    from pyopencl.reduction import ReductionKernel
    red = ReductionKernel(context, mmc_dtype,
            neutral="mmc_neutral()",
            reduce_expr="agg_mmc(a, b)", map_expr="mmc_from_scalar(x[i],i)",
            arguments="__global int *x", preamble=preamble)

    return red

class ArgumentArray(object):
    """This class handels an array that is passed as an argument. It automaticly 
    moves from the gpu to and from the cpu, and takes care of updating the cpu
    if needed. 
    
    While this class exists, it assums that the cpu array is untouched, unless
    cpu_updated() is called"""
    #@profile
    def __init__(self,queue, array,shape=None,dtype=None,
                     orginal_cpu_readonly=False,force_release_gpu=False):
        #We need to have an array, or the ablity to create an array
        assert (array is not None) or (shape is not None and dtype is not None)
        self._queue = queue 
        
        if array is not None:
            self._array = array;
            self._created_orignal = False
            self._orginaly_on_gpu = isinstance(array,cl_array.Array);
            
            if self._orginaly_on_gpu:
                self._gpu_array = array;
                self._cpu_array = None;
            else:
                self._cpu_array = array;
                self._gpu_array =cl_array.to_device(queue,self._cpu_array)
            
            if shape is not None and array.shape != shape:
                raise ValueError("Array is not in correct shape")
            if dtype is not None and array.dtype != dtype:
                raise ValueError("Array has wrong data type")
            
        else:
            self._gpu_array = cl_array.empty(queue,shape,dtype=dtype)
            self._cpu_array = None
            self._created_orignal = True
            self._orginaly_on_gpu = True;
        self._cpu_readonly = orginal_cpu_readonly
        self._force_release_gpu = force_release_gpu
            
    
    def get_cpu_array(self,queue=None,async=False):
        """
            Get the cpu vertion of the array, fresh fromt the gpu. If you
            change it in any way you should call cpu_updated to put the changes
            back in the gpu.
            
            Any changes on the gpu invalidates this cpu array
        """
        if queue is None:
            queue = self._queue
            
        if self._cpu_array == None or self._cpu_readonly:
            #The cpu was readonly, so create a new one to use for editing
            self._cpu_array = self._gpu_array.get(queue)
            self._cpu_readonly = False;
            
            return self._cpu_array
        else:
            return self._gpu_array.get(queue,self._cpu_array)

    def cpu_updated(self, queue=None, async=False):
        """Lets the class know that the cpu array is updated. It is then 
           placed on the gpu."""
        if queue is None:
            queue = self._queue
        if self._cpu_array is None:
            raise ValueError("No cpu array to use")
        
        self._gpu_array.set(queue,self._cpu_array,async)
        
    def get_gpu_array(self,queue=None):
        """
            get the gpu array. It is safe to use and modify this array until
            release is called (or the with statemetn is exited)
        """
        if queue is None:
            queue = self._queue 
            
        return self._gpu_array;
    
    def release(self, queue=None):
        if queue is None:
            queue = self._queue
            
        if self._orginaly_on_gpu and not self._force_release_gpu:
            return #If we were passed a gpu array, whoever passed it owns it
                   #Pluss, the cpu dies with up
        
        if not self._created_orignal and not self._cpu_readonly:
            #if we were passed a non read only cpu array, we need to update
            #it to the newer values 
            self._gpu_array.get(queue,self._cpu_array)
            self._gpu_array.data.release()
            
    def __enter__(self): 
        return self;
        
    def __exit__(self, exc_type, exc_value, traceback): 
        self.release()