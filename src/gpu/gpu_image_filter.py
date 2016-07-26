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

Created on Tue Jul 08 16:12:59 2014

@author: Yitzchak Lockerman
"""

import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('.')

import opencl_tools
import gpu_algorithms

cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
elementwise = opencl_tools.cl_elementwise
reduction = opencl_tools.cl_reduction

#Based on
#http://www.cmsoft.com.br/index.php?option=com_content&view=category&layout=blog&id=142&Itemid=201

_blur_kernal = """
    //Must define: ROWS,COLS,NUMBER_COLORS,FILTER_WIDTH, FILTER_HEIGHT
    //KERNAL_WIDTH, KERNAL_HEIGHT, half_fwidth, half_fheight,
    //FILTER_2_TO_KERNAL_WIDTH, FILTER_2_TO_KERNAL_HEIGHT
    //ALSO, must define an access for filter
    //filter size must be odd for now.
    
    //To handel boundry, one of the folowing should be defined:
    //BOUNDRY_CLAMP : ignore outside values 
    //BOUNDRY_ZERO : set boundry values to zero 

    
    #define image_at(mat,row,col,color) mat[color+ (row)*NUMBER_COLORS*COLS + (col)*NUMBER_COLORS]

    #define local_buffer_width KERNAL_WIDTH
    #define local_buffer_height KERNAL_HEIGHT
    
    #define buffer_at(row_val,col_val,color) local_buffer[(color) + (col_val)*NUMBER_COLORS + (row_val)*NUMBER_COLORS*local_buffer_width]
    __kernel 
    __attribute__((reqd_work_group_size(NUMBER_COLORS,KERNAL_HEIGHT,KERNAL_WIDTH))) 
    void blur_kernal(__global float* image_in,
                     __global float* image_out,
                     __global float* filter_)
    {
        
        
        
        __local float local_buffer[local_buffer_width*local_buffer_height*
                                                                NUMBER_COLORS];
                                                                
        #if FILTER_WIDTH*FILTER_HEIGHT < 1024
            __local float filter[FILTER_WIDTH*FILTER_HEIGHT];
            event_t copy_data= async_work_group_copy(filter,filter_,FILTER_WIDTH*FILTER_HEIGHT,0);
            #define USING_LOCAL
        #else
            #define filter filter_
        #endif
        const int local_color = get_local_id(0); 
        const int local_r = get_local_id(1); 
        const int local_c = get_local_id(2); 
        
        
        const int group_start_r = get_group_id(1)*KERNAL_HEIGHT;
        const int group_start_c = get_group_id(2)*KERNAL_WIDTH;
        

        const int my_color = get_global_id(0);
        
        float accum = 0.0f;
        float norm_accum = 0.0f;
        
        //For each section this will proccess
        #if FILTER_2_TO_KERNAL_HEIGHT*FILTER_2_TO_KERNAL_WIDTH <= 50
            #pragma unroll
        #endif
        for(int cpy_row = 0; cpy_row < FILTER_2_TO_KERNAL_HEIGHT; cpy_row++ )
            #if FILTER_2_TO_KERNAL_WIDTH <= 3
                #pragma unroll
            #endif
            for(int cpy_col = 0; cpy_col < FILTER_2_TO_KERNAL_WIDTH; cpy_col++)
            {
                const int local_start_row= group_start_r + cpy_row*KERNAL_HEIGHT - half_fheight;
                const int local_start_col= group_start_c + cpy_col*KERNAL_WIDTH - half_fwidth;
                
                //Step 1: Load data to local memory
                {
                    const int copy_row = local_start_row + local_r;
                    const int copy_col =  local_start_col  + local_c;
                    
                    if(copy_row >= 0 && copy_row < ROWS &&
                        copy_col >= 0 && copy_col < COLS &&
                            my_color < NUMBER_COLORS)
                    {
                            buffer_at(local_r,local_c,local_color) = 
                                image_at(image_in,copy_row,copy_col,my_color);
                    }
                    else
                    {
                        #if defined BOUNDRY_ZERO
                            buffer_at(local_r,local_c,local_color) = 0;
                        #elif defined BOUNDRY_CLAMP
                            buffer_at(local_r,local_c,local_color) = NAN;
                        #else
                            error('should define a boundry');
                        #endif
                    }
                    
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                #ifdef USING_LOCAL
                    if(cpy_row == 0 && cpy_col == 0)
                        wait_group_events(1,&copy_data);
                #endif     
                //Step 2: do the filter
                
                {
                    __local float* buffer_itter =&(buffer_at(0,0,local_color));
                    for(int dr=(cpy_row*KERNAL_HEIGHT) - local_r; 
                        dr<local_buffer_height+(cpy_row*KERNAL_HEIGHT) - local_r;dr++)
                    {
                        for(int dc=(cpy_col*KERNAL_WIDTH) - local_c; 
                            dc<local_buffer_width+(cpy_col*KERNAL_WIDTH) - local_c;dc++)
                        {
                            //const int dr = (cpy_row*KERNAL_HEIGHT) - local_r + r;
                            //const int dc = (cpy_col*KERNAL_WIDTH) - local_c + c;
                            
                            if(dr >= 0 && dr < FILTER_HEIGHT  &&
                                dc >= 0 && dc < FILTER_WIDTH && !isnan(*buffer_itter))
                            {

                                /*const float g_val = 
                                    select(filter_at(dr,dc,FILTER_WIDTH,FILTER_HEIGHT),
                                           0.0f,
                                           (int)isnan(*buffer_itter));
                                           
                                 accum += select( g_val* (*buffer_itter),0.0f,(int)isnan(*buffer_itter));
                                 norm_accum += g_val; */

                                const float g_val = filter_at(dr,dc,FILTER_WIDTH,FILTER_HEIGHT);
                                accum += g_val* (*buffer_itter);
                                norm_accum += g_val; 
                            }
                            
                            buffer_itter += NUMBER_COLORS;
                        }
                    }
                    
                }
                
                
                barrier(CLK_LOCAL_MEM_FENCE); 
                
            }
        

            if(get_global_id(1) < ROWS &&
                    get_global_id(2) < COLS && my_color < NUMBER_COLORS )
            {
                image_at(image_out,get_global_id(1),get_global_id(2),my_color) =
                                                norm_accum > 0 ? accum/norm_accum : 0;
            }

    }
    
    
""" 
class ImageFilter:
    
    def __init__(self,image_shape,filter_array,boundry='clamp',queue=None):

        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),
                                   properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue
        
        filter_array =filter_array.astype(np.float32)
            
        self._image_shape = image_shape
        self._filter_size = filter_array.shape;
        self._kernal_size = (4,21)
        

        half_kernal_sizes = (filter_array.shape[0]/2,filter_array.shape[1]/2)
        
        kernal_ratios = (int(np.ceil(2*filter_array.shape[0]/float(self._kernal_size[0]))),
                         int(np.ceil(2*filter_array.shape[1]/float(self._kernal_size[1]))))
        preamble = """
            #define ROWS %d
            #define COLS %d
            #define NUMBER_COLORS %d
        
            #define FILTER_HEIGHT %d
            #define FILTER_WIDTH %d

            #define KERNAL_HEIGHT %d
            #define KERNAL_WIDTH %d
            
            #define half_fheight %d
            #define half_fwidth %d
            
            #define FILTER_2_TO_KERNAL_HEIGHT %d
            #define FILTER_2_TO_KERNAL_WIDTH %d
        """ % (image_shape + 
                filter_array.shape + 
                self._kernal_size + 
                half_kernal_sizes +
                kernal_ratios )

        if(boundry == 'clamp'):
            preamble += "#define BOUNDRY_CLAMP\n"
        elif(boundry == 'zero'):
            preamble += "#define BOUNDRY_CLAMP\n"
        else:
            raise ValueError('Unknown boundry value: %s' % boundry)
            
        if not isinstance(filter_array,cl_array.Array): 
            self._filter = cl_array.to_device(self._queue,filter_array)
        else:
            self._filter = filter_array
            
        preamble += gpu_algorithms.add_matrix_axses_for('filter',self._filter)
        preamble = opencl_tools.build_preamble_for_context\
                                        (self._queue.context,preamble)

        prg = cl.Program(self._queue.context,preamble + _blur_kernal).build(options=[]);
        
        self._blur_kernal = prg.blur_kernal

                                
    def __call__(self,image,ret=None):
        assert image.shape == self._image_shape

        if image.dtype != np.float32:
            image = image.astype(np.float32) 
            
        con_back = False
        if not isinstance(image,cl_array.Array):        
            con_back = True
            image = cl_array.to_device(self._queue,image)
        
        if(ret is None or not isinstance(ret,cl_array.Array)):
            ret_stor = cl_array.empty(self._queue, self._image_shape, 
                                      dtype=np.float32)
        else:
            ret_stor = ret
       
        local_size = (image.shape[2],) + self._kernal_size
        global_size = (image.shape[2],image.shape[0],image.shape[1])
        global_size = map(opencl_tools.pad_overshoot,global_size,local_size)
        
        self._blur_kernal(self._queue,global_size,local_size,
                          image.data,ret_stor.data,self._filter.data)        
        
        if ret is None: #Was not given a return location, create one
            if con_back:
                ret=ret_stor.get(self._queue)
                return ret
            return ret_stor
        elif(not isinstance(ret,cl_array.Array)): #set the cpu return location
            ret[:] = ret_stor.get(self._queue)
            return ret
        else: #we were asked for a gpu location, return it
            return ret_stor


_separable_filter_kernal = """
    //Must define: ROWS,COLS,NUMBER_COLORS,FILTER_WIDTH, FILTER_HEIGHT
    //KERNAL_WIDTH, half_fwidth, half_fheight,local_buffer_width

    //ALSO, must define an access for filter
    //filter size must be odd for now.
    
    //To handel boundry, one of the folowing should be defined:
    //BOUNDRY_CLAMP : ignore outside values 
    //BOUNDRY_ZERO : set boundry values to zero 

    #define image_at(mat,row,col,color) mat[color+ (row)*NUMBER_COLORS*COLS + (col)*NUMBER_COLORS]

    __kernel 
    __attribute__((reqd_work_group_size(NUMBER_COLORS,1,KERNAL_WIDTH))) 
    void separable_filter_kernal(__global float* image_in,
                                 __global float* image_out,
                                 __global float* filter_row_,
                                 __global float* filter_col_)
    {
        
        
        
        __local float local_buffer[local_buffer_width*NUMBER_COLORS];
                                                                
        __local float filter_col[FILTER_WIDTH];
        event_t copy_filter= async_work_group_copy(filter_col,filter_col_,FILTER_WIDTH,0);
        
        __local float filter_row[FILTER_HEIGHT];
        copy_filter= async_work_group_copy(filter_row,filter_row_,FILTER_HEIGHT,copy_filter);
        
        const int local_color = get_local_id(0); 
        const int local_r = get_local_id(1); 
        const int local_c = get_local_id(2); 
        
        
        const int group_start_r = get_group_id(1);
        const int group_start_c = get_group_id(2)*KERNAL_WIDTH;
        

        const int my_color = get_global_id(0);
        
        float accum = 0.0f;
        float norm_accum = 0.0f;
        
        wait_group_events(1,&copy_filter);
        //For each section this will proccess
        for(int cpy_row = 0; cpy_row < FILTER_HEIGHT; cpy_row++ )
        {
            const int local_start_row= group_start_r + cpy_row - half_fheight;
            const float row_filter_value = filter_row[cpy_row];

            if(local_start_row > 0 && local_start_row < ROWS)
            {
                #pragma unroll
                for(int cpy_col = 0; cpy_col < FILTER_WIDTH + KERNAL_WIDTH ; cpy_col+=local_buffer_width)
                {

                    const int local_start_col= group_start_c + cpy_col - half_fwidth;
                    
                    const int copy_col_start = clamp(local_start_col,0,COLS);
                    const int copy_col_end = clamp(local_start_col+local_buffer_width,copy_col_start,COLS);
    
                    const int local_buffer_offset = copy_col_start-local_start_col;
                    const int number_coppy_to_buffer = copy_col_end-copy_col_start;
                    event_t copy_buffer= async_work_group_copy(local_buffer + 
                                                NUMBER_COLORS*local_buffer_offset,
                                                &(image_at(image_in,local_start_row,copy_col_start,0)),
                                                number_coppy_to_buffer*NUMBER_COLORS,0);


                    wait_group_events(1,&copy_buffer);
                    
                    __local float* local_buffer_itter = local_buffer + local_color;
                    //Step 2: do the filter
                    #pragma unroll
                    for(int c=0; c<local_buffer_width;c++)
                    {
    
                            const int dc = (cpy_col + c) - local_c;
                            if(dc >= 0 && dc < FILTER_WIDTH)
                            {
                                
                                const float gval = row_filter_value*filter_col[dc];
                                if(c >= local_buffer_offset && c < local_buffer_offset+number_coppy_to_buffer)
                                {
                                    accum = mad(gval,*local_buffer_itter,accum);
                                    norm_accum += gval;
                                }
                                else
                                {
                                    #if defined BOUNDRY_CLAMP
                                        //Do nothing, we are already clamped
                                    #elif defined  BOUNDRY_ZERO
                                        norm_accum += gval;
                                    #else
                                        error('should define a boundry');
                                    #endif
                                }                            
                            }
                            
                            local_buffer_itter += NUMBER_COLORS;
    
                    }                
                    
                    
                    
                    barrier(CLK_LOCAL_MEM_FENCE); 
                    

                }
            }
            else
            {
                #if defined  BOUNDRY_CLAMP
                    //Do nothing, we are already clamped
                #elif defined BOUNDRY_ZERO
                    norm_accum += row_filter_value; error, this will not work unless the col is normlised fix tthis before use
                #else
                    error('should define a boundry');
                #endif
            }
        }
    
        if(get_global_id(1) < ROWS &&
                get_global_id(2) < COLS && my_color < NUMBER_COLORS )
        {
            image_at(image_out,get_global_id(1),get_global_id(2),my_color) =
                                            norm_accum > 0 ? accum/norm_accum : 0;
        }
    
    }
    
    
""" 
class SeparableImageFilter:
    
    def __init__(self,image_shape,row_filter_array,
                     col_filter_array,boundry='clamp',queue=None):
        assert row_filter_array.shape[1] == 1
        assert col_filter_array.shape[0] == 1
        
        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),
                                   properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue
        
        row_filter_array =row_filter_array.astype(np.float32)
        col_filter_array =col_filter_array.astype(np.float32)
        
        self._image_shape = image_shape
        self._filter_size = (row_filter_array.shape[0],col_filter_array.shape[1]);
        self._kernal_size = (1,64)
        

        half_filter_sizes = (self._filter_size[0]/2,self._filter_size[1]/2)
        
        buffer_width = (self._kernal_size[1]+self._filter_size[1],)
        preamble = """
            #define ROWS %d
            #define COLS %d
            #define NUMBER_COLORS %d
        
            #define FILTER_HEIGHT %d
            #define FILTER_WIDTH %d

            #define KERNAL_HEIGHT %d
            #define KERNAL_WIDTH %d
            
            #define half_fheight %d
            #define half_fwidth %d
            
            #define local_buffer_width %d
        """ % (image_shape + 
                self._filter_size + 
                self._kernal_size + 
                half_filter_sizes +
                buffer_width )

        if(boundry == 'clamp'):
            preamble += "#define BOUNDRY_CLAMP\n"
        elif(boundry == 'zero'):
            preamble += "#define BOUNDRY_CLAMP\n"
        else:
            raise ValueError('Unknown boundry value: %s' % boundry)
            
        if not isinstance(row_filter_array,cl_array.Array): 
            self._row_filter_array = cl_array.to_device(self._queue,row_filter_array)
        else:
            self._row_filter_array = row_filter_array

        if not isinstance(col_filter_array,cl_array.Array): 
            self._col_filter_array = cl_array.to_device(self._queue,col_filter_array)
        else:
            self._col_filter_array = col_filter_array            
            
        preamble = opencl_tools.build_preamble_for_context\
                                        (self._queue.context,preamble)

        prg = cl.Program(self._queue.context,preamble + _separable_filter_kernal).build(options=[]);
        
        self._separable_filter_kernal = prg.separable_filter_kernal

                                
    def __call__(self,image,ret=None):
        assert image.shape == self._image_shape


            
        con_back = False
        if not isinstance(image,cl_array.Array):        
            
            if image.dtype != np.float32:
                image = image.astype(np.float32) 
                
            con_back = True
            image = cl_array.to_device(self._queue,image)
        
        if(ret is None or not isinstance(ret,cl_array.Array)):
            ret_stor = cl_array.empty(self._queue, self._image_shape, 
                                      dtype=np.float32)
        else:
            ret_stor = ret
       
        local_size = (image.shape[2],) + self._kernal_size
        global_size = (image.shape[2],image.shape[0],image.shape[1])
        global_size = map(opencl_tools.pad_overshoot,global_size,local_size)
        
        self._separable_filter_kernal(self._queue,global_size,local_size,
                                      image.data,ret_stor.data,
                                      self._row_filter_array.data,
                                      self._col_filter_array.data)        
        
        if ret is None: #Was not given a return location, create one
            if con_back:
                ret=ret_stor.get(self._queue)
                ret_stor.data.release()
                return ret
            return ret_stor
        elif(not isinstance(ret,cl_array.Array)): #set the cpu return location
            ret[:] = ret_stor.get(self._queue)
            ret_stor.data.release()
            return ret
        else: #we were asked for a gpu location, return it
            return ret_stor
            
    def release(self):
        self._row_filter_array.data.release()
        self._col_filter_array.data.release()
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.release()
           
if __name__ == '__main__':
        #From http://stackoverflow.com/questions/3579568/choosing-a-file-in-python-simple-gui
    from Tkinter import Tk
    from tkFileDialog import askopenfilename
    import matplotlib.pyplot as plt
    import time
    
    Tk().withdraw()
    #file_name = askopenfilename(filetypes=[('image file','*.jpg *.png *.bmp *.tif')])
    file_name = r'C:\Users\ydl2\Pictures\bull.JPG'
    
    if(file_name is None):
        import sys;
        sys.exit(0);
        
    from skimage import io, color
    image_rgb = io.imread(file_name)/255.0
    
    sigma = 5;
    our_range = np.arange(-(3*sigma),3*sigma + 1)
    
    row_filter = np.exp(- (our_range*our_range)/(2*sigma*sigma) ).reshape(-1,1)
    col_filter = np.exp(- (our_range*our_range)/(2*sigma*sigma) ).reshape(1,-1)

    #plt.imshow(g_filter);plt.show()

    iflt = SeparableImageFilter(image_rgb.shape,row_filter,col_filter)
    in_image = image_rgb
    
    #im = plt.imshow(in_image)
    #plt.pause(.1);
    
    t0 = time.time()
    i =0;
    while i < 50:
        i += 1;
        print i
        out_image = iflt(in_image)
        
        #im.set_data(out_image)
        #plt.title(str(i))
        #plt.pause(.1);
        
        in_image = out_image
    t1 = time.time();
    
    print "Took %fs seconds per itter" %  ( (t1-t0)/50) 
           
_separable_filter_kernal_float4 = """
    //Must define: ROWS,COLS,NUMBER_COLORS,FILTER_WIDTH, FILTER_HEIGHT
    //KERNAL_WIDTH, half_fwidth, half_fheight,local_buffer_width

    //ALSO, must define an access for filter
    //filter size must be odd for now.
    
    //To handel boundry, one of the folowing should be defined:
    //BOUNDRY_CLAMP : ignore outside values 
    //BOUNDRY_ZERO : set boundry values to zero 

    #define image_at(mat,row,col) mat[(row)*COLS + (col)]


    __kernel 
    __attribute__((reqd_work_group_size(1,KERNAL_WIDTH,1))) 
    void separable_filter_kernal(__global float4* image_in,
                                 __global float4* image_out,
                                 __global float* filter_row_,
                                 __global float* filter_col_)
    {
        
     
        
        __local float4 local_buffer[local_buffer_width];
                          
            
        __local float filter_col[FILTER_WIDTH];
        event_t copy_filter= async_work_group_copy(filter_col,filter_col_,FILTER_WIDTH,0);
        
        
        __local float filter_row[FILTER_HEIGHT];           
        copy_filter= async_work_group_copy(filter_row,filter_row_,FILTER_HEIGHT,copy_filter);
        
        const int local_r = get_local_id(0); 
        const int local_c = get_local_id(1); 
        
        const int group_start_r = get_group_id(0);
        const int group_start_c = get_group_id(1)*KERNAL_WIDTH;

        float4 accum = 0.0f;
        float norm_accum = 0.0f;
        
        wait_group_events(1,&copy_filter);
        //For each section this will proccess
        for(int cpy_row = 0; cpy_row < FILTER_HEIGHT; cpy_row++ )
        {
            const int local_start_row= group_start_r + cpy_row - half_fheight;
            const float row_filter_value = filter_row[cpy_row];

            if(local_start_row > 0 && local_start_row < ROWS)
            {
                #pragma unroll
                for(int cpy_col = 0; cpy_col < FILTER_WIDTH + KERNAL_WIDTH ; cpy_col+=local_buffer_width)
                {

                    const int local_start_col= group_start_c + cpy_col - half_fwidth;
                    
                    const int copy_col_start = clamp(local_start_col,0,COLS);
                    const int copy_col_end = clamp(local_start_col+local_buffer_width,copy_col_start,COLS);
    
                    const int local_buffer_offset = copy_col_start-local_start_col;
                    const int number_coppy_to_buffer = copy_col_end-copy_col_start;
                    
                    event_t copy_buffer= async_work_group_copy(local_buffer + 
                                                local_buffer_offset,
                                                &(image_at(image_in,local_start_row,copy_col_start)),
                                                number_coppy_to_buffer,0);


                    wait_group_events(1,&copy_buffer);
                    
                    __local float4* local_buffer_itter = local_buffer;
                    //Step 2: do the filter
                    #pragma unroll
                    for(int c=0; c<local_buffer_width;c++)
                    {
    
                            const int dc = (cpy_col + c) - local_c;
                            if(dc >= 0 && dc < FILTER_WIDTH)
                            {
                                
                                const float gval = row_filter_value*filter_col[dc];
                                if(c >= local_buffer_offset && c < local_buffer_offset+number_coppy_to_buffer)
                                {
                                    accum = mad(gval,*local_buffer_itter,accum);
                                    norm_accum += gval;
                                }
                                else
                                {
                                    #if defined BOUNDRY_CLAMP
                                        //Do nothing, we are already clamped
                                    #elif defined  BOUNDRY_ZERO
                                        norm_accum += gval;
                                    #else
                                        error('should define a boundry');
                                    #endif
                                }                            
                            }
                            
                            local_buffer_itter++;
    
                    }                
                    
                    
                    
                    barrier(CLK_LOCAL_MEM_FENCE); 
                    

                }
            }
            else
            {
                #if defined  BOUNDRY_CLAMP
                    //Do nothing, we are already clamped
                #elif defined BOUNDRY_ZERO
                    norm_accum += row_filter_value; error, this will not work unless the col is normlised fix tthis before use
                #else
                    error('should define a boundry');
                #endif
            }
        }
    
        
        if(get_global_id(0) < ROWS && get_global_id(1) < COLS  )
        {
            image_at(image_out,get_global_id(0),get_global_id(1)) =
                                            norm_accum > 0 ? accum/norm_accum : 0;        }
    
    }
    
    
""" 
class SeparableImageFilter_Float4:
    
    def __init__(self,image_shape,row_filter_array,
                     col_filter_array,boundry='clamp',queue=None):
        assert row_filter_array.shape[1] == 1
        assert col_filter_array.shape[0] == 1
        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),
                                   properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue
        
        row_filter_array =row_filter_array.astype(np.float32)
        col_filter_array =col_filter_array.astype(np.float32)
        
        self._image_shape = image_shape
        self._filter_size = (row_filter_array.shape[0],col_filter_array.shape[1]);
        self._kernal_size = (1,256)
        

        half_filter_sizes = (self._filter_size[0]/2,self._filter_size[1]/2)
        
        buffer_width = (self._kernal_size[1]+self._filter_size[1],)
        preamble = """
            #define ROWS %d
            #define COLS %d
            #define NUMBER_COLORS %d
        
            #define FILTER_HEIGHT %d
            #define FILTER_WIDTH %d

            #define KERNAL_HEIGHT %d
            #define KERNAL_WIDTH %d
            
            #define half_fheight %d
            #define half_fwidth %d
            
            #define local_buffer_width %d
        """ % (image_shape + 
                self._filter_size + 
                self._kernal_size + 
                half_filter_sizes +
                buffer_width )

        if(boundry == 'clamp'):
            preamble += "#define BOUNDRY_CLAMP\n"
        elif(boundry == 'zero'):
            preamble += "#define BOUNDRY_CLAMP\n"
        else:
            raise ValueError('Unknown boundry value: %s' % boundry)
            
        if not isinstance(row_filter_array,cl_array.Array): 
            self._row_filter_array = cl_array.to_device(self._queue,row_filter_array)
        else:
            self._row_filter_array = row_filter_array

        if not isinstance(col_filter_array,cl_array.Array): 
            self._col_filter_array = cl_array.to_device(self._queue,col_filter_array)
        else:
            self._col_filter_array = col_filter_array            
            
        preamble = opencl_tools.build_preamble_for_context\
                                        (self._queue.context,preamble)

        prg = cl.Program(self._queue.context,preamble + _separable_filter_kernal_float4).build(options=[]);
        
        self._separable_filter_kernal = prg.separable_filter_kernal

                                
    def __call__(self,image,ret=None):
        assert image.shape == self._image_shape


            
        con_back = False
        if not isinstance(image,cl_array.Array):        
            
            if image.dtype != np.float32:
                image = image.astype(np.float32) 
                
            con_back = True
            image = cl_array.to_device(self._queue,image)
        
        if(ret is None or not isinstance(ret,cl_array.Array)):
            ret_stor = cl_array.empty(self._queue, self._image_shape, 
                                      dtype=np.float32)
        else:
            ret_stor = ret
       
        local_size =  self._kernal_size
        global_size = (image.shape[0],image.shape[1])
        global_size = map(opencl_tools.pad_overshoot,global_size,local_size)
        
        self._separable_filter_kernal(self._queue,global_size,local_size,
                                      image.data,ret_stor.data,
                                      self._row_filter_array.data,
                                      self._col_filter_array.data)        
        
        if ret is None: #Was not given a return location, create one
            if con_back:
                ret=ret_stor.get(self._queue)
                ret_stor.data.release()
                return ret
            return ret_stor
        elif(not isinstance(ret,cl_array.Array)): #set the cpu return location
            ret[:] = ret_stor.get(self._queue)
            ret_stor.data.release()
            return ret
        else: #we were asked for a gpu location, return it
            return ret_stor
            
    def release(self):
        self._row_filter_array.data.release()
        self._col_filter_array.data.release()
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.release()            
            
            
if __name__ == '__main__':
        #From http://stackoverflow.com/questions/3579568/choosing-a-file-in-python-simple-gui
    from Tkinter import Tk
    from tkFileDialog import askopenfilename
    import matplotlib.pyplot as plt
    import time
    
    Tk().withdraw()
    #file_name = askopenfilename(filetypes=[('image file','*.jpg *.png *.bmp *.tif')])
    file_name = r'C:\Users\ydl2\Pictures\bull.JPG'
    
    if(file_name is None):
        import sys;
        sys.exit(0);
        
    from skimage import io, color
    image_rgb = io.imread(file_name)/255.0
    image_rgb = np.concatenate( (image_rgb,np.ones(image_rgb.shape[0:2]+(1,))), axis=2);
    
    sigma = 5;
    our_range = np.arange(-(3*sigma),3*sigma + 1)
    
    row_filter = np.exp(- (our_range*our_range)/(2*sigma*sigma) ).reshape(-1,1)
    col_filter = np.exp(- (our_range*our_range)/(2*sigma*sigma) ).reshape(1,-1)

    #plt.imshow(g_filter);plt.show()

    iflt = SeparableImageFilter_Float4(image_rgb.shape,row_filter,col_filter)
    in_image = image_rgb
    
#    im = plt.imshow(in_image)
#    plt.pause(.1);
#    
    t0 = time.time()
    i =0;
    while i < 50:
        i += 1;
        print i
        out_image = iflt(in_image)

#        im.set_data(out_image)
#        plt.title(str(i))
#        plt.pause(.1);
        
        in_image = out_image
    t1 = time.time();
    
    print "Took %fs seconds per itter" %  ( (t1-t0)/50)        