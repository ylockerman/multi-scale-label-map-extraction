# -*- coding: utf-8 -*-
"""
Created on Mon Jul 07 17:36:35 2014

@author: ydl2
"""

import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('.')

from gpu import opencl_tools
from gpu import gpu_algorithms
from gpu import gpu_image_filter

import bottleneck as bn

cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_algorithm = opencl_tools.cl_algorithm
elementwise = opencl_tools.cl_elementwise
cl_reduction = opencl_tools.cl_reduction
cl_scan = opencl_tools.cl_scan

import SLIC_gpu
import region_map

_center_calulator_kernal_predefine = """
    //Must define: ROWS, COLS, NUMBER_COLORS,NUMBER_OF_SUPERPIXELS
    //KERNAL_WIDTH, NUMBER_OF_KERNALS, LOG_2_KERNAL_SIZE,
    //
    //Also define NUMBER_OF_DIMS, which is the number of values (besides count)
    //being claulated. 
    //By defult just the sum of the colors are clulated however the folowing
    //options are avalable:
    //  CALC_COLOR_SQR - Also sum the square of the color (for std)
    //  CALCULATE_LOCATION - the sum of spatial locations
    //  CALCULATE_LOCATION_SQR - the sum of square spatial locations    
    
    #define NUMBER_OF_ELEMENTS (ROWS*COLS)
    
    #define COLOR_OFFSET (0)
    #define COLOR_SIZE (NUMBER_COLORS)    
    
    #ifdef CALC_COLOR_SQR
        #define SQR_OFFSET (COLOR_OFFSET+COLOR_SIZE)
        #define SQR_SIZE (NUMBER_COLORS)
    #else
        #define SQR_OFFSET (0)
        #define SQR_SIZE (0)
    #endif
    
    #ifdef CALCULATE_LOCATION
        #define LOC_OFFSET (SQR_OFFSET+SQR_SIZE)
        #define LOC_SIZE (2)
    #else
        #define LOC_OFFSET (0)
        #define LOC_SIZE (0)
    #endif    
    
    #ifdef CALCULATE_LOCATION_SQR
        #define LOC_SQR_OFFSET (LOC_OFFSET+LOC_SIZE)
        #define LOC_SQR_SIZE (2)
    #else
        #define LOC_SQR_OFFSET (0)
        #define LOC_SQR_SIZE (0)
    #endif    
    
"""

_center_calulator_kernal ="""

    
    #if NUMBER_OF_DIMS != (COLOR_SIZE+SQR_SIZE+LOC_SIZE)
        number of dims not defined correctly
    #endif
    
    #define sum_at(element,color) sum_in_data[color + element*NUMBER_OF_DIMS]

    #define sum_values_at(cluster_id,color) sum_values[get_group_id(0)*NUMBER_OF_SUPERPIXELS*NUMBER_OF_DIMS + cluster_id*NUMBER_OF_DIMS + color]
 
     //Set up the inital sums for reduction. This is one where each sum is of
     //one pixal
     //Should be started at a size of the image
    __kernel void initiate_reduction_values(__global float* image,
                                    __global float* sums_out)
    {
        const int r = get_global_id(0);
        const int c = get_global_id(1);
        

        __global float* imag_ptr = (image + r*COLS*NUMBER_COLORS
                                 +c*NUMBER_COLORS );                             
        __global float* sum_ptr = (sums_out + r*COLS*NUMBER_OF_DIMS
                                     +c*NUMBER_OF_DIMS );
        

        {
            __global float* img_iter = imag_ptr;
            __global float* color_sum_iter = sum_ptr + COLOR_OFFSET;
            for(int c=0;c<COLOR_SIZE;c++){
                *color_sum_iter = *img_iter;
                color_sum_iter++;
                img_iter++;
            }
        }
        
        #ifdef CALC_COLOR_SQR
        {
            __global float* img_iter = imag_ptr;
            __global float* color_sqr_iter = sum_ptr + SQR_OFFSET;
            for(int c=0;c<COLOR_SIZE;c++){
                *color_sqr_iter = (*img_iter)*(*img_iter);
                color_sqr_iter++;
                img_iter++;
            }
        }
        #endif
        
        #ifdef CALCULATE_LOCATION
        {
            __global float* img_iter = imag_ptr;
            __global float* color_sqr_iter = sum_ptr + LOC_OFFSET;
            color_sqr_iter[0] = r;
            color_sqr_iter[1] = c;
        } 
        #endif
        
        #ifdef CALCULATE_LOCATION_SQR
        {
            __global float* img_iter = imag_ptr;
            __global float* color_sqr_iter = sum_ptr + LOC_SQR_OFFSET;
            color_sqr_iter[0] = r*r;
            color_sqr_iter[1] = c*c;
        }             
        #endif
    }
    
    //Returns the sum of and count of of each cluster
    //Returns true if we are the first element of this cluster
    int center_calulation(const int element_count,
                           __global float* sum_in,
                           __global int* cluster_ids_,
                           __global int* count_in,
                           __local int* cluster_ids,
                           __local float* sum_in_data,
                           __local int* counts,
                           float* local_sum,
                           int* local_count)    
    {
        
        
        const int my_id = get_global_id(0);
        const int local_id = get_local_id(0);
        
        const int starting_point = get_group_id(0)*KERNAL_WIDTH;
        int ending_point = starting_point + KERNAL_WIDTH;
        
        if(starting_point >= element_count)
            return false; //this entire group is outside the image, no reason to run it
        
        if(ending_point > element_count)
            ending_point = element_count;
            
        const int number_of_ellements = ending_point - starting_point;
        
        //Copy the image data
        event_t copy_filter= async_work_group_copy(sum_in_data,sum_in + starting_point * NUMBER_OF_DIMS,
                                                   number_of_ellements*NUMBER_OF_DIMS,0);
        
        //And cluster data
        int my_cluster;
        if(local_id < number_of_ellements)
        {
            my_cluster  =  cluster_ids_[my_id];
            cluster_ids[local_id] = my_cluster;
            counts[local_id] = count_in[my_id];
        }
        (*local_count) = 0;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if(local_id < number_of_ellements)
        {
            //Calulate the sum for each cluster. We only need one thread to store
            //the sum, so it will be the first one that sees it
            //http://www.bealto.com/gpu-sorting_parallel-selection-local.html
            
            for(int i = 0; i < NUMBER_OF_DIMS; i++)
                local_sum[i] = 0;
            
            bool is_first = true;
            
            wait_group_events(1,&copy_filter);
            for (int j=0;j<KERNAL_WIDTH;j++)
            {
                if(j < number_of_ellements)
                {
                    const int other_cluster = cluster_ids[j];
                    const int correct_cluster = (my_cluster == other_cluster); 
        
                    if(correct_cluster)
                    {
                        for(int i = 0; i <NUMBER_OF_DIMS; i++)
                            local_sum[COLOR_OFFSET+i] += sum_at(j,i);

                        (*local_count) += counts[j];
                    }
                    
                    

                    
                    //If we found soemone who came before us with equal value, we are
                    //not first
                    const bool found_cluster_before_us = ( correct_cluster && j < local_id);
                    //Update if we still think we are first
                    is_first = is_first && !found_cluster_before_us;
                }
            }
    
            return is_first;
        }
        
        return false;
    }
    
    __kernel 
    __attribute__((reqd_work_group_size(KERNAL_WIDTH,1,1))) 
    void center_reduction(const int element_count,
                                __global float* sum_in,
                                __global int* cluster_ids_in,
                                __global int* counts_in,
                                __global float* sums_out,
                                __global int* cluster_ids_out,
                                __global int* counts_out,                                    
                                __global int* good_items_for_workgroup)
    {
    
        const int group_offset = get_group_id(0)*KERNAL_WIDTH;
        const int my_id = get_local_id(0);
        
        __local int cluster_ids[KERNAL_WIDTH];
        __local float sum_in_data[KERNAL_WIDTH*NUMBER_OF_DIMS];
        __local int counts[KERNAL_WIDTH];
        
        
        float local_sum[NUMBER_OF_DIMS];
        int local_count;
        
        int is_good_value = center_calulation(element_count,
                                               sum_in, cluster_ids_in,counts_in,
                                               cluster_ids,sum_in_data,counts,
                                               local_sum,&local_count);
                                               
                                               
        __local int place_reduction[KERNAL_WIDTH+1];
        place_reduction[0] = 0;
        place_reduction[my_id + 1] = is_good_value;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int offst = 1; offst < KERNAL_WIDTH+1; offst <<= 1 )
        {
            const int other_place = my_id+1-offst;
            int old_value_add = 0;
            
            if(other_place >= 0 )
                old_value_add = place_reduction[other_place];
            barrier(CLK_LOCAL_MEM_FENCE);
            place_reduction[my_id+1] += old_value_add;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
                                       
        //Find the location to place our data (if we are good)
        int ourplace = place_reduction[my_id];

        //The first thread that gets the sum, must store it in the calulated
        //location
        if(is_good_value)        
        {
            __global float* sums_location = 
                    sums_out +  (group_offset+ourplace)*NUMBER_OF_DIMS;
                        
            for(int i = 0; i < NUMBER_OF_DIMS; i++)
            {
                *sums_location = local_sum[i];
                sums_location++;
            }
            cluster_ids_out[group_offset+ourplace] = cluster_ids[get_local_id(0)];
            counts_out[group_offset+ourplace] = local_count;
        }
        
        //The last thread needs to store how many values we have
        if(get_local_id(0) == KERNAL_WIDTH-1)
            good_items_for_workgroup[get_group_id(0)] = place_reduction[KERNAL_WIDTH];
    }
    
    //Move the locations elements into a row
    //good_items_for_workgroup should be a scan of the value returned above
    __kernel 
    __attribute__((reqd_work_group_size(KERNAL_WIDTH,1,1)))
    void reposition_kernal_elements(__global float* sum_in,
                                    __global int* cluster_ids_in,
                                    __global int* counts_in,
                                    __global float* sums_out,
                                    __global int* cluster_ids_out,
                                    __global int* counts_out,                                    
                                    __global int* good_items_for_workgroup)
    {
        const int old_location = get_global_id(0);
        const int start_offset = get_group_id(0) > 0 ?  
                                    good_items_for_workgroup[get_group_id(0)-1] : 0;
        const int end_offset = good_items_for_workgroup[get_group_id(0)];     
        
        const bool isgoodlocation = get_local_id(0) < end_offset-start_offset;
        
        if(isgoodlocation)
        {
            const int new_location = start_offset + get_local_id(0);
            
            __global float* old_sums_location = 
                    sum_in +  (old_location)*NUMBER_OF_DIMS;             
            __global float* new_sums_location = 
                    sums_out +  (new_location)*NUMBER_OF_DIMS;
                    
            for(int i = 0; i < NUMBER_OF_DIMS; i++)
            {
                *new_sums_location = *old_sums_location;
                old_sums_location++;
                new_sums_location++;
            }
            
            cluster_ids_out[new_location] = cluster_ids_in[old_location];
            counts_out[new_location] = counts_in[old_location];
        }
    }
    
    __kernel 
    __attribute__((reqd_work_group_size(KERNAL_WIDTH,1,1)))
    void final_locations(const int element_count,
                        __global float* sum_in,
                        __global int* cluster_ids_in,
                        __global int* counts_in,
                        __global float* sums_out,
                        __global int* counts_out)
    {
        const int local_id = get_local_id(0);
        
        const int our_histogram_position = get_global_id(0);
        float local_sum[NUMBER_OF_DIMS];
        int local_count = 0;
        for(int i = 0; i < NUMBER_OF_DIMS; i++)
            local_sum[i] = 0;        
        
        __local int cluster_ids[KERNAL_WIDTH];

        //We will look at all the input, in KERNAL size cunks. 
        //Each chunk is sorted, so we can do a binery surch for our location
        for(int chunck_start = 0; 
                    chunck_start < element_count;
                            chunck_start+=KERNAL_WIDTH)
        {
            //Copy all the cluster ids to local memory
            int our_cluster_location = chunck_start+local_id;
            if(our_cluster_location < element_count)
                cluster_ids[local_id] = cluster_ids_in[our_cluster_location];
            barrier(CLK_LOCAL_MEM_FENCE);
            
            if(our_histogram_position < NUMBER_OF_SUPERPIXELS)
            {
                //Do a binery search for our location
                int start_point = 0;
                int end_point = min(KERNAL_WIDTH,
                                    element_count-chunck_start );
                                   
                //For some reson, the binery search is not working, so we
                //just do a brute force search for now.
                /*for(int steps = 0; steps < LOG_2_KERNAL_SIZE; steps++ )
                {
                    int mid_point = (start_point+end_point)/2;
                    
                    if(cluster_ids[mid_point] <  our_histogram_position)
                    {
                        start_point = mid_point;
                    }
                    else if(cluster_ids[mid_point] >  our_histogram_position)
                    {
                        end_point = mid_point;
                    }
                    else
                    {
                        start_point = mid_point;
                        end_point = mid_point + 1;
                    }
                    
                }*/
                
                for(int index = 0; index< end_point; index++)   
                    if(cluster_ids[index] ==  our_histogram_position)
                        start_point = index;
                
                //If we found one of our clusters, add it
                if(start_point < NUMBER_OF_SUPERPIXELS && 
                    cluster_ids[start_point] == our_histogram_position )
                {
                    __global float* input_iter = sum_in +
                            (chunck_start+start_point)*NUMBER_OF_DIMS;
                    for(int i = 0; i < NUMBER_OF_DIMS; i++)
                    {
                        local_sum[i] += *input_iter;
                        input_iter++;
                    }                
                    
                    local_count += counts_in[chunck_start+start_point];
                }

                
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        
        //Finaly, store the result
        if(our_histogram_position < NUMBER_OF_SUPERPIXELS)
        {     
            __global float* out_itter =sums_out+our_histogram_position*NUMBER_OF_DIMS;
            for(int i = 0; i < NUMBER_OF_DIMS; i++)
            {
                *out_itter = local_sum[i];
                out_itter++;
            }                
            
            counts_out[our_histogram_position] = local_count;
        }
        
    }
    
"""


class CenterCalulator:
    def __init__(self,image_shape,total_superpixel_count,
                 reduction_options = [],queue=None):
        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),
                                   properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue

        self._kernal_size = (128,)
        
        self._number_of_kernals = (int(np.ceil(image_shape[0]*image_shape[1]/float(self._kernal_size[0]))),)
        self._global_size = (self._number_of_kernals[0]*self._kernal_size[0],)
        
        self._image_shape = image_shape
        self._total_superpixel_count = total_superpixel_count
        
        preamble = """
            #define ROWS %d
            #define COLS %d
            #define NUMBER_COLORS %d
        
            #define KERNAL_WIDTH %d
            #define NUMBER_OF_KERNALS %d
            #define LOG_2_KERNAL_SIZE %d
        
            #define NUMBER_OF_SUPERPIXELS %d
        """ % (image_shape + 
                self._kernal_size + 
                self._number_of_kernals +
                (int(np.log2(self._kernal_size[0])),) +
                (total_superpixel_count,) )
                
        #Calulate the pramiters for the vector reductions based on our options
        self.reduction_vector_size = image_shape[2]
        self.reduction_vector_prams = ""
        
        if 'square' in reduction_options:
            self.reduction_vector_size += image_shape[2]
            self.reduction_vector_prams += '#define CALC_COLOR_SQR\n'
            
        if 'location' in reduction_options:
            self.reduction_vector_size += 2
            self.reduction_vector_prams += '#define CALCULATE_LOCATION\n'
            
        self.reduction_vector_prams += ('#define NUMBER_OF_DIMS %d\n' 
                                                % self.reduction_vector_size)
        self.reduction_vector_prams += _center_calulator_kernal_predefine
        
        preamble += self.reduction_vector_prams
        preamble = opencl_tools.build_preamble_for_context\
                                        (self._queue.context,preamble)

        prg = cl.Program(self._queue.context,preamble + _center_calulator_kernal).build(options=[]);
        self._initiate_reduction_values=prg.initiate_reduction_values
        self._center_reduction = prg.center_reduction
        self._reposition_kernal_elements = prg.reposition_kernal_elements
        self._final_locations = prg.final_locations
        
        self._center_reduction.set_scalar_arg_dtypes([
                                        np.int32,#const int element_count,
                                        None,#__global float* sum_in,
                                        None,#__global int* cluster_ids_in,
                                        None,#__global int* counts_in,
                                        None,#__global float* sums_out,
                                        None,#__global int* cluster_ids_out,
                                        None,#__global int* counts_out,                                    
                                        None,#__global int* good_items_for_workgroup
                                        ])
                                        
        self._final_locations.set_scalar_arg_dtypes([
                                             np.int32,#const int element_count,
                                             None,#__global float* sum_in,
                                             None,#__global int* cluster_ids_in,
                                             None,#__global int* counts_in,
                                             None,#__global float* sums_out,
                                             None#__global int* counts_out
                                        ])    
        #Buffers for middle calulations
        self._count_buffer = [cl_array.empty(self._queue,image_shape[:2],np.int32) for _ in xrange(2) ]
        
        self._color_sum_buffer = [ cl_array.empty(self._queue,image_shape[:2]+(self.reduction_vector_size,),np.float32 ) for _ in xrange(2) ]
        
        self._cluster_id_buffer =[ cl_array.empty(self._queue,image_shape,np.int32 ) for _ in xrange(2) ]
        self._good_items_for_workgroup =cl_array.empty(self._queue,self._number_of_kernals,np.int32 )
        
        
        #WE need to find the location to start placeing each item
        self._place_results = cl_scan.GenericScanKernel(
                                    self._queue.context, np.int32,
                                    arguments="__global int* ary",
                                    input_expr="ary[i]",
                                    scan_expr="a+b", neutral="0",
                                    output_statement="ary[i] = item;")
                
        
        
        
        
    def __call__(self,image,super_pixals,centers_out,number_out):
        #self.sum_stor.fill(0,self._queue)
        #self.count_stor.fill(0,self._queue);
        self._count_buffer[0].fill(1,self._queue);
        
        assert self._image_shape == image.shape
        assert centers_out.shape[0] == self._total_superpixel_count
        assert centers_out.shape[1] == self.reduction_vector_size
        assert number_out.shape[0] == self._total_superpixel_count


        element_count = image.shape[0]*image.shape[1];
        last_element_count = np.Inf
        
        #set up the reduction values
        self._initiate_reduction_values(self._queue,image.shape,None,
                                        image.data,
                                        self._color_sum_buffer[0].data)
                                                               
        index_buffer = super_pixals
        while element_count < last_element_count:
            last_element_count = element_count
            
            global_size = ( opencl_tools.pad_overshoot(
                                element_count,self._kernal_size[0]),)
            work_group_count = (global_size[0])/self._kernal_size[0]
            
            self._center_reduction(self._queue,
                                   global_size,
                                   self._kernal_size,
                                   element_count,
                                   self._color_sum_buffer[0].data,
                                   index_buffer.data,
                                   self._count_buffer[0].data,
                                   self._color_sum_buffer[1].data,
                                   self._cluster_id_buffer[1].data,
                                   self._count_buffer[1].data,
                                   self._good_items_for_workgroup.data
                                   )
    
            self._place_results(self._good_items_for_workgroup,queue=self._queue)
            

            self._reposition_kernal_elements(self._queue,
                                           global_size,
                                           self._kernal_size,
                                           self._color_sum_buffer[1].data,
                                           self._cluster_id_buffer[1].data,
                                           self._count_buffer[1].data,
                                           self._color_sum_buffer[0].data,
                                           self._cluster_id_buffer[0].data,
                                           self._count_buffer[0].data,
                                           self._good_items_for_workgroup.data
                                           )
                                           

            element_count = self._good_items_for_workgroup.get(self._queue)[work_group_count-1]
            
            index_buffer = self._cluster_id_buffer[0]
            

        global_size = ( opencl_tools.pad_overshoot(
                            self._total_superpixel_count,
                                            self._kernal_size[0]),)   
        

        self._final_locations(self._queue,
                                 global_size,
                                 self._kernal_size,
                                 element_count,
                                 self._color_sum_buffer[1].data,
                                 self._cluster_id_buffer[1].data,
                                 self._count_buffer[1].data,
                                 centers_out.data,
                                 number_out.data)
        

#        full_count =  number_out.get(self._queue)
#                
#        #diff = full_count - ground_truth_full_count
#        #diff_nz = np.nonzero(diff)
#        
#        #print diff_nz, full_count[diff_nz], ground_truth_full_count[diff_nz]
#        #print full_count.shape, self._total_superpixel_count
#            
#        sv = centers_out.get()
#        im_res = sv[super_pixals.get().ravel(),:];
#        count = full_count[super_pixals.get().ravel()]
#        
#        im_res[:,0] /= count;
#        im_res[:,1] /= count;
#        im_res[:,2] /= count;
#        #print im_res[:,2];
#        
#        plt.clf()
#        plt.imshow(im_res.reshape(image.shape),interpolation="none" )
#        plt.pause(.1)
        
        
    def release(self):
        def release_all(buff):
            map(lambda x : x.data.release(),buff)
        
        release_all(self._count_buffer)
        release_all(self._color_sum_buffer)
        release_all(self._cluster_id_buffer)
        self._good_items_for_workgroup.data.release()
        
        
        
_neghbors_kernal = """
    int2 fill_value(int mylable,int outher_value)
    {
        int fv = select(-1,mylable,mylable != outher_value);
        int sv = select(-1,outher_value,mylable != outher_value);
        
        int2 ret;
        ret.x = min(fv,sv);
        ret.y = max(fv,sv);
        
        return ret;
    }


    //Must define:lables_at
    __kernel
    void list_neghbors(__global int* lables,
                           __global int2* neghtbors)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        
        const int rows = get_global_size(0);
        const int cols = get_global_size(1);
        
        const int mylable = lables_at(row,col,rows,cols);
        
        __global int2* our_neghbor_list = neghtbors + 4*col + 4*row*cols;
        
        if( row > 0 )
        {
            const int outher_value = lables_at(row-1,col,rows,cols);
            our_neghbor_list[0] = fill_value(mylable,outher_value);
        }
        else
        {
            our_neghbor_list[0].x = -1;
            our_neghbor_list[0].y = -1;
        }
        
        if( col > 0 )
        {
            const int outher_value = lables_at(row,col-1,rows,cols);
            our_neghbor_list[1] = fill_value(mylable,outher_value);
        }
        else
        {
            our_neghbor_list[1].x = -1;
            our_neghbor_list[1].y = -1;
        }
        
        if( row < rows - 1 )
        {
            const int outher_value = lables_at(row+1,col,rows,cols);
            our_neghbor_list[2] = fill_value(mylable,outher_value);
        }
        else
        {
            our_neghbor_list[2].x = -1;
            our_neghbor_list[2].y = -1;
        }
                
        if( col < cols - 1 )
        {
            const int outher_value = lables_at(row,col+1,rows,cols);
            our_neghbor_list[3] = fill_value(mylable,outher_value);
        }
        else
        {
            our_neghbor_list[3].x = -1;
            our_neghbor_list[3].y = -1;
        }
        
                    
    }
    
    
"""


_combination_calulator = """
    //This returns a int2 indicating which list the pair should be placed into
    //the x element should be 1 if the neghbors should be murged, the y element
    //should be 1 if the pair should be keeped for a future merged
    float combine_error(__global float* first_sum,
                        int first_count,
                        __global float* second_sum,
                        int second_count,
                        float m_S_2_const)
    {
        if(first_count == 0 || second_count == 0) 
            return 0;
        
        __global float* first_color = first_sum+COLOR_OFFSET;
        __global float* second_color = second_sum+COLOR_OFFSET;       


        
        float color_mean_sqr_a = 0;
        float color_mean_sqr_b = 0;

        for(int i = 0; i < NUMBER_COLORS; i++)
        {
            
            float color_mean_a = (*first_color)/first_count;
            float color_mean_b = (*second_color)/second_count;
            float color_mean_ab = ((*first_color) + (*second_color))/
                                            (first_count+second_count);

                                            
            color_mean_sqr_a += (color_mean_a-color_mean_ab)*(color_mean_a-color_mean_ab);
            color_mean_sqr_b += (color_mean_b-color_mean_ab)*(color_mean_b-color_mean_ab);       
            
     
            //Increse the itter
            first_color++;
            second_color++;

        }
        

        float location_mean_sqr_a = 0;
        float location_mean_sqr_b = 0;                   
        
        __global float* first_location = first_sum+LOC_OFFSET;
        __global float* second_location = second_sum+LOC_OFFSET;        
        for(int i = 0; i < 2; i++)
        {

            float location_mean_a = (*first_location)/first_count;
            float location_mean_b = (*second_location)/second_count;
            float location_mean_ab = ((*first_location) + (*second_location))/
                                            (first_count+second_count);
            location_mean_sqr_a += (location_mean_a-location_mean_ab)*(location_mean_a-location_mean_ab);
            location_mean_sqr_b += (location_mean_b-location_mean_ab)*(location_mean_b-location_mean_ab);
            
            first_location++;
            second_location++;
        }
        

        float value =
            ((first_count*color_mean_sqr_a+second_count*color_mean_sqr_b) 
                + m_S_2_const*(first_count*location_mean_sqr_a+second_count*location_mean_sqr_b) );// /(first_count+second_count);
        //float value = (first_count*color_mean_sqr_a+second_count*color_mean_sqr_b);
        return value;
    }
    
    
"""


def calulate_multiscale(image,lowest_level_cluster,
                        highest_scale,m,sigma,ctx = None):
    if ctx is None:
        ctx = opencl_tools.get_a_context()
    devices = ctx.get_info(cl.context_info.DEVICES)
        
    lowest_level_cluster = lowest_level_cluster.astype(np.int32)
    if image.dtype != np.float32:
        image = image.astype(np.float32) 
    assert lowest_level_cluster.shape == image.shape[:2]
    
    color_chanels = image.shape[2]
    total_clusters = np.max(lowest_level_cluster) + 1
    S_org = float(np.sqrt((image.shape[0]*image.shape[1])/total_clusters))

    
    queue = cl.CommandQueue(ctx,properties=opencl_tools.profile_properties)
    
    
    preamble = """
        #define ROWS %d
        #define COLS %d
        #define NUMBER_COLORS %d

    """ % (image.shape )
    
    #First we will create a stuct to handel the data
    image_dtype = np.dtype({'names' : ['f%d' % x for x in xrange(color_chanels)],
                                               'formats' : ['float32']*color_chanels})
    
    image_dtype, image_cdecl = \
        cl.tools.match_dtype_to_c_struct(devices[0], 'image_dtype', 
                                                         image_dtype,ctx)
    image_dtype = cl.tools.get_or_register_dtype('image_dtype', image_dtype)
    
    lowest_level_cluster = cl_array.to_device(queue,lowest_level_cluster)  
    image_gpu = cl_array.to_device(queue,image)
    image_gpu_buffer = cl_array.empty_like(image_gpu)

    #Create the neghbor graph
    
    pointwise_neghbors = cl_array.empty(queue,lowest_level_cluster.shape+(4,),
                                                      dtype=cl_array.vec.int2)
    
    neghbor_preamble = preamble+gpu_algorithms.add_matrix_axses_for('lables',lowest_level_cluster)
    neghbor_preamble = opencl_tools.build_preamble_for_context(ctx,neghbor_preamble)
      
    prg = cl.Program(queue.context,neghbor_preamble + _neghbors_kernal).build(options=[]);
    prg.list_neghbors(queue,image.shape,None,
                      lowest_level_cluster.data,pointwise_neghbors.data)
    
    all_neghbors = np.unique(pointwise_neghbors.get(queue))
    while(all_neghbors[0]['x'] < 0 or all_neghbors[0]['y']  < 0):
        all_neghbors = all_neghbors[1:]

    pointwise_neghbors.data.release();
    pointwise_neghbors = cl_array.to_device(queue,all_neghbors)
    
    live_pointwise_neghbors= pointwise_neghbors.shape[0]

    center_calulator = CenterCalulator(image_gpu.shape,
                                           total_clusters,['location','square'],queue)
    center_prams =  center_calulator.reduction_vector_size                                     

    #Create a system to calulate centers 
    center_locations = cl_array.empty(queue,(total_clusters,center_prams),dtype=np.float32)
    center_count = cl_array.empty(queue,(total_clusters,),dtype=np.int32)
    
    

    #Create the blure kernal
    range_vales = np.arange(-(3*sigma),3*sigma + 1);
    filter_vals = np.exp(-range_vales**2/(2*sigma**2));
    
    row_filter = filter_vals.reshape(-1,1);
    col_filter = filter_vals.reshape(1,-1);
    blure_filter = gpu_image_filter.SeparableImageFilter(image.shape,row_filter,
                                                         col_filter,queue=queue)
    


    #Create a kernal to calulate the combination value
    error_values_kernal = elementwise.ElementwiseKernel(ctx,
                        arguments="""__global int2* neghtbors,
                                     __global float* center_sum,
                                     __global int* center_count,
                                     __global float* error_value,
                                     float m_S_2_""",
                         operation= """
                                         error_value[i] = 
                                             combine_error(center_sum+neghtbors[i].x*NUMBER_OF_DIMS,
                                                     center_count[neghtbors[i].x],
                                                     center_sum+neghtbors[i].y*NUMBER_OF_DIMS,
                                                     center_count[neghtbors[i].y],
                                                     m_S_2_);
                                    """,
                         name="merge_error_vale",
                            preamble = preamble + 
                                center_calulator.reduction_vector_prams + 
                                               _combination_calulator)
    error_values = cl_array.empty(queue,pointwise_neghbors.shape[0],dtype=np.float32);

    #Create a scan to find clusters to merge
    #We reduce a int2 value which contians two arrays. The x keeps track of all 
    #Pairs that should be combined. The y keeps track of all the pairs that 
    #will be keepted. 
    place_results = cl_scan.GenericScanKernel(
                        ctx, np.int32,
                        arguments="""__global int2* neghtbors,
                                     __global float* error_valuss,
                                     __global float cut_off,
                                     __global int2* merged_negbors,
                                     __global int2* keep_negbors,
                                     __global int* final_count""",
                        input_expr="""(error_valuss[i] <= cut_off ? 1 : 0)""",
                        scan_expr="a+b", neutral="0",
                        output_statement="""int change = item-prev_item;
                        
                                            if(change > 0 )
                                                merged_negbors[prev_item] = neghtbors[i];
                                            else
                                                keep_negbors[i-prev_item] = neghtbors[i];

                                            if(i == 0)
                                                *final_count = last_item;
                                         """,
                        preamble = preamble )
               
    #Create a kernal that removes identical loops
    remove_ident = cl_scan.GenericScanKernel(
                        ctx, np.int32,
                        arguments="""__global int2* neghtbors,
                                     __global int2* keep_negbors,
                                     __global int* final_count""",
                        input_expr="""neghtbors[i].x != neghtbors[i].y ? 1 : 0""",
                        scan_expr="a+b", neutral="0",
                        output_statement="""int change = item-prev_item;
                                            if(change > 0 )
                                                keep_negbors[prev_item] = neghtbors[i];
                                            (*final_count) = last_item;""",
                        preamble = preamble )                        
                                
    merge_list = cl_array.empty_like(pointwise_neghbors)
    keep_negbors = cl_array.empty_like(pointwise_neghbors)
    new_size_buffer = cl_array.empty(queue,(1,),np.int32)
       
    #Create a kernal to do the actual replacement
    replacement = elementwise.ElementwiseKernel(ctx,
                         arguments="""__global int* cluster_ids,
                                       __global int* cluster_rename_table
                                    """,
                         operation= """
                                         cluster_ids[i] = cluster_rename_table[cluster_ids[i]];
                                    """,
                         name="id_replacement_kernal")
                         
    #and to renaim the neghbors
    replacement_negbors = elementwise.ElementwiseKernel(ctx,
                         arguments="""__global int2* saved_negbors,
                                       __global int* cluster_rename_table
                                    """,
                         operation= """
                                      saved_negbors[i].x = cluster_rename_table[saved_negbors[i].x];
                                      saved_negbors[i].y = cluster_rename_table[saved_negbors[i].y];
                                    """,
                         name="neghbor_replacement_kernal")                         




    #Create a kernal to make a replacement table
    update_rename_table = elementwise.ElementwiseKernel(ctx,
                         arguments=""" __global int* cluster_rename_table,
                                       __global int* cluster_rename_table_out,
                                       __global int2* merged_negbors,
                                       __global int* merge_indicator,
                                       __global int* finsh_buffer,
                                       int number_of_neghbors
                                    """,
                         operation= """
                                      cluster_rename_table_out[i] = cluster_rename_table[i];
                                      for(int j=0;j<number_of_neghbors;j++)
                                      {
                                         int replace_value = min(cluster_rename_table[merged_negbors[j].x],
                                                                 cluster_rename_table[merged_negbors[j].y]);
                                                                 
                                         int old_value     = max(cluster_rename_table[merged_negbors[j].x],
                                                                 cluster_rename_table[merged_negbors[j].y]);   
                                                                 
                                         if(cluster_rename_table[i] == old_value &&
                                                             replace_value != old_value)
                                         {
                                              cluster_rename_table_out[i] = replace_value;
                                              merge_indicator[i] = 1;
                                              (*finsh_buffer) = 0;
                                         }
                                      }
                                    """,
                         name="replacement_kernal")
                         
    finsh_buffer = cl_array.empty(queue,(1,),np.int32);
    
    #Create a kernal to count aktive clusters
    active_clusters_count = cl_reduction.ReductionKernel(
                                ctx, np.int32,neutral=0,
                                arguments="""__global int* cluster_rename_table """,
                                map_expr="""cluster_rename_table[i]== i ? 1 : 0""",
                                reduce_expr="a+b", 
                                preamble = preamble )
                      
                         
    cluster_rename_table = cl_array.arange(queue,total_clusters,dtype=np.int32)
    cluster_rename_table_out = cl_array.zeros_like(cluster_rename_table)
    merge_indicator = cl_array.empty(queue,total_clusters,dtype=np.int32)

    center_calulator(image_gpu,lowest_level_cluster,center_locations,center_count)

    
    #blure until we reach the starting scale
    #S_current = 0;
   # while S_current < S_org:
    #    print S_current
    #    blure_filter(image_gpu,image_gpu_buffer)
    #    image_gpu, image_gpu_buffer = image_gpu_buffer,image_gpu
        
     #   S_current += sigma;        
    
    S_current = S_org
    
#    ###############Debug
#    def make_indicator_image():
#        sp = lowest_level_cluster.get(queue)
#        im_res = center_locations.get(queue)[sp.ravel(),:];
#        im_out = np.zeros_like(image).reshape((-1,3))
#        var_out = np.zeros_like(image).reshape((-1,3))
#        count = center_count.get(queue)[sp.ravel()]
#        
#        #im_res[:,0] /= count;
#        #im_res[:,1] /= count;
#        #im_res[:,2] /= count;
#        
#        im_out[:,0] = im_res[:,0]/count;
#        im_out[:,1] = im_res[:,1]/count;
#        im_out[:,2] = im_res[:,2]/count;   
#        
#        var_out[:,0] = np.sqrt(count*im_res[:,3]-im_res[:,0]*im_res[:,0])/count;
#        var_out[:,1] = np.sqrt(count*im_res[:,4]-im_res[:,1]*im_res[:,1])/count;
#        var_out[:,2] = np.sqrt(count*im_res[:,5]-im_res[:,2]*im_res[:,2])/count;
#        var_out = np.sum(var_out,axis=1)
#        return color.lab2rgb(im_out.reshape(image.shape).astype(np.float64)), var_out.reshape(image.shape[:2]);
#    #plt.subplot(2,2,0)
#    debug_figs =[]
#    debug_figs.append(plt.figure())
#    img_blure = plt.imshow(color.lab2rgb(image_gpu.get(queue).astype(np.float64)))
#    #plt.subplot(2,2,1)
#    deb_dis_im,deb_dis_var = make_indicator_image()
#    debug_figs.append(plt.figure())
#    img_indi = plt.imshow(deb_dis_im,interpolation="none")
#    debug_figs.append(plt.figure())
#    var_indi = plt.imshow(deb_dis_var,interpolation="none")
#    #plt.subplot(2,2,2)
#    import skimage.segmentation
#    debug_figs.append(plt.figure())
#    img_line = plt.imshow(skimage.segmentation.mark_boundaries(
#                                    color.lab2rgb(image.astype(np.float64)),
#                                    lowest_level_cluster.get(queue)),
#                                                    interpolation="none")    
#    plt.pause(15)
#    ################  
    
    #Create an table of super pixals and add the inital ones
    pixal_table = {};
    error_list = []
    merge_scales=[]
    
    org_sp =  [np.array([x]) for x in xrange(total_clusters) ]
    pixal_table[S_org] = org_sp  
    
    #image_table = {}
    #image_table[S_org] = image
    
    living_clusters = total_clusters
    while(S_current < highest_scale and living_clusters > 1):

        blure_filter(image_gpu,image_gpu_buffer)
        image_gpu, image_gpu_buffer = image_gpu_buffer,image_gpu
        
        S_current += sigma;
        target_cluster_count = int((image.shape[0]*image.shape[1])/(S_current*S_current))

        to_remove = living_clusters - target_cluster_count;
        print living_clusters,S_current, live_pointwise_neghbors, to_remove
        if to_remove > 0 :
            #to_remove -= 1;
            center_calulator(image_gpu,lowest_level_cluster,center_locations,center_count)
    
            error_values_kernal(pointwise_neghbors,center_locations,
                                center_count,error_values,m/(S_current*S_current))
            
            error_values_cpu = error_values.get(queue)[:live_pointwise_neghbors]
            error_list.append(np.sum(error_values_cpu))
            
            smallest = np.max(bn.partsort(error_values_cpu,to_remove)[:to_remove])#
            #smallest = cl_array.min(error_values[:live_pointwise_neghbors])

            
            merge_scales.append(S_current)

            place_results(pointwise_neghbors,error_values,smallest,
                          merge_list,keep_negbors,new_size_buffer,
                          size=live_pointwise_neghbors, queue=queue)
            print live_pointwise_neghbors
            
            remove_neghbors = int(new_size_buffer.get(queue))
            live_pointwise_neghbors = int(live_pointwise_neghbors-remove_neghbors)
            

#            ###############Debug
#            plt.figure(15)
#            plt.clf()
#            locations = center_locations.get(queue)[:,6:8];
#            locations[:,0]/=center_count.get(queue);
#            locations[:,1]/=center_count.get(queue);
#            cmap = plt.get_cmap();
#            #ourder_values = np.argsort(error_values_cpu)
#            max_val = np.max(error_values_cpu)
#            print error_values_cpu/max_val, max_val
#            plt.imshow(deb_dis_im,interpolation="none")
#            for vals in pointwise_neghbors.get()[:live_pointwise_neghbors]:
#                colorTY = cmap(error_values_cpu[vals]/float(max_val))
#                plt.plot(locations[[vals['x'],vals['y']],1],
#                             locations[[vals['x'],vals['y']],0],
#                                color=colorTY)
#                                                        
#            plt.pause(4)
#            ###############        
    
            merge_indicator.fill(0,queue)
            while True:
                finsh_buffer.fill(1,queue)
                update_rename_table(cluster_rename_table,
                                        cluster_rename_table_out,
                                            merge_list,merge_indicator,finsh_buffer,
                                                remove_neghbors,queue=queue)
                                                                
                cluster_rename_table,cluster_rename_table_out =  \
                                cluster_rename_table_out,cluster_rename_table
                
                if(finsh_buffer.get(queue)[0] == 1):
                    break;
    
            merge_indicator_cpu = merge_indicator.get(queue).astype(np.bool)
            cluster_rename_table_cpu = cluster_rename_table.get(queue)

            #print np.sum(cluster_rename_table_cpu[keep_negbors.get(queue)['x']][:live_pointwise_neghbors] != cluster_rename_table_cpu[keep_negbors.get(queue)['y']][:live_pointwise_neghbors])      

            replacement(lowest_level_cluster,cluster_rename_table,queue=queue)
            replacement_negbors(keep_negbors,cluster_rename_table,
                                    range=slice(live_pointwise_neghbors),queue=queue)
                       
            #Remove loops, and merge back tot he stanered bufer      
            remove_ident(keep_negbors,pointwise_neghbors,new_size_buffer,
                          size=live_pointwise_neghbors, queue=queue)
                          
            cluster_bases_to_use = np.unique(cluster_rename_table_cpu[merge_indicator_cpu]  )      
                          
            new_sp =  [np.nonzero(cluster_rename_table_cpu == base)[0] 
                            for base in cluster_bases_to_use
                              if np.sum(cluster_rename_table_cpu == base) > 0 ]
            
            if( len(new_sp) > 0 ):
                pixal_table[S_current] = new_sp  
                #image_table[S_current] = image_gpu.get(queue)
                
            live_pointwise_neghbors = new_size_buffer.get(queue)[0]

            living_clusters = active_clusters_count(cluster_rename_table,queue=queue).get(queue)

#            ##############Debug
#            img_blure.set_data(color.lab2rgb(image_gpu.get(queue).astype(np.float64)))
#            deb_dis_im,deb_dis_var = make_indicator_image()
#            img_indi.set_data(deb_dis_im)
#            var_indi.set_data(deb_dis_var)
#            img_line.set_data(
#                skimage.segmentation.mark_boundaries(
#                                    color.lab2rgb(image.astype(np.float64)),
#                                    lowest_level_cluster.get(queue)))    
#            map(lambda x: x.canvas.draw(), debug_figs)
#            plt.pause(.1)
#            print live_pointwise_neghbors, live_pointwise_neghbors+remove_neghbors
#            print(merge_list.get(queue)[:remove_neghbors])
#            ###########
    finsh_buffer.data.release()
    
    merge_indicator.data.release()
    cluster_rename_table_out.data.release()
    cluster_rename_table.data.release()
    
    keep_negbors.data.release()
    merge_list.data.release()
    pointwise_neghbors.data.release()
    
    error_values.data.release()
    
    center_calulator.release()
    center_count.data.release()
    center_locations.data.release()
        
    im_cpu = image_gpu.get(queue)
    image_gpu.data.release()
    image_gpu_buffer.data.release()
    blure_filter.release()
    lowest_level_cluster.data.release();
    
    #import matplotlib.pyplot as plt
    #plt.plot(merge_scales[:-1],np.diff(np.log(error_list))/np.diff(merge_scales),'-*')
    #plt.figure()
    #plt.plot(merge_scales,np.log(error_list),'-*r')
    #plt.show(block=True)
    
    
    return pixal_table#, image_table
    



def pixal_table_to_tree(pixal_table,number_of_base_sp):
    replacement_table = np.array([None]*number_of_base_sp,
                                     dtype=region_map.HierarchicalRegion)

    scales_sorted =  sorted(pixal_table.keys())
    

    for scale in scales_sorted:
        for new_sp in pixal_table[scale]:
            children = np.unique(replacement_table[new_sp])
            
            new_parrent = region_map.HierarchicalRegion(new_sp,scale)
            
            for child in children:
                if child is not None:
                    new_parrent.children.append(child)
            
            replacement_table[new_sp] = new_parrent

        
    return list(np.unique(replacement_table))
    
def HierarchicalSLIC(image,base_tile_size,max_tile_size=None,m_base=20,
                     m_scale=0,sigma=3,total_runs = 1,max_itters=1000,
                                                     min_cluster_size=32):
    """
        A Hierarchical SLIC mapping
    """
                                                       
    base_slic = SLIC_gpu.SLIC(image,base_tile_size,m=m_base,
                                          total_runs = total_runs,
                                          max_itters=max_itters,
                                          min_cluster_size=min_cluster_size)
    
            
    
    multiscale_table = calulate_multiscale(image,
                                           base_slic.get_indicator_array(),
                                           max_tile_size,m_scale,sigma)
                                    
    
    root_table = pixal_table_to_tree(multiscale_table,len(base_slic))
    
    return region_map.HierarchicalRegionMap(base_slic,root_table)
    

        
if __name__ == '__main__':
        #From http://stackoverflow.com/questions/3579568/choosing-a-file-in-python-simple-gui
    from Tkinter import Tk
    from tkFileDialog import askopenfilename 
    import matplotlib.pyplot as plt
    
    Tk().withdraw()
    file_name = askopenfilename(filetypes=[('image file','*.jpg *.png *.bmp *.tif')])

    if(file_name is None):
        import sys;
        sys.exit(0);
        
    from skimage import io, color
    image_rgb = io.imread(file_name)/255.0
    image = color.rgb2lab(image_rgb) 
    
    slic_calc = SLIC_gpu.SLIC_calulator(image)
    _,cluster_index_cpu,color_value,locations = \
                        slic_calc.calulate_SLIC(2000,50,min_cluster_size=40)

    pixal_table = calulate_multiscale(image,cluster_index_cpu,500,20,10)
   
    print sum( len(pixal_table[key]) for key in pixal_table ), locations.shape[0]

    replacement_table = np.arange(locations.shape[0],dtype=np.int32)
    
    def make_inital_replacement_table(list_of_nodes):
        for node in list_of_nodes:
            replacement_table[node.list_of_sp] = min(node.list_of_sp)    
    
    def make_replacement_table(list_of_nodes):
        for node in list_of_nodes:
                for cnode in node.children:
                    replacement_table[cnode.list_of_sp] = min(cnode.list_of_sp)
    
    color_value_rgb = color.lab2rgb(color_value.reshape((-1,1,3)).astype(np.float64))
    color_value_rgb = color_value_rgb.reshape((-1,3))
    def make_indicator_image():
        #return color_value_rgb[replacement_table[cluster_index_cpu],:]
        import skimage.segmentation
        return skimage.segmentation.mark_boundaries(image_rgb,replacement_table[cluster_index_cpu])
        
    node_table,toplevel = pixal_table_to_tree(pixal_table,locations.shape[0])
    make_inital_replacement_table(toplevel)
    
    keylist_sort = sorted([key for key in node_table],reverse=True)


                                                  

    
###Gui
    
    
######To screen   
    img_indi = plt.imshow(make_indicator_image(),interpolation="none")
    plt.title("base")
    plt.pause(5)
    
    
    
    for key in keylist_sort:
        make_replacement_table(node_table[key])
        
        print len(node_table[key]),np.unique(replacement_table).size
        img_indi.set_data(make_indicator_image())
        plt.title("%f" % key)
        plt.pause(1)

##To Video 
#    import matplotlib.animation as animation
#    
#    # Set up formatting for the movie files
#    Writer = animation.writers['ffmpeg']
#    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=18000)
#    
#    fig1= plt.figure()
#    
#    img_indi = plt.imshow(make_indicator_image(),interpolation="none")
#    plt.title("base")
#    plt.axis('off')
#    
#    
#    def show_key(key):
#        make_replacement_table(node_table[key])
#        
#        print key, len(node_table[key]),np.unique(replacement_table).size
#
#        img_indi.set_data(make_indicator_image())
#        plt.title("%f" % key)
#    
#    line_ani = animation.FuncAnimation(fig1, show_key,keylist_sort, interval=150)
#    line_ani.save('T:\\SLIC_multiscale.mp4', writer=writer)
        
    #plt.imshow(im);
    #plt.show()