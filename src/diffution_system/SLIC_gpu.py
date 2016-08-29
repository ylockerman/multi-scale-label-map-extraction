# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:01:15 2014

@author: Yitzchak David Lockerman
"""
import collections 
import numpy as np
import numpy.ma as ma


import skimage.morphology
import skimage.segmentation

if __name__ == '__main__':
    import sys
    sys.path.append('.')

from gpu import opencl_tools
    
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
elementwise = opencl_tools.cl_elementwise
reduction = opencl_tools.cl_reduction
cl_scan = opencl_tools.cl_scan


import joblib
import operator
 
import region_map


_SLIC_kernal_code  = """
    //Must define NUMBER_COLORS, IMAGE_ROWS, IMAGE_COLS, 
    //recalculate_centers_blocksize, count_center_table_locations_blocksize

    #define color_at(row,col) (image_data + (row)*NUMBER_COLORS*IMAGE_COLS + (col)*NUMBER_COLORS )
    #define location_center_of(i)  (location_center + 2*(i))
    #define color_center_of(i)  (color_center + NUMBER_COLORS*(i))
    #define cluster_table_at(mat,row,col) (mat[(row)*IMAGE_COLS+(col)])

    
    float color_sqr_dist(__global float* v1,__global float* v2)
    {
        float dist_sqr = 0;
        for(int c=0;c<NUMBER_COLORS;c++){
            float dif = v1[c] - v2[c];
            dist_sqr += dif*dif;
        }
        
        return dist_sqr;
    }


    int index_of_center(__global float *center,float S)
    {
        int row_box = convert_int_rtn(center[0]/S);
        int col_box = convert_int_rtn(center[1]/S); 
        
        int total_cols = convert_int_rtp(IMAGE_COLS/S);
        int total_rows = convert_int_rtp(IMAGE_ROWS/S);
        
        //if(row_box < 0) row_box = 0;
        //if(row_box >= total_rows) row_box = total_rows-1;

        //if(col_box < 0) col_box = 0;
        //if(col_box >= total_cols) col_box = total_cols-1;        
        
        return row_box*total_cols + col_box;
    }
    

    __kernel void find_best_center(__global float* image_data, //The image data,(row major order)
                                   __global float* location_center, //the location center of each cluster (row major order)
                                   __global float* color_center, //the color center of each cluster (row major order)
                                   __global int* center_table_locations,
                                   __global int* center_table,
                                   int number_of_centers, //the number of center points
                                   float S, //the averige distance between clusters
                                   float msqr, //the distance to color constant
                                   __global int* cluster_index, //which center each image point belongs to (row major order - output)
                                   __global float* cluaster_distance) //How far we are from the center (row major order - output)

    {
    
            const int our_row = get_global_id(0);
            const int our_col = get_global_id(1);
                        
            __global float* our_color = color_at(our_row,our_col);
            
            
            int our_row_box = convert_int_rtn(our_row/S);
            int our_col_box = convert_int_rtn(our_col/S); 
            int total_rows = convert_int_rtp(IMAGE_ROWS/S);
            int total_cols = convert_int_rtp(IMAGE_COLS/S);
            
            
            int start_row_box = our_row_box > 0 ? our_row_box - 1 : 0;
            int start_col_box = our_col_box > 0 ? our_col_box - 1 : 0;
            
            int end_row_box = our_row_box < total_rows - 1 ? our_row_box + 1 : total_rows - 1;
            int end_col_box = our_col_box < total_cols - 1 ? our_col_box + 1 : total_cols - 1;            
            

        
            
            int best_index = -1;
            float best_dist_sqr = INFINITY;
            
            for(int row_box = start_row_box; row_box <= end_row_box; row_box++)
                for(int col_box = start_col_box; col_box <= end_col_box; col_box++)
                {
                    int box_id =  row_box*total_cols + col_box;
                    int start_center_list = box_id > 0 ? 
                                            center_table_locations[box_id-1] : 0;
                    int end_center_list = center_table_locations[box_id];     
                    
                    for(int cluster_id = start_center_list;
                                cluster_id<end_center_list;cluster_id++)
                    {
                        int cluster = center_table[cluster_id];
                        __global float* custer_location = location_center_of(cluster);
                        __global float* color_location = color_center_of(cluster);             
                        
                        float d1 = (custer_location[0]-our_row)/S;
                        float d2 = (custer_location[1]-our_col)/S;
                
                        float location_dist = d1*d1+d2*d2;
                
                        if(-2 <d1 && d1 < 2 && -2 < d2 && d2 < 2)
                        {
                            float dist_sqr = location_dist*msqr + color_sqr_dist(color_location,our_color);
                                                
                            if(dist_sqr < best_dist_sqr ){
                                best_index = cluster;
                                best_dist_sqr = dist_sqr;
                            }
                        }
                    }
                }
                
            cluster_table_at(cluster_index,our_row,our_col) = best_index;
            if(best_index == -1)
                cluster_table_at(cluaster_distance,our_row,our_col) = 0;
            else
                cluster_table_at(cluaster_distance,our_row,our_col) =sqrt(best_dist_sqr);
    }
        
        
    __kernel void recalculate_centers(__global float* image_data,//The image data,(row major order)
                                      __global int* cluster_index,//which center each image point belongs to (row major order)
                                        float S, //the averige distance between clusters
                                        float m, //the distance to color constant
                                        int number_of_clusters, //The true number of clusters
                                      __global float* location_center, //the location center of each cluster (row major order - in,out)
                                      __global float* color_center, //the color center of each cluster (row major order - out),
                                      __local float* color_reduction
                                      )
    {
        int our_offset = get_local_id(0);
        int our_index = get_global_id(1);
        
        float color_workspace[NUMBER_COLORS]; 
        __local float3 center_reduction[recalculate_centers_blocksize];
  
        __global float* old_custer_location = location_center_of(our_index);
        
        int start_row = convert_int_rtn(old_custer_location[0]-2*S); //round down
        int end_row = convert_int_rtp(old_custer_location[0]+2*S); //round up
        int start_col= convert_int_rtn(old_custer_location[1]-2*S); //round down
        int end_col = convert_int_rtp(old_custer_location[1]+2*S); //round up
        
        if(start_row < 0) start_row = 0; 
        if(end_row >= IMAGE_ROWS) end_row = IMAGE_ROWS-1;

        if(start_col < 0) start_col = 0; 
        if(end_col >= IMAGE_COLS) end_col = IMAGE_COLS-1;

        int found_count = 0;        
        float center_row = 0;
        float center_col = 0;
        
        for(int c=0;c<NUMBER_COLORS;c++){
            color_workspace[c] = 0;
        }
        
        
        for(int col=start_col+our_offset;col<=end_col;col+=get_local_size(0))
        {
            for(int row=start_row;row<=end_row;row++)
            {
                if( cluster_table_at(cluster_index,row,col) == our_index)
                {
                    found_count++;
                    center_row += row;
                    center_col += col;
                    
                    __global float* point_color = color_at(row,col);
                    for(int c=0;c<NUMBER_COLORS;c++){
                        color_workspace[c] += point_color[c];
                    }
                }
            }
        }
        
        
        center_reduction[our_offset].s0 = found_count;        
        center_reduction[our_offset].s1 = center_row;
        center_reduction[our_offset].s2 = center_col;
        
        for(int c=0;c<NUMBER_COLORS;c++){
            color_reduction[our_offset*NUMBER_COLORS+c] = color_workspace[c];
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
        
        //We need to do a coustom reduction, as we need to sum collor
        if (recalculate_centers_blocksize >= 1024) 
        { 
            if (our_offset < 512) 
            { 
                center_reduction[our_offset] += center_reduction[our_offset + 512]; 
                for(int c=0;c<NUMBER_COLORS;c++)
                    color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                        color_reduction[(our_offset+ 512)*NUMBER_COLORS+c];
            } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if (recalculate_centers_blocksize >= 512) 
        { 
            if (our_offset < 256) 
            { 
                center_reduction[our_offset] += center_reduction[our_offset + 256]; 
                for(int c=0;c<NUMBER_COLORS;c++)
                    color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                        color_reduction[(our_offset+ 256)*NUMBER_COLORS+c];
            } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if (recalculate_centers_blocksize >= 256) 
        { 
            if (our_offset < 128) 
            { 
                center_reduction[our_offset] += center_reduction[our_offset + 128]; 
                for(int c=0;c<NUMBER_COLORS;c++)
                    color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                        color_reduction[(our_offset+ 128)*NUMBER_COLORS+c];
            } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        if (recalculate_centers_blocksize >= 128) 
        { 
            if (our_offset <  64) 
            { 
                center_reduction[our_offset] += center_reduction[our_offset +  64]; 
                for(int c=0;c<NUMBER_COLORS;c++)
                    color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                        color_reduction[(our_offset+ 64)*NUMBER_COLORS+c];
            } 
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        
        #ifdef WARPSPEED
        if (our_offset < 32)
        #endif
        {
            if (recalculate_centers_blocksize >=  64) 
            { 
                if (our_offset < 32) 
                {
                    center_reduction[our_offset] += center_reduction[our_offset + 32];
                    for(int c=0;c<NUMBER_COLORS;c++)
                        color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                            color_reduction[(our_offset+ 32)*NUMBER_COLORS+c];
                } 
                NOWARPBLOCK  
            }
            if (recalculate_centers_blocksize >=  32) 
            { 
                if (our_offset < 16) 
                {
                    center_reduction[our_offset] += center_reduction[our_offset + 16];
                    for(int c=0;c<NUMBER_COLORS;c++)
                        color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                            color_reduction[(our_offset+ 16)*NUMBER_COLORS+c];
                } 
                    NOWARPBLOCK  
            }
            if (recalculate_centers_blocksize >=  16) 
            { 
                if (our_offset < 8)  
                {
                    center_reduction[our_offset] += center_reduction[our_offset +  8];
                    for(int c=0;c<NUMBER_COLORS;c++)
                        color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                            color_reduction[(our_offset+ 8)*NUMBER_COLORS+c];
                } 
                NOWARPBLOCK 
            }
            if (recalculate_centers_blocksize >=   8) 
            { 
                if (our_offset < 4)  
                {
                    center_reduction[our_offset] += center_reduction[our_offset +  4];
                    for(int c=0;c<NUMBER_COLORS;c++)
                        color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                            color_reduction[(our_offset+ 4)*NUMBER_COLORS+c];
                } 
                NOWARPBLOCK  
            }
            if (recalculate_centers_blocksize >=   4) 
            { 
                if (our_offset < 2)  
                {
                    center_reduction[our_offset] += center_reduction[our_offset +  2];
                    for(int c=0;c<NUMBER_COLORS;c++)
                        color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                            color_reduction[(our_offset+ 2)*NUMBER_COLORS+c];
                } 
                NOWARPBLOCK  
            }
            if (recalculate_centers_blocksize >=   2) 
            { 
                if (our_offset < 1)  
                {
                    center_reduction[our_offset] += center_reduction[our_offset +  1];
                    for(int c=0;c<NUMBER_COLORS;c++)
                        color_reduction[(our_offset)*NUMBER_COLORS+c] += 
                            color_reduction[(our_offset+ 1)*NUMBER_COLORS+c];
                } 
                NOWARPBLOCK  
            }
        } 
        
        if(our_offset == 0 )
        {
            found_count = center_reduction[our_offset].s0;        
            center_row = center_reduction[our_offset].s1;
            center_col = center_reduction[our_offset].s2;
            
            if(found_count > 1e-2)
            {
                __global float* cluster_location = location_center_of(our_index);
                cluster_location[0] = center_row/found_count;
                cluster_location[1] = center_col/found_count;
                
                __global float* cluster_color = color_center_of(our_index);
                for(int c=0;c<NUMBER_COLORS;c++){
                    cluster_color[c] = color_reduction[c]/found_count;
                }
            }
        }
        
    }
    
    __kernel void count_center_table_locations(__global float* location_center,
                                               float S,
                                               int number_of_clusters,
                                               __global int* center_table_locations)
    {
        int my_offset = get_local_id(0);
        int my_index = get_global_id(1); 
        
        
        __local int count_table[count_center_table_locations_blocksize];

        count_table[my_offset] = 0;
        
        for(int cluster = my_offset;
                cluster < number_of_clusters; 
                    cluster+=count_center_table_locations_blocksize)
        {
            int location = index_of_center(location_center+2*cluster,S);
            count_table[my_offset] += (location==my_index) ? 1 : 0;
        }
        
        #ifdef WARPSPEED
        if (count_center_table_locations_blocksize > 32)
        #endif        
            barrier(CLK_LOCAL_MEM_FENCE); 
"""+opencl_tools.get_inkernal_reduction('count_table',
                    'count_center_table_locations_blocksize','my_offset')+ """
                    
        if (my_offset == 0 && 
            my_index < (int)ceil(IMAGE_COLS/S)*(int)ceil(IMAGE_ROWS/S))
        {
                center_table_locations[my_index] = count_table[0];
        }
    }
    
    
    __kernel void create_center_table(__global float* location_center,
                                      __global int* center_table_locations,
                                      float S,
                                      int number_of_clusters,
                                      __global int* center_table
                                     )
    {
        int my_offset = get_local_id(0);
        int my_index = get_global_id(1); 
        
        
        
        __local int count_table[count_center_table_locations_blocksize];

        count_table[my_offset] = 0;
        
        for(int cluster = my_offset;
                cluster < number_of_clusters; 
                    cluster+=count_center_table_locations_blocksize)
        {
            int location = index_of_center(location_center+2*cluster,S);
            count_table[my_offset] += (location==my_index) ? 1 : 0;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE); 
        
        int local_start = my_index > 0 ? 
                            center_table_locations[my_index-1] : 0;
        for(int i=0;i<my_offset;i++)
            local_start += count_table[i];
        
        for(int cluster = my_offset;
                cluster < number_of_clusters; 
                    cluster+=count_center_table_locations_blocksize)
        {
            int location = index_of_center(location_center+2*cluster,S);
            if(location==my_index)
            {
                center_table[local_start] = cluster;
                local_start++;
            }
        }        
    }
    
    
    __kernel void remove_points(__global int* cluster_index_in,
                                __global int* locations_to_remove_in,
                                __global int* cluster_index_out,
                                __global int* locations_to_remove_out)
    {
            const int our_row = get_global_id(0);
            const int our_col = get_global_id(1); 
            
            
            int should_remove = cluster_table_at(locations_to_remove_in,our_row,our_col);      
            
            if(should_remove)
            {


                //if ther is a neghbor that has a differt value, use it
                if(our_row > 0 && 
                    cluster_table_at(locations_to_remove_in,our_row-1,our_col)==0 )
                {
                     cluster_table_at(cluster_index_out,our_row,our_col) = 
                             cluster_table_at(cluster_index_in,our_row-1,our_col);
                     cluster_table_at(locations_to_remove_out,our_row,our_col) = 0;
                }
                else if(our_row < IMAGE_ROWS - 1 && 
                    cluster_table_at(locations_to_remove_in,our_row+1,our_col)==0 )
                {
                     cluster_table_at(cluster_index_out,our_row,our_col) = 
                             cluster_table_at(cluster_index_in,our_row+1,our_col);  
                     cluster_table_at(locations_to_remove_out,our_row,our_col) = 0;
                }
                else if(our_col > 0 && 
                    cluster_table_at(locations_to_remove_in,our_row,our_col-1)==0 )
                {
                     cluster_table_at(cluster_index_out,our_row,our_col) = 
                             cluster_table_at(cluster_index_in,our_row,our_col-1); 
                     cluster_table_at(locations_to_remove_out,our_row,our_col) = 0;
                }
                else if(our_col < IMAGE_COLS - 1 && 
                    cluster_table_at(locations_to_remove_in,our_row,our_col+1) ==0 )
                {
                     cluster_table_at(cluster_index_out,our_row,our_col) = 
                             cluster_table_at(cluster_index_in,our_row,our_col+1); 
                     cluster_table_at(locations_to_remove_out,our_row,our_col) = 0;
                }
                else //Otherwise, wait till next
                {
                     cluster_table_at(cluster_index_out,our_row,our_col) =
                         cluster_table_at(cluster_index_in,our_row,our_col); 
                     cluster_table_at(locations_to_remove_out,our_row,our_col) = 1;
                }
                
            }
            else
            {
                cluster_table_at(cluster_index_out,our_row,our_col) = 
                        cluster_table_at(cluster_index_in,our_row,our_col);
                            
                cluster_table_at(locations_to_remove_out,our_row,our_col) = 0;
            }
            
    }
    
    

"""

class SLIC_calulator:
    
    def __init__(self,image,queue=None):
        self._image = image

        if(queue is None):
            self._queue = cl.CommandQueue(opencl_tools.get_a_context(),properties=opencl_tools.profile_properties)
        else:
            self._queue =  queue
            
        self._blocksize = 128;
        
        
        
        preamble = """
            #define IMAGE_ROWS %d
            #define IMAGE_COLS %d
            #define NUMBER_COLORS %d
            #define recalculate_centers_blocksize %d
            #define count_center_table_locations_blocksize %d
        """ % (image.shape + (self._blocksize,self._blocksize))
        
        preamble = opencl_tools.build_preamble_for_context\
                                        (self._queue.context,preamble)
        
        prg = cl.Program(self._queue.context,preamble + _SLIC_kernal_code).build();
        
        self._find_best_center = prg.find_best_center
        self._find_best_center.set_scalar_arg_dtypes([
                                                None, #float* image_data, 
                                                None, #float* location_center,
                                                None, #float* color_center, 
                                                None, #__global int* center_table_locations,
                                                None, #__global int* center_table,
                                                np.int32, #int number_of_centers, 
                                                np.float32, #float S, 
                                                np.float32, #float m,
                                                None, #int* cluster_index,
                                                None, #float* cluaster_distance
                                            ])
                                

        self._recalculate_centers = prg.recalculate_centers
        self._recalculate_centers.set_scalar_arg_dtypes([
                                                None, #float* image_data,
                                                None, #int* cluster_index,
                                                np.float32, #float S, 
                                                np.float32, #float m, 
                                                np.int32, #int number_of_clusters,
                                                None, #float* location_center, 
                                                None, #float* color_center,
                                                None
                                            ]) 

        self._count_center_table_locations = prg.count_center_table_locations
        self._count_center_table_locations.set_scalar_arg_dtypes([
                                               None,#__global float* location_center,
                                               np.float32,#float S,
                                               np.int32,#int number_of_clusters,
                                               None,#__global int* center_table_locations
                                            ])  
        self._center_table_placement = cl_scan.InclusiveScanKernel(self._queue.context,
                                                                   np.int32,
                                                                   'a+b',
                                                                   neutral='0')
          

        self._create_center_table = prg.create_center_table
        self._create_center_table.set_scalar_arg_dtypes([
                                                None,#__global float* location_center,
                                                None,#__global int* center_table_locations
                                                np.float32,#float S,
                                                np.int32,#int number_of_clusters,
                                                None#__global float* center_table
                                            ])  
                                            
        self._remove_points = prg.remove_points
                                    
    def calulate_SLIC(self,number_of_clusters,m,max_itters=1000,min_cluster_size=32,max_tryes=1):
        S= int(np.sqrt((self._image.shape[0]*self._image.shape[1])/number_of_clusters))
        m = int(m)
        number_of_clusters =int(number_of_clusters)
        print "Finding tiles spaced by ",S," with a m of ",m
        
        float32_size = int(np.float32(0).nbytes)
        
        image_gpu = cl_array.to_device(self._queue,self._image.astype(np.float32))
        
        initial_locations = np.zeros((number_of_clusters,2),np.float32)
        initial_locations[:,0] = np.random.random_sample(number_of_clusters)*self._image.shape[0]
        initial_locations[:,1] = np.random.random_sample(number_of_clusters)*self._image.shape[1]
        
        cluster_locations = cl_array.to_device(self._queue,initial_locations)
        cluster_color_value = cl_array.zeros(self._queue,(number_of_clusters,self._image.shape[2]),np.float32)
        
        cluster_index = cl_array.zeros(self._queue,self._image.shape[:2],np.int32)
        cluster_distances = cl_array.zeros(self._queue,self._image.shape[:2],np.float32)

        number_of_table_rows = int(np.ceil(self._image.shape[0]/float(S)))
        number_of_table_cols = int(np.ceil(self._image.shape[1]/float(S)))
        total_table_size = number_of_table_rows*number_of_table_cols
                                            
        table_count = cl_array.zeros(self._queue,
                                         (number_of_table_rows,number_of_table_cols,),np.int32)
                                         
        location_table = cl_array.zeros(self._queue,(number_of_clusters,),np.int32)
        
#        plt.figure();
#        cluster_index_cpu =cluster_index.get(self._queue)     
#        im = plt.imshow(skimage.segmentation.mark_boundaries(image_rgb,cluster_index_cpu))
#        plt.gca().get_xaxis().set_visible(False)
#        plt.gca().get_yaxis().set_visible(False)    
#        plt.pause(.1);
        
        for try_up in xrange(max_tryes):
            last_error = np.inf
            recalculate_centers_local_size = (self._blocksize,1)
            recalculate_centers_global_size = ( self._blocksize,number_of_clusters)
            
            count_center_table_local_size = (self._blocksize,1)
            count_center_table_global_size = ( self._blocksize,total_table_size)

            
            for itter in xrange(max_itters):
                
                   
                self._count_center_table_locations(self._queue,count_center_table_global_size,
                                                   count_center_table_local_size,
                                                    cluster_locations.data,S,
                                                    number_of_clusters,
                                                    table_count.data)   
                                                    
                self._center_table_placement(table_count.ravel())
                
                self._create_center_table(self._queue,count_center_table_global_size,
                                          count_center_table_local_size,
                                          cluster_locations.data,
                                          table_count.data,S,
                                          number_of_clusters,
                                          location_table.data) 
                                      

                self._find_best_center(self._queue,self._image.shape[:2],None,
                                       image_gpu.data,cluster_locations.data,
                                       cluster_color_value.data,table_count.data,location_table.data,
                                       number_of_clusters,S,m*m,cluster_index.data,
                                       cluster_distances.data
                                       )
                                       
                self._recalculate_centers(self._queue,recalculate_centers_global_size,recalculate_centers_local_size,
                                          image_gpu.data,cluster_index.data,S,m,number_of_clusters,
                                          cluster_locations.data,cluster_color_value.data,
                                          cl.LocalMemory(float32_size*self._blocksize*self._image.shape[2]))
            
                
                new_error = cl_array.sum(cluster_distances,queue=self._queue).get(self._queue) 
                print itter,  new_error,  last_error- new_error
                if( new_error == last_error):
                    break;
 

#                cluster_index_cpu =cluster_index.get(self._queue)     
#                im.set_data(skimage.segmentation.mark_boundaries(image_rgb,cluster_index_cpu))
#                cluster_locations_cpu =cluster_locations.get(self._queue) 
#                plt.plot(cluster_locations_cpu[:,1],cluster_locations_cpu[:,0],'*')
#                plt.pause(2);
                
                last_error=new_error             
    #            if itter % 1 == 0:
                
                #plt.figure(1)
                #plt.clf()
                #plt.imshow(cluster_index.get(self._queue));
                #points = cluster_locations.get(self._queue)
                #plt.plot(points[:,1],points[:,0],'*')
                #from skimage import io, color
                #image_new = cluster_color_value.get(self._queue)[cluster_index.get(self._queue),:]
                #image_new_rgb = color.lab2rgb(image_new.astype(np.float64))
                #plt.imshow(image_new_rgb)
                #plt.pause(.1)

                #plt.figure(2)
                #plt.clf()
                #plt.imshow(cluster_distances.get(self._queue));
                #plt.plot(points[:,1],points[:,0],'*')
                #plt.colorbar()
                #plt.pause(.1)
                
            
            cluster_index_cpu =cluster_index.get(self._queue)        
            cluster_color_value_cpu =cluster_color_value.get(self._queue)    
            cluster_locations_cpu =cluster_locations.get(self._queue) 
            
           #limmet our selves to connected componetns
            cluster_index_cpu,\
                cluster_color_value_cpu,\
                    cluster_locations_cpu = \
                        relable_connected_clusters(cluster_index_cpu,
                                                   cluster_color_value_cpu,
                                                   cluster_locations_cpu);
                                 
            cluster_index.data.release()
            cluster_color_value.data.release()
            cluster_locations.data.release()  
            location_table.data.release() 
            #if we have another upcoming try, prepare for it
            if try_up < max_tryes - 1:
                small_clusters_indexes = np.nonzero(np.bincount(cluster_index_cpu.ravel()) < min_cluster_size)[0]
                indexes_to_keep = np.nonzero(np.bincount(cluster_index_cpu.ravel()) >= min_cluster_size)[0]
                
                to_remove_indicator = np.in1d(cluster_index_cpu.ravel(),small_clusters_indexes).astype(np.bool).reshape(cluster_index_cpu.shape)
                to_keep_indicator = np.in1d(cluster_index_cpu.ravel(),indexes_to_keep).astype(np.bool).reshape(cluster_index_cpu.shape) 
                
                
                new_to_old_lables,cluster_index_cpu_rename = \
                   np.unique(cluster_index_cpu[to_keep_indicator],
                                                      return_inverse = True)
                
                #Delete the bad cluster locations                
                cluster_index_cpu[to_remove_indicator] = -1;
                cluster_index_cpu[to_keep_indicator] = cluster_index_cpu_rename
                cluster_color_value_cpu = cluster_color_value_cpu[new_to_old_lables,:]
                cluster_locations_cpu = cluster_locations_cpu[new_to_old_lables,:]
                
                            
                #return to GPU for the next try
                number_of_clusters = new_to_old_lables.size
                cluster_index = cl_array.to_device(self._queue,cluster_index_cpu.astype(np.int32))
                cluster_color_value = cl_array.to_device(self._queue,cluster_color_value_cpu.astype(np.float32))
                cluster_locations =cl_array.to_device(self._queue,cluster_locations_cpu.astype(np.float32))
                location_table = cl_array.zeros(self._queue,(number_of_clusters,),np.int32)
                
        table_count.data.release()
        cluster_distances.data.release()
        image_gpu.data.release()
        
  
        
        #Remove small clusters:
        cluster_index_cpu = self.remove_small_clusters(cluster_index_cpu,
                                                               min_cluster_size)
         
            
                
        #limmet our selves to connected componetns
        cluster_index_cpu,\
            cluster_color_value_cpu,\
                cluster_locations_cpu = \
                    relable_connected_clusters(cluster_index_cpu,
                                               cluster_color_value_cpu,
                                               cluster_locations_cpu);        
        
        
        return new_error,cluster_index_cpu,cluster_color_value_cpu,cluster_locations_cpu
        
    #Remove small clusters
    def remove_small_clusters(self,cluster_index_cpu,min_cluster_size):
        good_values = cluster_index_cpu.ravel() >= 0
        small_clusters_indexes = np.nonzero(np.bincount(cluster_index_cpu.ravel()[good_values]) < min_cluster_size)[0]
        
        to_remove_indicator = np.in1d(cluster_index_cpu.ravel(),small_clusters_indexes).astype(np.int32)
        to_remove_indicator[~good_values] = 1
        to_remove_indicator = to_remove_indicator.reshape(cluster_index_cpu.shape)
        
        
        cluster_index_gpu = cl_array.to_device(self._queue,cluster_index_cpu.astype(np.int32))
        cluster_index_gpu_out = cl_array.zeros_like(cluster_index_gpu)
        
        to_remove_indicator_gpu = cl_array.to_device(self._queue,to_remove_indicator);
        to_remove_indicator_gpu_out = cl_array.zeros_like(to_remove_indicator_gpu)
        
        
        points_left = np.Inf
        

    
        while(points_left > 0 ):
            self._remove_points(self._queue,self._image.shape[:2],None,
                                   cluster_index_gpu.data,
                                   to_remove_indicator_gpu.data,
                                   cluster_index_gpu_out.data,
                                   to_remove_indicator_gpu_out.data,
                                   )            
            #Swap input and output
            cluster_index_gpu,cluster_index_gpu_out =\
                        cluster_index_gpu_out, cluster_index_gpu
            to_remove_indicator_gpu, to_remove_indicator_gpu_out =\
                        to_remove_indicator_gpu_out, to_remove_indicator_gpu
            
            
            
            points_left = cl_array.sum(to_remove_indicator_gpu,queue = self._queue).get(self._queue)
 
        
        cluster_index_cpu = cluster_index_gpu.get(self._queue)
        
        
        
        
        cluster_index_gpu.data.release()
        cluster_index_gpu_out.data.release()
        
        to_remove_indicator_gpu.data.release()
        to_remove_indicator_gpu_out.data.release()
        return cluster_index_cpu


def relable_connected_clusters(cluster_index_cpu,cluster_color_value_cpu,cluster_locations_cpu):
        new_cluster_index = skimage.morphology.label(cluster_index_cpu,8);
        new_lable_cont = np.max(new_cluster_index)+1
        

        #create a table of new to old labels
        new_cluster_values, new_cluster_sample_locations =\
                                np.unique(new_cluster_index, return_index=True)
                                
        
                                
                                
        #print isinstance(new_cluster_sample_locations, np.Array)
        lable_table = np.zeros((new_lable_cont),np.int64)
        lable_table[new_cluster_values] = cluster_index_cpu.ravel()[new_cluster_sample_locations]
        
        #and update our arrays
        new_cluster_color_value = cluster_color_value_cpu[lable_table]
    
        new_cluster_locations = cluster_locations_cpu[lable_table]
        
        return new_cluster_index,new_cluster_color_value,new_cluster_locations

def _do_single_calulation(image,number_of_clusters,m,max_itters,min_cluster_size):
    slic_calc = SLIC_calulator(image)
    return slic_calc.calulate_SLIC(number_of_clusters,m,max_itters,min_cluster_size)
        
        
def SLIC(image,tile_size,m=20,total_runs = 1, max_itters=1000,min_cluster_size=32):     
    target_number_of_clusters=  (image.shape[0]*image.shape[1])/tile_size**2
    print target_number_of_clusters
    
    all_trys = \
        [joblib.delayed(_do_single_calulation)(image,target_number_of_clusters,m,max_itters,min_cluster_size)
            for _ in xrange(total_runs)
        ]
    
    all_results = joblib.Parallel(n_jobs=1,verbose=200)(all_trys)
    
    _,cluster_index_cpu,_, _ = min(all_results,key=operator.itemgetter(0))
    
    return region_map.AtomicRegionMap(image,cluster_index_cpu)
        
        
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
    
    
    slic_calc = SLIC_calulator(image)
    new_error,\
        cluster_index_cpu,\
            cluster_color_value_cpu,\
                cluster_locations_cpu = \
                    slic_calc.calulate_SLIC(10000,20,min_cluster_size=20)
    print cluster_color_value_cpu.shape
    
    plt.imshow(cluster_index_cpu);
    
    image_new = cluster_color_value_cpu[cluster_index_cpu,:]
    image_new_rgb = color.lab2rgb(image_new.astype(np.float64))
    
    plt.figure();
    plt.imshow(image_new_rgb);
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    
    plt.figure();
    plt.imshow(image_new_rgb-image_rgb);
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    
    plt.imshow(skimage.segmentation.mark_boundaries(image_rgb,cluster_index_cpu))    
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    plt.show()    