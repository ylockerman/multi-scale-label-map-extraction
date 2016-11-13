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
Created on Tue Aug 23 17:23:10 2016

@author: Yitzchak David Lockerman
"""
import  abc
import collections 

import numpy as np
import numpy.ma as ma



class AbstractRegionMap(collections.Mapping):
    """
    An AbstractRegionMap is the base class for the different region maps.  
    """
    
    __metaclass__ = abc.ABCMeta
 
    @abc.abstractmethod  
    def __getitem__(self,key):
        """
            Return a tile based on the key
        """
        return None
        
    @abc.abstractmethod  
    def __setitem__(self,key,value):
        """
            Change the value of a tile based on a key
        """
        return None
        
    @abc.abstractmethod  
    def __len__(self):
        """
            Return the number of regions
        """    
        return None
        
    @abc.abstractmethod  
    def __iter__(self):
        """
            itterate through all the keys
        """          
        return None

    @abc.abstractmethod  
    def index_from_key(self,key):
        """
            Retrun a index from 0 to len for a key
        """
        return None
    
    @abc.abstractmethod  
    def key_from_index(self,index):
        """
            Return key number index
        """
        return None

    @abc.abstractmethod  
    def copy_map_for_image(self,image):
        """
            Returns a new tile map with the same properyes for a diffrent image
        """
        return None
        
    @abc.abstractmethod  
    def get_region_location(self, key):
        """
            Returns the location of the region within the image
        """
        return (None,None)
        

    @abc.abstractmethod  
    def get_key_set(self):
        """
            Returns the keyset
        """
        
        return None

    @abc.abstractmethod  
    def get_atomic_map(self):
        """
        Returns the atomic map that this map is based on.  
        """
        return None
        
    @abc.abstractmethod  
    def get_atomic_keys(self,key):
        """
        Returns the a list of the atomic keys that make up a  key of this map.
        
        This satisfies
        map.get_atomic_map()[map.get_atomic_keys(key)] == map[key]
        """
        return None  
        
    @abc.abstractmethod
    def get_raw_data(self):
        """
           Outputs the content of this map as a numpy array
        """
        return None
        
    def get_atomic_indexes(self,key):
        """
        Returns the a list of the atomic indexes that make up a  key of this map.
        """
        return None   

class AtomicRegionMap(AbstractRegionMap):
    """
    An AtomicRegionMap resents a region map where all regions are atomic, 
    that is cannot be divided into sub regions. It is guaranteed that each 
    pixel belongs to one and only one region. 
        
    Parameters
    ----------
    image: 2d or 3d array
        The image that the regions are built from. Can be a 2d array 
        (grayscale) or 3d array (color). 
    other: AtomicRegionMap
        Another AtomicRegionMap to copy regions from. 
    """

    def __init__(self,image,regions_or_other):
        #if we are a grayscale image, convert it a one chanal color
        if len(image.shape) == 2:
            image = image[:,:,None]

        #We can also use this as a "copy constructor", this happens when
        #tile_size is an istance of this class. Note that we pass the image
        #on its own
        #This is only ment to e called from within the class
        if(isinstance(regions_or_other,AtomicRegionMap)):
            other = regions_or_other
            assert image.shape[:2] == other._cluster_index.shape
            
            self._image = image
            
            self._cluster_index = other._cluster_index
            self._number_of_superpixels = other._number_of_superpixels
            
            self._key_set = other._key_set
        else:
            self._image = image
            
            #Since we don’t update when the index is changed, we lock it to 
            #prevent accidental error. 
            self._cluster_index = regions_or_other.copy();
            self._cluster_index.setflags(write=False)
            
            self._number_of_superpixels = np.max(self._cluster_index) + 1
            
            #Precompute all the lists
            table = np.zeros((3,)+self._cluster_index.shape,np.int32 )
            
            table[0,:,:], table[1,:,:] = np.meshgrid(
                                           np.arange(self._cluster_index.shape[0]),
                                           np.arange(self._cluster_index.shape[1]),
                                           indexing='ij')
            table[2,:,:] = self._cluster_index
            
            table=np.reshape(table,(3,-1))
            
            indexes = np.argsort(table[2,:])
            table_resort = np.array(table[:,indexes])
            

            key_set = []
            
            _,location_of_ellement = np.unique(table_resort[2,:],True)
            
            start = 0 #The fist starting point is in the begining
            for ii in xrange(self._number_of_superpixels):
                if( ii + 1 < self._number_of_superpixels):
                    end = location_of_ellement[ii+1]
                else:
                    end = table_resort.shape[1]
                
                key_set.append((table_resort[0,start:end],table_resort[1,start:end]))
                start = end #Set the next starting point
            
            self._key_set = key_set



    def __getitem__(self,key):
        """
            Return a tile based on the key (which is the row/col)
        """
                
        if isinstance(key,list):
            key = tuple(map( np.concatenate ,zip(*key)))
        
        (r_loc,c_loc) = key
        r_min = np.min(r_loc)
        r_max = np.max(r_loc)
        
        c_min = np.min(c_loc)
        c_max = np.max(c_loc)          
        
        key_offset = (r_loc-r_min,c_loc-c_min)
        
        masked_values = np.ones((r_max-r_min+1,
                                  c_max-c_min+1,
                                  self._image.shape[2]), dtype=np.bool)
        masked_values[key_offset] = False
        image_masked = ma.MaskedArray(
                        self._image[r_min:(r_max+1),c_min:(c_max+1),:],
                                        mask=masked_values,hard_mask=True,fill_value=0)
        return image_masked
        
    def __setitem__(self,key,value):
        """
            Change the value of a tile based on a key (which is the row/col)
        """
        if isinstance(key,list):
            key = tuple(map( np.concatenate ,zip(*key)))
            
        if(ma.is_masked(value)):
            (r_loc,c_loc) = key
            r_min = np.min(r_loc)
            c_min = np.min(c_loc)
            key_offset = (r_loc-r_min,c_loc-c_min)            
            
            self._image[key] = value[key_offset]
        elif np.isscalar(value):
            self._image[key] = value
        else:
            self._image[key] = np.reshape(value,(-1,self._image.shape[2]))
        
    def __len__(self):
        """
            Return the number of tiles, which is deturmand by size
        """    
        return self._number_of_superpixels
        
    def __iter__(self):
        """
            itterate through all the keys
        """          
        return self._key_set.__iter__()
    
    def get_key_from_location(self,location):
        if location[0] > 0 and location[0] < self._cluster_index.shape[0] and \
                    location[1] > 0 and location[1] < self._cluster_index.shape[1]:
                        
            return self.key_from_index(self._cluster_index[location])
            
        else:
            raise KeyError("Invalid image point")
            

    def index_from_key(self,key):
        """
            Retrun a index from 0 to len for a key
        """

        return self._cluster_index[key][0]
    

    def key_from_index(self,index):
        """
            Return key number index
        """
        return self._key_set[index]

    def copy_map_for_image(self,image):
        """
            Returns a new tile map with the same properyes for a diffrent image
        """
        return self.__class__(image,self)
        
    def get_region_location(self, key):
        """
            Returns the location of the tile within the image
        """
        if isinstance(key,list):
            key = tuple(map( np.concatenate ,zip(*key)))
            
        (r_loc,c_loc) = key
        r_min = np.min(r_loc)
        c_min = np.min(c_loc)
        
        return (r_min,c_min)
        
    def get_indicator_array(self):
        """
            Returns the indicator array. This output is readonly.
        """
        return self._cluster_index
    
    def get_key_set(self):
        """
            Returns the keyset
        """
        
        return self._key_set
        
    def get_atomic_map(self):
        """
        Returns the atomic map that this map is based on.  
        """
        return self
        
    def get_atomic_keys(self,key):
        """
        Returns the a list of the atomic keys that make up a  key of this map.
        
        This satisfies
        map.get_atomic_map()[map.get_atomic_keys(key)] == map[key]
        """
        if isinstance(key,list):
            return key
            
        return [key]    
        
        
    def get_atomic_indexes(self,key):
        """
        Returns the a list of the atomic indexes that make up a  key of this map.
        
        This satisfies
        map.get_atomic_indexes(key) == 
            [ map.get_atomic_map().index_from_key(k) 
                        for k in map.get_atomic_keys(key)  ]
        """
        if isinstance(key,list):
            return [self.index_from_key(k) for k in key]
            
        return [self.index_from_key(key)]  
                        
    def display_region(self,key):
        if isinstance(key,list):
            key = tuple(map( np.concatenate ,key))
        
        (r_loc,c_loc) = key
        
        r_min = np.min(r_loc)
        r_max = np.max(r_loc)
        
        c_min = np.min(c_loc)
        c_max = np.max(c_loc)  
        
        image_out = np.zeros((r_max-r_min+1,
                                  c_max-c_min+1,
                                      self._image.shape[2]),
                                        dtype = self._image.dtype )
        image_out[r_loc-r_min,c_loc-c_min] = self[key]
        
        return image_out
        
    def get_raw_data(self):
        """
           Outputs the content of this map as a numpy array
        """
        return np.copy(self.get_indicator_array())
        
class CompoundRegion(object):
    """
    A CompoundRegion represents a region built from combination of atomic 
    regions. 
        
    Parameters
    ----------
    atomic_regions: list
        A list of indixes representing the combination of atomic regions. 
    """
    
    def __init__(self,atomic_regions):
        self.atomic_regions = atomic_regions


class CompoundRegionMap(AbstractRegionMap):
    """
    A CompoundRegionMap consist of regions that are built by the union of 
    atomic regions. 
        
    Parameters
    ----------
    atomic_region_map_or_image: AtomicRegionMap or image
        If a new CompoundRegionMap is being created, atomic_region_map_or_image
        is the base regions that combine to form the regions in this set. 
        
    region_map_or_other: a list of CompoundRegions or CompoundRegionMap
        If we are creating a new CompoundRegionMap this should be a list of 
        CompoundRegion that compose it. 
        If this is copying a CompoundRegionMap, then this is the image that 
        would be used in the new map.
    """

    def __init__(self,atomic_region_map_or_image,region_map_or_other=None):
                   
        if(isinstance(region_map_or_other,CompoundRegionMap)):
            other = region_map_or_other
            image = atomic_region_map_or_image
            
            self._atomic_region_map = \
                other._atomic_region_map.copy_map_for_image(image)
            
            self._region_set = other._region_set
            self._index_lookup_table = other._index_lookup_table
        else:
            self._atomic_region_map = atomic_region_map_or_image

            self._region_set = region_map_or_other

            self._index_lookup_table = {}
            for idx,node in enumerate(self._region_set):
                    self._index_lookup_table[node] = idx

        
    def get_submap(self,keyset):
        root_set = set(keyset)
        new_map = self.__class__(self._atomic_region_map,root_set)
        return new_map
                
    def __getitem__(self,key):
        """
            Return a tile based on the key (which is the row/col)
        """
        
        base_keys = self.get_atomic_keys(key)
        return self._atomic_region_map[base_keys]
        
    def __setitem__(self,key,value):
        """
            change the value of a tile based on a key (which is the row/col)
        """      
        base_keys = self.get_atomic_keys(key)
        self._atomic_region_map[base_keys]  = value
        
    def __len__(self):
        """
            Return the number of regions
        """    
        return len(self._region_set)
   
    def __iter__(self):
        """
            itterate through all the keys
        """          
        return self._region_set.__iter__()
    
       
    def get_indexes_from_location(self,location):

        inital_key = self._atomic_region_map.get_key_from_location(location)
        inital_index = self._atomic_region_map.index_from_key(inital_key)

        all_indexes = [index 
                       for index in xrange(len(self._region_set))
                           if inital_index in self._region_set[index].atomic_regions ];
                               
        return all_indexes
            
 

    def index_from_key(self,key):
        """
            Retrun a index from 0 to len for a key
        """

        return self._index_lookup_table[key]
        
    

    def key_from_index(self,index):
        """
            Return key number index
        """
        return self._region_set[index]
        
    def get_region_location(self, key):
        """
            Returns the location of the tile within the image
        """
        return self._atomic_region_map.get_tile_location(self,self._node_to_base_key(key))
        

    def copy_map_for_image(self,image):
        """
            Returns a new tile map with the same properyes for
            a diffrent image
        """
        return self.__class__(image,self)
        
    def display_tile(self,key):
        return self._atomic_region_map.display_tile(self,self._node_to_base_key(key))
    
    def get_key_set(self):
        return self._region_set
        
    def get_atomic_map(self):
        """
        Returns the atomic map that this map is based on.  
        """
        return self._atomic_region_map
        
    def get_atomic_keys(self,key):
        """
        Returns the a list of the atomic keys that make up a  key of this map.
        
        This satisfies
        map.get_atomic_map()[map.get_atomic_keys(key)] == map[key]
        """
        if isinstance(key,list):
            base_keys =  [self._atomic_region_map.key_from_index(idx)
                            for singlekey in key
                                for idx in singlekey.atomic_regions
                         ]
        else:
            base_keys =  [self._atomic_region_map.key_from_index(idx)
                            for idx in key.atomic_regions 
                         ]
                            
        return base_keys
        
    def get_atomic_indexes(self,key):
        """
        Returns the a list of the atomic indexes that make up a  key of this map.
        
        This satisfies
        map.get_atomic_indexes(key) == 
            [ map.get_atomic_map().index_from_key(k) 
                        for k in map.get_atomic_keys(key)  ]
        """
        if isinstance(key,list):
            base_indexes =  sum( (singlekey.atomic_regions 
                                            for singlekey in key), [] )
        else:
            base_indexes = list(key.atomic_regions)
                            
        return base_indexes
        
    def get_raw_data(self):
        """
            Outputs the content of this map as a numpy array with the base
            superpixel of each pixel and compound regions. 
        """

        raw_data = [ { 'list_of_atomic_superpixels' :np.copy( node.atomic_regions ) } 
                         for node in self._region_set 
                   ]

        
        return (self._atomic_region_map.get_raw_data(),raw_data)
        
def build_compound_region(old_map,new_region_mapping):
    """
    Builds a compound map from mapping of regions in an existing map to a 
    new set of clusters. This new compound region will have the same base 
    region map as the existing one, but will have a new set of regions. 
    
    Parameters
    ----------
    old_map: AbstractRegionMap
        The existing map where regions will come from.  
    new_region_mapping: array
        A mapping from the existing regions to the new regions they will 
        belong.   
        
        If this is an array, then it should be a one dimensional integer array 
        with an element for each of the existing regions. The value at each 
        location is the new region. 
        
        If this is a list, then it should contain an entry for each new region. 
        Each of those entries should be a list of old regions described by an index or key.
    """
    base_map = old_map.get_atomic_map()
    
    if isinstance(new_region_mapping,np.ndarray):
        #http://stackoverflow.com/questions/6688223/python-list-multiplication-3-makes-3-lists-which-mirror-each-other-when
        new_nodes = [[] for _ in xrange(np.max(new_region_mapping)+1)]
        
        for idx in xrange(new_region_mapping.shape[0]):
            idx_key = old_map.key_from_index(idx)
            atomic_indexes = old_map.get_atomic_indexes(idx_key) 
            new_nodes[ new_region_mapping[idx] ] += atomic_indexes
        
        new_nodes = [CompoundRegion(list(rg)) for rg in new_nodes ]
    elif isinstance(new_region_mapping,list):
        def old_region_to_base_region(old_region_disc):
            if isinstance(old_region_disc,int):
                idx_key = old_map.key_from_index(old_region_disc)
            else:
                idx_key = old_region_disc
                
            return old_map.get_atomic_indexes(idx_key) 
            
        new_nodes = [CompoundRegion(
                        sum( (old_region_to_base_region(el) for el in rg),[]) ) 
                                                for rg in new_region_mapping ]
            
    else:
        raise Exception("Unknown mapping type: %s" %(type(new_region_mapping)))
    
    
    return CompoundRegionMap(base_map,new_nodes)

                    
class HierarchicalRegion(CompoundRegion):
    """
    A HierarchicalRegion is a CompoundRegion that can contain subregions.   
    
    Parameters
    ----------
    atomic_regions: list of int
        The indexes of the atomic regions we are part of
    scale: float
        The scale of this region
    """
    def __init__(self,atomic_regions,scale):
        super(HierarchicalRegion,self).__init__(atomic_regions)
        
        self.scale = scale

        self.children = []

            

class HierarchicalRegionMap(CompoundRegionMap):
    """
    A HierarchicalRegionMap is a forest of regions across different scales. 
    Regions on larger scales are the union of regions on smaller scales.  
        
    Parameters
    ----------
    atomic_region_map_or_image: AtomicRegionMap or image
        If a new HierarchicalRegionMap is being created, 
        atomic_region_map_or_image is the base regions that combine to form the
        regions in this set. 
        
    region_map_or_other: a list of CompoundRegions or CompoundRegionMap
        If we are creating a new CompoundRegionMap this should be a list of 
        HierarchicalRegion that form the root. 
        If this is copying a HierarchicalRegionMap, then this is the image 
        that will be used in the new map.
    """

    def __init__(self,atomic_region_map_or_image,region_tree_or_other=None):

        if(isinstance(region_tree_or_other,HierarchicalRegionMap)):
            other = region_tree_or_other
            super(HierarchicalRegionMap,self).__init__(atomic_region_map_or_image,
                                                          region_tree_or_other)
           

            self._root_table = other._root_table
            self._regions_at_scale = other._regions_at_scale
            self._scales_sorted = other._scales_sorted
        else:
            
            self._root_table = list(region_tree_or_other)
            
            #Go through all the regions and buld the a list and lookup table
            all_regions = []
            self._regions_at_scale = {}
            
            regions_to_look_at = list(self._root_table)
            
            while len(regions_to_look_at) > 0:
                top = regions_to_look_at.pop()
                all_regions.append(top)
                
                if not top.scale in self._regions_at_scale:
                    self._regions_at_scale[top.scale] = []
                
                self._regions_at_scale[top.scale].append(top)

                regions_to_look_at += top.children

            self._scales_sorted =  sorted(self._regions_at_scale.keys())
            super(HierarchicalRegionMap,self).__init__(atomic_region_map_or_image,all_regions)                     

    
    def get_scales(self):
        return self._scales_sorted
        
    def _get_nodes_less_then_scale(self,max_scale):
        list_to_process = list(self._root_table)
        out_list = []
        while len(list_to_process) > 0:
            node = list_to_process.pop()
            
            if node.scale < max_scale or len(node.children) == 0:
                out_list.append(node)
            else:
                list_to_process += node.children
        
        return out_list
        
    def get_single_scale_map(self,scale):
        
        new_map = CompoundRegionMap(self._atomic_region_map
                                      ,self._get_nodes_less_then_scale(scale) )
 
        return new_map
        
    def get_submap(self,keyset):
        root_set = set(keyset)       
        #We need to be sure that we only add the root nodes. Remove any other        
        
        #http://stackoverflow.com/questions/716477/join-list-of-lists-in-python
        #All regions will be incressing order
        list_to_process = []
        map(list_to_process.extend,[ key.children for key in keyset])

        while len(list_to_process) > 0:
            node = list_to_process.pop()
            if node in root_set:
                root_set.remove(node)

            list_to_process += node.children
        
        new_map = self.__class__(self._atomic_region_map,root_set)
         
        return new_map


    def show_gui(self,image=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        if image is None:
            image = self._atomic_region_map._image
        
        plt.figure()
        ax_image = plt.subplot2grid((7,2),(0,0),rowspan=6)
        im_plot = ax_image.imshow(image)
        
        ax_mean_image = plt.subplot2grid((7,2),(0,1),rowspan=6)
        im_mean_image = ax_mean_image.imshow(image)        
        
        ax_scale = plt.subplot2grid((7,2),(6,0),colspan=2)

        initial_scale = (np.max(self._scales_sorted) + np.min(self._scales_sorted))/2
        scale_slider = Slider(ax_scale, 'Scale', np.min(self._scales_sorted), np.max(self._scales_sorted), valinit=initial_scale )
        
        def set_scale(scale):
            selected_scale =min(self._scales_sorted, key=lambda x:abs(x-scale)) 
            tile_map = self.get_single_scale_map(selected_scale)
            
            indicator = np.zeros(image.shape[:2]+(1,),np.int32)
            indicator_map = tile_map.copy_map_for_image(indicator)
            
            mean_color = np.zeros(image.shape,np.float32)
            mean_color_map = tile_map.copy_map_for_image(mean_color)
            
            existing_color_map = tile_map.copy_map_for_image(image)
            
            for loc in xrange(len(tile_map)):
                key = tile_map.key_from_index(loc)
                indicator_map[key] = loc
                
                tile = existing_color_map[key]
                tile = ma.reshape(tile,(-1,image.shape[-1]))
                mean_color_map[key] = ma.mean(tile,axis=0)
                
            import skimage.segmentation
            seg = skimage.segmentation.mark_boundaries(image,indicator[:,:,0])            
            im_plot.set_data(seg)
            
            im_mean_image.set_data(mean_color)
            
        scale_slider.on_changed(set_scale)    
        set_scale(initial_scale)
        plt.show(block=True)
 
        
    def get_raw_data(self):
        """
            Outputs the content of this map as a numpy array with the base
            superpixel of each pixel and a tree with the superpixels at each 
            level. 
        """

        def recusive_build_tree(list_of_nodes):      
            return [
                     { 
                        'scale' : node.scale,
                        'list_of_atomic_superpixels' : np.copy(node.atomic_regions),
                        'children' : recusive_build_tree(node.children)
                     } 
                         for node in list_of_nodes 
                   ]

        
        return (self._atomic_region_map.get_raw_data(), 
                    recusive_build_tree(self._root_table))
        
def hierarchical_region_map_from_stack(stack_dict):
    """
    Converts a stack of region maps to a single hierarchical region map. The 
    input stack should be in the form of a dictionary from scale to the region 
    map at that scale. All the region maps must be based on the same atomic 
    region map.
    
    Parameters
    ----------
    stack_dict: dict from float to AbstractRegionMap
        The dictionary giving the map at each scale.  
    
    """
    all_scales = sorted(stack_dict.keys(),reverse=True)
    
    #First, create the top regions
    root_scale = all_scales[0]
    root_map = stack_dict[root_scale]
    root_regions = [HierarchicalRegion(
                        list(root_map.get_atomic_indexes(key))
                        ,root_scale) 
                        for key in  root_map ]
    
    atomic_map = root_map.get_atomic_map()
    
    #Create a lookup table that indicates which region each atomic region 
    #belongs to
    atomic_region_table = np.array([None]*len(atomic_map),
                                           dtype=HierarchicalRegion)                               
    for region in root_regions:
        atomic_region_table[root_map.get_atomic_indexes(region)] = region
        
    
    for scale in all_scales[1:]:
        map_at_scale = stack_dict[scale]
        
        #All scales should have the same atomic map
        assert map_at_scale.get_atomic_map() is atomic_map  
        
        #We need to find the intersection of all regions at the new scale and 
        #all the parent regions. Each nonempty intersection is a new region. 
        for region in map_at_scale:
            region_atomic_index = root_map.get_atomic_indexes(region)
            
            #Find all parents that interact with the region in the new scale. 
            parent_nodes = np.unique(atomic_region_table[region_atomic_index])
            
            #For each of those parents, create a region that intersects with us. 
            for parent in parent_nodes:
                new_base_indexes = np.intersect1d(parent.atomic_regions,
                                                      region_atomic_index)
                                                      
                new_region = HierarchicalRegion(list(new_base_indexes),scale)
                parent.children.append(new_region)
                
                atomic_region_table[new_base_indexes] = new_region
    
    return HierarchicalRegionMap(atomic_map,root_regions)