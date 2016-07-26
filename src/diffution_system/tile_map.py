# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:01:28 2013

@author: Yitzchak David Lockerman
"""

from __future__ import print_function

import abc
import collections 

#####An abstract class that handels tiles
class TileMap(collections.Mapping):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __getitem__(self,key):
        """
            Return a tile based on the key
        """
        pass;
    
    @abc.abstractmethod
    def __setitem__(self,key,value):
        """
            change the value of a tile based on a key
        """
        pass;
        
    @abc.abstractmethod
    def __len__(self):
        """
            Return the number of tiles
        """        
        pass;
    
    @abc.abstractmethod
    def __iter__(self):
        """
            itterate through all the keys
        """          
        while False:
            yield None    
            
    @abc.abstractmethod
    def get_key_from_image(self,image,location):
        """
            If the tile in the image is in the set, this will return the key,
            otherwise it will rase an exception
        """
        pass;
        
    @abc.abstractmethod
    def index_from_key(self,key):
        """
            Retrun a index from 0 to len for a key
        """
        pass;
    
    @abc.abstractmethod
    def key_from_index(self,index):
        """
            Return key number index
        """
        pass
        
    @abc.abstractproperty
    def tile_size(self):
        """
            Returns the size of the tiles
        """
        pass
        
    @abc.abstractproperty
    def copy_map_for_image(self,image):
        """
            Returns a new tile map with the same properyes for
            a diffrent image
        """
        pass
    
    @abc.abstractproperty
    def get_tile_location(self, key):
        """
            Returns the location of the tile within the image
        """
        pass
    
    def get_tiles(self,heat_map,number):
        """
            get the given number of best tiles, based on the heatmap
        """
        #Select the number largest tiles
        import bottleneck as bn
        tiles = bn.argpartsort(-heat_map.reshape(-1),number)[:number]
        
        #genrator for listing the tiles
        #http://stackoverflow.com/questions/5010489/building-ranges-using-python-set-builder-notation
        def list_tiles():
            for tile_idx in tiles:
                key = self.key_from_index(tile_idx)
                yield  key,self[key]
        
        #And return them
        return list(list_tiles())       

###A class that takes tiles from a single image
class SingleImageTileMap(TileMap):
    """
        This class is a map of tiles that are obtenind from evenly spaced 
        points in an image.
        
        The key is the row/coln for the tile in the image
    """
    def __init__(self,image,tile_size,jump_size = None):
        
        #If we don't have a jump size, use the tile size
        if jump_size is None:
            jump_size = tile_size
            
        #if we are a grayscale image, convert it a one chanal color
        if len(image.shape) == 2:
            image = image.reshape(image.shape+(1,))
        
        self._image = image
        self._tile_size = tile_size
        self._jump_size = jump_size
        
        from skimage.util import shape
        self._windows = \
             shape.view_as_windows(image,(tile_size,tile_size,image.shape[2]));

        self._r_tiles = int(self._windows.shape[0]/self._jump_size)
        self._c_tiles = int(self._windows.shape[1]/self._jump_size)
        
    def __getitem__(self,key):
        """
            Return a tile based on the key (which is the row/col)
        """
        r,c = key
        return self._windows[r*self._jump_size,c*self._jump_size,0,:,:,:]
        
    def __setitem__(self,key,value):
        """
            change the value of a tile based on a key (which is the row/col)
        """
        r,c = key
        self._windows[r*self._jump_size,c*self._jump_size,0,:,:,:] = value
        
    def __len__(self):
        """
            Return the number of tiles, which is deturmand by size
        """    
        return self._r_tiles*self._c_tiles
        
    def __iter__(self):
        """
            itterate through all the keys
        """          
        for r in xrange(self._r_tiles):
            for c in xrange(self._c_tiles):
                yield (r,c)
    
    def get_key_from_image(self,image,location):
        if image is self._image and \
            location[0] > 0 and location[0] < image.shape[0] and \
                    location[1] > 0 and location[1] < image.shape[0]:
                        
            return int(self._r_tiles*location[0]),int(self._c_tiles*location[1]) 
            
        else:
            raise KeyError("Invalid image point")
            

    def index_from_key(self,key):
        """
            Retrun a index from 0 to len for a key
        """
        r,c = key
        return r*self._c_tiles + c
    

    def key_from_index(self,index):
        """
            Return key number index
        """
        return index/self._c_tiles , index % self._c_tiles
        
    def tile_size(self):
        return self._tile_size 
        
    def copy_map_for_image(self,image):
        """
            Returns a new tile map with the same properyes for
            a diffrent image
        """
        return SingleImageTileMap(image,self._tile_size,self._jump_size)
        
    def get_tile_location(self, key):
        """
            Returns the location of the tile within the image
        """
        raise NotImplementedError("Not yet implemented")    
    
    def get_tile_grid(self):
        """
            returns the gird of tile counts, 
            this is NOT a abstract methoid of the parrent
        """        
        return self._r_tiles, self._c_tiles
        
        
