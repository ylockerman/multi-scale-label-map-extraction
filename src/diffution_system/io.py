# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 12:20:17 2017

@author: ylockerman
"""

import numpy as np

import scipy.io

import region_map

def output_RLE(image):
    out = [] 

    for line in xrange(image.shape[0]):
        place = 1
        run_lenght = 1
        value = image[line,0]
        
        while place < image.shape[1]:
            if value != image[line,place]:
                out.append( np.array( [run_lenght,value] ) )
                run_lenght = 1
                value = image[line,place]
            else:
                run_lenght += 1
                
            place += 1
        
        #output the final run
        out.append( np.array( [run_lenght,value] ) )
    out = np.vstack(out)
    return out
    

def input_RLE(image_shape,rle):
    out_image = np.empty(image_shape,np.int32)
    location = 0;
    
    for idx in xrange(rle.shape[0]):
    
        run_lenght,value = rle[idx,:]
        
        out_image.flat[location:(location+run_lenght)] = value
        location += run_lenght
        
    return out_image


def load_region_map(data,regionmapname='HSLIC',image=None):
    """
    Loads a region map from the mat file saved by multiscale_extraction.py 
    
    Parameters
    ----------
    data : str or dict
        The filename of the file to load the data file, or the dict returned 
        when loading the file. 
    regionmapname: str
    """
    if data is str:
        data = scipy.io.loadmat(data,squeeze_me=True)
        

    image_shape = tuple(data['image_shape'].ravel())
    
    if image is None:
        image = np.zeros(image_shape,np.float32)
    
    atomic_SLIC_raw = input_RLE(image_shape[:2],data['atomic_SLIC_rle'])
    

    if regionmapname == 'SLIC':
        return region_map.AtomicRegionMap.from_raw(image,atomic_SLIC_raw)
    elif regionmapname == 'texture_lists':
        return region_map.CompoundRegionMap.from_raw(image,
                                             (atomic_SLIC_raw,data[regionmapname]))        
    elif regionmapname == 'HSLIC' or regionmapname == 'texture_tree':
        return region_map.HierarchicalRegionMap.from_raw(image,
                                             (atomic_SLIC_raw,data[regionmapname]))
    else:
        raise Exception("Unknown region map name "+regionmapname)


    
    