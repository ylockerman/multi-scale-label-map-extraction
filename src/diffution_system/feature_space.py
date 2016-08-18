# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:12:41 2013

@author: Yitzchak David Lockerman
"""

from __future__ import print_function

import abc
import numpy as np
import numpy.ma as ma
import skimage.feature
import skimage.filters
import skimage.morphology
from scipy import signal

import joblib
import matplotlib.pyplot as plt


class FeatureSpace(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def prepare_image(self,imageinput):
        pass
    
    @abc.abstractproperty
    def feature_size(self,tile_map):
        pass
    
    @abc.abstractmethod
    def feature_from_tile(self,tile,out):
        pass    
    
    @abc.abstractmethod
    def get_name(self):
        pass;
    
    def create_image_discriptor(self,tile_map):
        fs = self.feature_size(tile_map)
        out = np.zeros( (len(tile_map), fs ) );
        
        all_call = \
            [ joblib.delayed(_call_feature_from_tile)(self,tile_map[key],
                                                     out[tile_map.index_from_key(key),:])
                                                for key in tile_map ]
                                                                               
        return np.vstack(joblib.Parallel(n_jobs=3, verbose=10)(all_call))
        #  for key in tile_map:  
        #      index =  tile_map.index_from_key(key)    
        #      tile = tile_map[key]
        #      self.feature_from_tile(tile,out[index,:])
        # return out
        
#since we can't pickle object functions, we use this to call 
#the feature_from_tile methoid during exicution, also
#since we can't pass out correctly, we use this.
def _call_feature_from_tile(obj,tile,out):
    obj.feature_from_tile(tile,out)
    return out
 
class ManyMomentFeatureSpace(FeatureSpace):
    """
        Create the simple six peace descriptor
    """
    
    def __init__(self,moment_count):
        self._moment_count = moment_count;
    
    def prepare_image(self,imageinput):
        return imageinput
    
    def feature_size(self,tile_map):
        first_key = next(tile_map.__iter__())
        return tile_map[first_key].shape[-1]*self._moment_count
    
    def feature_from_tile(self,tile,out):
        import math
        
        fs = tile.shape[-1]
        
        if ma.is_masked(tile):
            tile = tile.compressed()
            
        tile = tile.reshape(-1,fs)
        
        out[0:fs] = np.mean(tile,axis=0);
        
        #a acoumilator
        norm_f = tile - out[0:fs]
        norm_acum = norm_f.copy()
        
        for m in xrange(1,self._moment_count):
            fact = 1.0/math.factorial(m+1);
            #fact = 1.0/(m+1);
            norm_acum *= norm_f;
            meen_vale = np.mean(norm_acum,axis=0)
            out[m*fs:m*fs+fs] =\
                    fact*np.power(np.abs(meen_vale),1.0/(m+1))*np.sign(meen_vale)
        
    def get_name(self):
        return "%d-moments" % self._moment_count
        
    
class GaborFeatureSpace(FeatureSpace):
    """
        Create the simple six peace descriptor
    """
    
    def __init__(self,octave_count=2,
                      octave_jump=1.4,
                      lowest_centrail_freq=0.03589,
                      angle_sensitivity=0.698132,
                      debug = False):
        self._octave_count = octave_count;
        self._octave_jump = octave_jump
        self._lowest_centrail_freq = lowest_centrail_freq;  
        self._angle_sensitivity = angle_sensitivity
        self._debug = debug
        
        self._bands_per_octave = int(np.floor(np.pi/self._angle_sensitivity))
        
        
        self._filters = {} 
        self._indicators = {} 
        
              
            
        
    def prepare_image(self,imageinput):
        return imageinput
    
    def feature_size(self,tile_map):
        first_key = next(tile_map.__iter__())
        tile = tile_map[first_key];

        return self._octave_count*self._bands_per_octave*tile.shape[2]*2
    
    def gabor_dft(self,conv_size,filter_size,F0,a,b,theta):
        frq_range_x = np.fft.fftfreq(conv_size[1]) - F0*np.cos(theta)
        frq_range_y = np.fft.fftfreq(conv_size[0]) - F0*np.sin(theta)

        
        freq_r_p,freq_c_p = np.meshgrid(frq_range_x,frq_range_y)
        
        freq_r =  freq_r_p*np.cos(theta) + freq_c_p*np.sin(theta)
        freq_c = -freq_r_p*np.sin(theta) + freq_c_p*np.cos(theta)  
        
        x_0 = .5*filter_size[1]
        y_0 = .5*filter_size[0]        
                
        magnitude = 1/(a*b)*np.exp(-np.pi*( (freq_r)**2/a**2 + 
                                                    (freq_c)**2/b**2 ) )
        phase = -2*np.pi*(x_0*freq_r_p + y_0*freq_c_p )
        
        return magnitude*np.exp(1j*phase)
        
    
    def feature_from_tile(self,tile,out):
        fs = tile.shape[2]
        has_mask = ma.is_masked(tile)
        
        if has_mask:
            tile_filled = ma.filled(tile,fill_value=0)
            mask_indicator = np.ones(tile.shape[:2],dtype=tile.dtype)
            mask_indicator[tile.mask[:,:,0]] = 0
            #mask_indicator /= np.sum(mask_indicator)
        else:
            tile_filled = tile
            mask_indicator = np.ones(tile.shape[:2],dtype=tile.dtype)
        
        out = out.view();
        out.shape = (self._octave_count,self._bands_per_octave,fs,2)



        K_a = (np.exp2(self._octave_jump) - 1)/(np.exp2(self._octave_jump) + 1)
        R = (1+K_a)/(1-K_a)
        K_b = np.tan(.5*self._angle_sensitivity)
        
        #The gabor function in skimage dose not include the pi factor in the
        #exp. Thus, we remove it here.
        C = np.sqrt(np.log(2)/np.pi)
        for octave in xrange(self._octave_count):
            F0 = self._lowest_centrail_freq*np.power(R,octave)
            a = F0*K_a/C
            b = F0*K_b/C
            
            size_pram = min(a,b)
            filter_size_x = np.int32(2.0/size_pram)
            filter_size_y = np.int32(2.0/size_pram)    
        
            #Create an inidcator of locations that are fully convolved
            indicator = np.ones((filter_size_y/2,filter_size_x/2))
            indicator /= np.sum(indicator)
            indicator = skimage.morphology.erosion(mask_indicator,indicator)
        
            fft_shape = np.array(tile_filled.shape[:2]) + np.array([filter_size_y,filter_size_x]) - 1
            fft_tile = np.fft.fft2(tile_filled,fft_shape,axes=(0,1))

            ###################
            if self._debug:
                plt.figure()     
                
                plt.subplot2grid((2*self._bands_per_octave+1,5),(0,0))
                plt.imshow(indicator)
                
                plt.subplot2grid((2*self._bands_per_octave+1,5),(0,1))
                plt.imshow(tile_filled)
                
                for color in xrange(tile.shape[2]):
                    plt.subplot2grid((2*self._bands_per_octave+1,5),(0,2+color))
                    plt.imshow(tile_filled[:,:,color]) 
                    plt.colorbar()
            ###########################
            
            for band in xrange(self._bands_per_octave):
                theta = self._angle_sensitivity*band
                
                fft_g_new = self.gabor_dft(fft_shape,(filter_size_y,filter_size_x),F0,a,b,theta)
                

                con_rez = signal.signaltools._centered(np.fft.ifft2(fft_g_new[:,:,None]*fft_tile,axes=(0,1)),tile_filled.shape)
                con_rez[np.logical_not(indicator)] = 0


                out[octave,band,:,0] = np.sum(np.square(con_rez.real),axis=(0,1) ) 
                out[octave,band,:,1] = np.sum(np.square(con_rez.imag),axis=(0,1) )
                
                
                ####################
                if self._debug:
                    plt.subplot2grid((2*self._bands_per_octave+1,5),(2*band+1,0))
                    plt.imshow(np.fft.ifft2(fft_g_new)[:filter_size_y,:filter_size_x].real )
                    
                    plt.subplot2grid((2*self._bands_per_octave+1,5),(2*band+1,1))
                    plt.imshow(con_rez.real)
                    
                    for color in xrange(tile.shape[2]):
                        plt.subplot2grid((2*self._bands_per_octave+1,5),(2*band+1,2+color))
                        plt.imshow(con_rez.real[:,:,color]) 
                        plt.colorbar()
                        
                    
                    plt.subplot2grid((2*self._bands_per_octave+1,5),(2*band+2,0))
                    plt.imshow(np.fft.ifft2(fft_g_new)[:filter_size_y,:filter_size_x].imag )
                    
                    plt.subplot2grid((2*self._bands_per_octave+1,5),(2*band+2,1))
                    plt.imshow(con_rez.imag)
                    
                    for color in xrange(tile.shape[2]):
                        plt.subplot2grid((2*self._bands_per_octave+1,5),(2*band+2,2+color))
                        plt.imshow(con_rez.imag[:,:,color])
                        plt.colorbar()
                #################################
        ###########
        if self._debug:
            plt.show(block=True)

    def get_name(self):
        return "%d-moments" % self._moment_count
        
class QuadTreeFeatureSpace(FeatureSpace):
    
    def __init__(self,number_of_levels=3):
        self._number_of_levels = number_of_levels
        self._number_of_locations = (4**number_of_levels-1)/3
        
    def prepare_image(self,imageinput):
        return imageinput
    
    def feature_size(self,tile_map):
        first_key = next(tile_map.__iter__())
        return tile_map[first_key].shape[2]*self._number_of_locations
    
    def feature_from_tile(self,tile,out):
        fs = tile.shape[-1]

        out = out.view();
        out.shape = (self._number_of_locations,fs)
        
        def build_tree_vector(points_r,points_c,levels_left,local_out_array):
            
            tile_rs = tile[points_r,points_c].reshape( -1,fs);
            local_out_array[0,:] = ma.mean(tile_rs,axis=0)
            
                #plt.plot(points_r,points_c,'o')
            if levels_left > 1:
                remaining_out_array = local_out_array[1:,:]
                mean_r = np.mean(points_r);
                mean_c = np.mean(points_c)
                
                offset_size = remaining_out_array.shape[0]/4
        
                top = points_r < mean_r
                bottom = np.logical_not(top)
                left = points_c < mean_c
                right = np.logical_not(left)
                
                quadrents = [ (top,right),(top,left),(bottom,left),(bottom,right)  ]
                
                #Fill the solution for all 4 quadrents 
                for idx,quadrent in enumerate(quadrents):
                    q = np.logical_and(quadrent[0],quadrent[1])
                    q_out = remaining_out_array[ idx*offset_size : (idx+1)*offset_size, : ]
                    build_tree_vector(points_r[q],points_c[q],levels_left - 1,q_out)
                #renormilize 
                remaining_out_array *= .25
                
                
        if ma.is_masked(tile):
            points_r,points_c = np.nonzero(np.logical_not(tile.mask[:,:,0]))
        else:
            grid = np.mgrid[0:tile.shape[0],
                            0:tile.shape[1]]
            points_r = grid[0,:,:].ravel()
            points_c = grid[1,:,:].ravel()
          
        build_tree_vector(points_r,points_c,self._number_of_levels,out)
        #plt.show(block=True)
        
    def get_name(self):
        return "QuadTree Feature Space"  
        
        
class BinCountFeatureSpace(FeatureSpace):
    """
        A discriptior that counts the relitive frequency of diffrent bins in
        the image. The image must be integer
    """
    
    def __init__(self,bin_count):
        self.bin_count = bin_count;
    
    def prepare_image(self,imageinput):
        return imageinput.astype(np.int32)
    
    def feature_size(self,tile_map):
        return self.bin_count
    
    def feature_from_tile(self,tile,out):
        if ma.is_masked(tile):
            tile = tile.compressed()
            
        tile = tile.reshape(-1)
        
        out[:] = np.bincount(tile,minlength=self.bin_count)
        
        out/=np.sum(out)
        
    def get_name(self):
        return "%d-bin count" % self._moment_count