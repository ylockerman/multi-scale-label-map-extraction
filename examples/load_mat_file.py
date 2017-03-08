# -*- coding: utf-8 -*-
"""
Created on Mon Jul 08 18:38:55 2013

@author: Yitzchak David Lockerman
"""
from __future__ import print_function

#if __name__ == '__main__':
#    import matplotlib
#    import mplh5canvas
#    mplh5canvas.set_log_level("warning")
     # set the log level before using module
     # default is 'warning', it is done here
     # to illustrate the syntax
#    matplotlib.use('module://mplh5canvas.backend_h5canvas')


import multiprocessing 

import os
import site
site.addsitedir(os.getcwd());

import diffution_system.io
import scipy.io
import skimage.io

if __name__ == '__main__':    
    multiprocessing.freeze_support()
    
    #From http://stackoverflow.com/questions/3579568/choosing-a-file-in-python-simple-gui
    from Tkinter import Tk
    from tkFileDialog import askopenfilename
    
    Tk().withdraw()
    image_name = askopenfilename(filetypes=[('image file','*.jpg *.png *.bmp *.tif')])

    if(image_name is None):
        import sys;
        print("Uses quit without selecting a file");
        sys.exit(0);
                
    image = skimage.io.imread(image_name)/255.0
                             
                             
    file_name = askopenfilename(filetypes=[('mat file','*.mat')])

    if(file_name is None):
        import sys;
        print("Uses quit without selecting a file");
        sys.exit(0);
                
    
                     
    mat_file = scipy.io.loadmat(file_name,squeeze_me=True)
    
    HSLIC = diffution_system.io.load_region_map(mat_file,'HSLIC',image)
    
    HSLIC.show_gui(image)
    
    if 'texture_tree'  in mat_file:
        texture_tree = diffution_system.io.load_region_map(mat_file,'texture_tree',image)
        texture_tree.show_gui(image)