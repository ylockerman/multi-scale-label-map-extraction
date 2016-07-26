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


Created on Sun May 17 16:08:04 2015

@author: Yitzchak David Lockerman
"""

import subprocess
import tempfile
import shutil
import os
import os.path
from skimage import io

import numpy as np
SLIC_excicutable = "SLICSuperpixelSegmentation.exe"

def get_SLIC_regions(image,m,count):
    try:
        print "Running SLIC with %d %d" %(m,count)
        #Create a temp directory 
        temp_dir = tempfile.mkdtemp()
        
        #save the image
        image_filename = os.path.join(temp_dir,'image.bmp')
        io.imsave(image_filename,image)
        
        #Call the SLIC code
        subprocess.check_call([SLIC_excicutable, image_filename, "%d"% m, "%d" % count, temp_dir+'\\'])
        
        #open the datafile
        slic_data = os.path.join(temp_dir,'image.dat')
        slic_eg = np.fromfile(slic_data,np.int32).reshape(image.shape[:2])
        
        return slic_eg
    finally:
        shutil.rmtree(temp_dir)  # delete directory