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

Created on Mon Feb 02 16:38:08 2015

@author: Yitzchak David ockerman
"""
import cx_Freeze
import tempfile
import os
import sys
import pkg_resources
import zipfile
import string
import shutil

import PyInstaller
import PyInstaller.main

#From: http://stackoverflow.com/questions/21356551/python-zipping-a-folder-file
def zip_dir(zipname, dir_to_zip):
    dir_to_zip_len = len(dir_to_zip.rstrip(os.sep)) + 1
    with zipfile.ZipFile(zipname, mode='w') as zf:
        for dirname, subdirs, files in os.walk(dir_to_zip):
            for filename in files:
                path = os.path.join(dirname, filename)
                entry = path[dir_to_zip_len:]
                zf.write(path, entry)


if __name__ == "__main__":
    main_system_temp = None
    single_file_temp = None
    try:
        from Tkinter import Tk
        from tkFileDialog import asksaveasfilename
        Tk().withdraw()
        save_file_name = asksaveasfilename(filetypes=[('zip','*.zip')],
                                                          defaultextension=".zip")

        if(save_file_name is  None):
            import sys
            sys.exit(0)

        # Assume using WinPython
        #From: http://stackoverflow.com/questions/12041525/a-system-independent-way-using-python-to-get-the-root-directory-drive-on-which-p
        SITE_PACKAGE_DIR = (os.path.split(sys.executable)[0] +
                                        '\\Lib\\site-packages\\')
                                        
        # Include packages not detected by cx_Freeze
        includes = ['matplotlib.backends.backend_qt4agg', 
                    'scipy.special._ufuncs_cxx',
                    'scipy.sparse.csgraph._validation',
                    #'scipy.sparse.linalg.dsolve.umfpack',
                    'scipy.integrate.vode',
                    'scipy.integrate.lsoda'
                   ]
        # includes += ['skimage.io._plugins.'+modname
                        # for __, modname, __ in 
                            # pkgutil.iter_modules(skimage.io._plugins.__path__) ]
        
        # Include data/package files not accounted for by cx_Freeze
        import matplotlib
        include_files = [(matplotlib.get_data_path(), 'mpl-data'),
                         ('matplotlibrc', '')]
        
        #A simple util to get the egg dir name
        def get_egg_dir(name):
            base_name = pkg_resources.get_distribution(name).egg_name()
            for posible_extention in ['.egg-info','.dist-info']:
                full_path = os.path.join(SITE_PACKAGE_DIR , base_name + posible_extention)
                print full_path
                if os.path.exists(full_path):
                    return base_name+posible_extention
            
            
            
        def get_pth_dir(name):
            return pkg_resources.get_distribution(name).egg_name()+'-nspkg.pth'  
            
        def get_files_for_package_start(name):
            try:
                location = pkg_resources.get_distribution(name).location
            except:
                location = SITE_PACKAGE_DIR
                
            return [filename for filename in os.listdir(location)
                                                if filename.startswith(name)]
        
        # Simply include entire package, to save effort
        full_includes = ['skimage',  'pygments']
                         
        full_includes += get_files_for_package_start('pyopencl')
        full_includes += get_files_for_package_start('scikits.ann')        
        full_includes += get_files_for_package_start('mpl_toolkits')        
        full_includes += get_files_for_package_start('sklearn')           
        
                        
        for package in full_includes:
            if package is not None:
                include_files.append( (SITE_PACKAGE_DIR + package, package) )
            
        executables = [cx_Freeze.Executable('texture_selection.py', base=None),
                       cx_Freeze.Executable('multiscale_extraction.py', base=None),
                       cx_Freeze.Executable('gpu_device_selection.py', base=None),
                       cx_Freeze.Executable('compare_clusters.py', base=None)]
        
        main_system_temp = tempfile.mkdtemp();

        freezer = cx_Freeze.Freezer(executables,
                                    includes=includes,
                                    includeFiles=include_files,
                                    targetDir=main_system_temp)
        freezer.Freeze();
        
        zip_dir(save_file_name,main_system_temp)
    except:
        raise;
    finally:           
        if main_system_temp:                      
            shutil.rmtree(main_system_temp)