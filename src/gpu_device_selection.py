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

Created on Wed Feb 04 13:53:06 2015

@author: Yitzchak David Lockerman
"""




from gpu import opencl_tools
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_algorithm = opencl_tools.cl_algorithm
elementwise = opencl_tools.cl_elementwise
cl_reduction = opencl_tools.cl_reduction
cl_scan = opencl_tools.cl_scan


import ConfigParser
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

matplotlib.rcParams['toolbar'] = 'None'

import multiprocessing

def select_gpu(default_platform=None, defualt_device=None):
    if default_platform is None:
        default_platform =opencl_tools.openc_platform_index
        
    if defualt_device is None:
        defualt_device = opencl_tools.openc_device_index
        
    figure = plt.figure()
    figure_number = figure.number
    selected_options = {'platform' : default_platform, 
                        'device' : defualt_device,
                        'platforms' : cl.get_platforms(),
                        'devices' : cl.get_platforms()[default_platform].get_devices(),
                        'return_values' : False }

    platform_names = [ "%d. %s" % (index,platform.get_info(cl.platform_info.NAME))
                            for index,platform in enumerate(selected_options['platforms'])]

    def get_device_names():
         return [ "%d. %s" % (index,device.get_info(cl.device_info.NAME))
                                    for index,device in enumerate(selected_options['devices'])]        
    
    grid_size = (6,6)
    ax_platform_selection = plt.subplot2grid(grid_size,(0,0),rowspan=5,colspan=3)
    ax_device_selection = plt.subplot2grid(grid_size,(0,3),rowspan=5,colspan=3)
    ax_save_button = plt.subplot2grid(grid_size,(5,4),rowspan=1,colspan=1)    
    ax_cancel_button = plt.subplot2grid(grid_size,(5,5),rowspan=1,colspan=1)  
    
    ax_platform_selection.set_title("Platform:")
    ax_device_selection.set_title("Device:")
    
    platform_selection = RadioButtons(ax_platform_selection,platform_names,active=selected_options['platform'])
    device_selection =[ RadioButtons(ax_device_selection,get_device_names(),active=selected_options['device']) ]
    
    def on_platform_select(selected_name):
        id_of_platform = platform_names.index(selected_name)

        selected_options['platform'] = id_of_platform
        selected_options['device'] = 0
        selected_options['devices'] = cl.get_platforms()[id_of_platform].get_devices()
        
        device_selection[0].disconnect_events()
        allobs = list(iter(device_selection[0].observers))
        for obs in allobs:
            device_selection[0].disconnect(obs)
        ax_device_selection.cla();
        device_selection[0] = RadioButtons(ax_device_selection,get_device_names(),active=selected_options['device'])
        ax_device_selection.set_title("Device:")
        figure.canvas.draw()
        
    platform_selection.on_clicked(on_platform_select)
    
    def on_device_select(selected_name):  
        device_names = get_device_names()
        selected_options['device'] = device_names.index(selected_name)
    device_selection[0].on_clicked(on_device_select)
    
    save_button = Button(ax_save_button,"Save")
    def save_click(*args):
        selected_options['return_values'] = True
        plt.close(figure)
    save_button.on_clicked(save_click)
    
    
    cancel_button = Button(ax_cancel_button,"Cancel")
    def cancel_click(*args):
        selected_options['return_values'] = False
        plt.close(figure)
    cancel_button.on_clicked(cancel_click)   
    
    
    plt.show(block=True)

    if selected_options['return_values']:
        return selected_options['platform'], selected_options['device']
    else:
        return None, None
        
def save_gpu_to_file(file_name):
    platform,device = select_gpu()
    if platform is not None:
        config = ConfigParser.RawConfigParser()
        config.set("DEFAULT","OPENCL_PLATFORM_INDEX",platform)
        config.set("DEFAULT","OPENCL_DEVICE_INDEX",device)
    
        with open(file_name, 'wb') as configfile:
            config.write(configfile)
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    save_gpu_to_file('gpu_conf.ini')