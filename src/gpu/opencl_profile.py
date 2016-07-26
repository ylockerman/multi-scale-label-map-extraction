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

Created on Wed Mar 06 21:22:50 2013

@author: Yitzchak David Lockerman
"""
from Queue import Queue
from itertools import ifilterfalse
import threading 
import traceback
import csv
import random
import sys

import numbers
import numpy as np

import pyopencl as _cl
import pyopencl.array as _cl_array
import pyopencl.reduction as _cl_reduction
import pyopencl.elementwise as _cl_elementwise
import pyopencl.algorithm as _cl_algorithm
import pyopencl.scan as _cl_scan
import pyopencl.clrandom as _cl_random

######################################################################
############################Profiler code###########################
######################################################################


file_name = 'profile_out'+str(random.randint(0,10000))+'.csv'
profile_output = open(file_name,'w')             
         

    
#the output of the file
profile_csv = csv.writer(profile_output, lineterminator = '\n')
profile_csv.writerow(('file_name','line_number','function_name','line_text',
                      'command_type','submit_time','queued_time',
                      'elapsed_time'))                      
                      
#A queue of eventsto be recorded
event_queue = Queue()

#The output file to dump results

#Trys to record an event, retruns True if the event could be recorded
def _try_record_event(full_event):
    name,evt = full_event
    if evt.get_info(_cl.event_info.COMMAND_EXECUTION_STATUS) \
                != _cl.command_execution_status.COMPLETE:
            return False;
    command_type = evt.get_info(_cl.event_info.COMMAND_TYPE)
    command_type = _cl.command_type.to_string(command_type)
    try:

    
        submit =  evt.get_profiling_info(_cl.profiling_info.SUBMIT)
        queued= evt.get_profiling_info(_cl.profiling_info.QUEUED)    
        start =  evt.get_profiling_info(_cl.profiling_info.START)
        end= evt.get_profiling_info(_cl.profiling_info.END)
        elapsed  = end-start
        
        full_info = name + (command_type,submit,queued,elapsed)
        
        profile_csv.writerow(full_info)
        profile_output.flush()
    except:
        print sys.exc_info()[0]
        print "Profilling Error!! Command type (",command_type,")"
        
    return True

def _profiler_thread():
    to_record = []
    while(True):
        #add the newist event to the list of items to record
        to_record.append(event_queue.get())
        #Solution form 
        #http://stackoverflow.com/questions/1207406/remove-items-from-a-list-while-iterating-in-python
        to_record[:] = list(ifilterfalse(_try_record_event, to_record))

profile_thread = threading.Thread(target=_profiler_thread, name="profiler_thread")
profile_thread.start()

######################################################################
##############Code to wrap all cl code, and find events###############
######################################################################

def _fix_argument(arg):
        return getattr(arg,'wrap_obj',arg)


def _fix_all_arguments(args,kwargs):
    args = tuple(map(_fix_argument,args))
    kwargs = dict(zip(kwargs.keys(),map(_fix_argument,kwargs.values())))
                   
    return args,kwargs

boost_python_type =  _cl.device_info.__class__
        
def wrap_object(obj):
       
    if isinstance(obj,numbers.Number):
        return obj
    if isinstance(obj,np.ndarray):
        return obj
    if isinstance(obj,type) and not isinstance(obj,boost_python_type):
        
        class newMetaClass(type):
            def __instancecheck__(self, other):
                return isinstance(_fix_argument(other),obj)
                
            def __subclasscheck__(self, other):
                return issubclass(_fix_argument(other),obj)
                
            def __call__(self,*args,**kwargs):
                args, kwargs = _fix_all_arguments(args,kwargs)
                return wrap_object(obj(*args,**kwargs))
                
        class newOpencClProfileWrapper(OpencClProfileWrapper,object):
            __metaclass__  = newMetaClass
            wrap_obj = obj
            
            def __getattribute__(self, name):
                if name == 'wrap_obj':
                    return self.__dict__['wrap_obj']
                    
                ret = getattr(self.__dict__['wrap_obj'],name)
                if ret.__class__ == self.__class__:
                    return ret
                else:
                    return wrap_object(ret)
                    
            def __getattr__(self, name):
                if name == 'wrap_obj':
                    return self.__dict__['wrap_obj']
                    
                ret = getattr(self.__dict__['wrap_obj'],name)
                if ret.__class__ == self.__class__:
                    return ret
                else:
                    return wrap_object(ret) 
        return newOpencClProfileWrapper
                
    return OpencClProfileWrapper(obj)



class OpencClProfileWrapper():
    
    #Unless otherwise noted, section lables are from
    #http://docs.python.org/2/reference/datamodel.html
    #3.4.1.
    def __init__(self,wrap_obj):
        self.__dict__['wrap_obj'] = wrap_obj
        
    def __del__(self):
        del self.__dict__['wrap_obj']
        
    def __repr__(self,*args,**kwargs):
        return repr(self.__dict__['wrap_obj'])
    def __str__(self,*args,**kwargs):
        return str(self.__dict__['wrap_obj'])
        
    def __lt__(self, other):
        return self.__dict__['wrap_obj'] < (_fix_argument(other))        
    def __le__(self, other):
        return self.__dict__['wrap_obj'] <= (_fix_argument(other))
    def __eq__(self, other):
        return self.__dict__['wrap_obj'] == (_fix_argument(other))
    def __ne__(self, other):
        return self.__dict__['wrap_obj'] != (_fix_argument(other))
    def __gt__(self, other):
        return self.__dict__['wrap_obj'] > (_fix_argument(other))
    def __ge__(self, other):
        return self.__dict__['wrap_obj'] >= (_fix_argument(other))

    def __cmp__(self, other):
            return self.__dict__['wrap_obj'] == _fix_argument(other)
            
    def __rcmp__(self, other):
        return _fix_argument(other) == self.__dict__['wrap_obj']
   
    def __hash__(self):
         return self.__dict__['wrap_obj'].__hash__()  
    def __nonzero__(self):
        return self.__dict__['wrap_obj'].__nonzero__()  
    def __unicode__(self):
        return self.__dict__['wrap_obj'].__unicode__()  

        
    #3.4.2.
    def __getattr__(self, name):
        if name == 'wrap_obj':
           return self.__dict__['wrap_obj']
           
        ret = getattr(self.__dict__['wrap_obj'],name)
        if isinstance(ret,OpencClProfileWrapper):
            return ret
        else:
            return wrap_object(ret)
    
    def __setattr__(self, name, value):
        if name == 'wrap_obj':
           return self.__dict__['wrap_obj']
                    
        value = _fix_argument(value)
        setattr(self.__dict__['wrap_obj'],name,value)
                
    def __delattr__(self, name):
        delattr(self.__dict__['wrap_obj'],name)

    #3.4.4
    def __instancecheck__(self, other):
        return isinstance(_fix_argument(other),self.__dict__['wrap_obj'])
        
    def __subclasscheck__(self, other):
        return issubclass(_fix_argument(other),self.__dict__['wrap_obj'])
   
   
    #3.4.5.
    def __call__(self, *args, **kwargs):
        #Update the arguments to unwrap wrappers
        
        args, kwargs = _fix_all_arguments(args,kwargs)
        #try:
        #    print self.__dict__['wrap_obj'].__class__
        #except: 
        #    pass
        
        ret = self.__dict__['wrap_obj'](*args, **kwargs)
        
        #If we get an event, we need to record it!
        if isinstance(ret,_cl.Event):
            call_from = traceback.extract_stack(limit=2)[0]
            event_queue.put((call_from,ret))
        
        if ret.__class__ == self.__class__:
            return ret
        else:
            return wrap_object(ret)    
            
    #3.4.6.
    def __len__(self):
        return self.__dict__['wrap_obj'].__len__()
    def __contains__(self, item):
         return self.__dict__['wrap_obj'].__contains__(item)

      
    #3.4.8.    
    def __complex__(self):
         return self.__dict__['wrap_obj'].__complex__()
    def __int__(self):
         return self.__dict__['wrap_obj'].__int__()
    def __long__(self):
         return self.__dict__['wrap_obj'].__long__()
    def __float__(self):
         return self.__dict__['wrap_obj'].__float__()
    def __oct__(self):
         return self.__dict__['wrap_obj'].__oct__()
    def __hex__(self):
         return self.__dict__['wrap_obj'].__hex__()
    def __index__(self):
         return self.__dict__['wrap_obj'].__index__()
    def __coerce__(self,other):
         try:
             return self.__dict__['wrap_obj'].__coerce__(other)
         except:
             return None
    #http://docs.python.org/2/library/functions.html
    def __dir__(self):
         return self.__dict__['wrap_obj'].__dir__()
 

  
cl = OpencClProfileWrapper(_cl)
cl_array = OpencClProfileWrapper(_cl_array)
cl_reduction = OpencClProfileWrapper(_cl_reduction)
cl_elementwise = OpencClProfileWrapper(_cl_elementwise)
cl_algorithm = OpencClProfileWrapper(_cl_algorithm)
cl_scan = OpencClProfileWrapper(_cl_scan)
cl_rand = OpencClProfileWrapper(_cl_random)