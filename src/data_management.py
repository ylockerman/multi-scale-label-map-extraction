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


This file must be inported before matplotlib

Created on Tue Apr 16 11:10:50 2013

@author: Yitzchak David Lockerman
"""

import ConfigParser
import os
import zipfile
import tempfile
import sys
import textwrap

import matplotlib as mpl
#Set up the setting needed for matplotlib, must be done before pyplot is imported
#mpl.use('ps')
import matplotlib.pyplot as plt


from datetime import datetime


_config_file = ConfigParser.SafeConfigParser()
_config_file.read(['texture_dataset.cfg', 
                      os.path.expanduser('~/.texture_dataset.cfg')])

def get_full_data_path(name,section=ConfigParser.DEFAULTSECT):
    """
        Returns the file to load the data from
    """
    location = _config_file.get(section,'input_path')
    return os.path.join(location,name)
    
    
class ExperimentOutput(object):
    """
        ExperimentOutput is the base class for output log entries 
    """
    
    def __init__(self):
        self.time_stamp = datetime.now()
    
    def get_text_log(self):
        raise NotImplementedError()
    
    def add_to_archive(self,arcive_file):
        pass;
    
    def close(self):
        pass;
    
class TextExperimentOutput(ExperimentOutput):
    """
        TextExperimentOutput logs a simple text message.
    """
    def __init__(self,text):
        super(TextExperimentOutput,self).__init__()
        self.text = text
    
    def get_text_log(self):
        return self.text


class FileOutput(ExperimentOutput):
    """
        Output that is stored in a file until it is placed in the 
        arcive. 
        
        if existingtempfile is not provided, the file will be consdered temp.
        otherwise setting treat_as_tmp as true will return it to a temp file
        
    """
    def __init__(self,filename, existingtempfile = None, treat_as_tmp=False ):
        super(FileOutput,self).__init__()
        self._filename = filename
        
        if existingtempfile is None:
            extention = os.path.splitext(filename)[1].lower()
            if extention[0].startswith('.'):
                suffix = extention
            else:
                suffix = ''
                
            fd, self._tempfilepath = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            self._is_temp = True
        else:
            self._tempfilepath = existingtempfile
            self._is_temp = treat_as_tmp
    
    def __del__(self):  
        self.close()
            
    def close(self):
        try:
            if self._is_temp:
                os.remove(self._tempfilepath)
                self._is_temp = False #don't double delete
        except:
            print self._tempfilepath, "not closed correctly!! (temp file for",\
                                        self._filename, ")"        
        
    def open_file(self,mode='w+b'):
        return open(self._tempfilepath,mode)
    
    
    def get_text_log(self):
        return "Created file " +  self._filename 
    
    def add_to_archive(self,arcive_file):
       arcive_file.write(self._tempfilepath,self._filename)
    
    
    
  
class Experiment(object):
    """
        The Experiment class just stores the files then deletes them
        it allows code to run unchanged
    """
    def __init__(self,name=None):
        pass
    
    def __del__(self):        
        pass
        
    def close(self):
        """Closese the experiment file, and creates the final output"""
        pass

            
            
    def output(self,out):
        """
        Give some precreated output
        """
        if not isinstance(out,ExperimentOutput):
            raise ValueError('Given non ExperimentOutput type '+str(type(out)))
        
        out.close()
    
    
    def print_(self,text):
        """
        Print plain text
        """
        text = textwrap.dedent(text)
        self.output(TextExperimentOutput(text))
        print text
        
        
    def add_temp_file(self,filename,existing_file = None,mode='w+b'):
        """
            Adds a file to the experiment results. 
            If existing_file is provided, it is used as the source
            
            The orignal file will be deleated after being copyed
            
            otherwise a temp file will be created and opend with the fd returnd
        """
        
        tfo = FileOutput(filename,existing_file,treat_as_tmp=True)
        self.output(tfo)
        
        if existing_file is None:
            return tfo.open_file(mode)
    
    def add_file(self,existing_file_name,filename=None):
        """
            Adds a file to the experiment results. 
        """
        
        if filename is None:
            filename = os.path.basename(existing_file_name)
        
        tfo = FileOutput(filename,existing_file_name)
        self.output(tfo)      
        
    
    def add_files(self,list_of_files):
        """
            Adds a list of files to the experiment log.
            The format can be either:
                    A dict with existing files as the keys and new name as the 
                        item
                    A list of files, or pairs of existing file names, new
                        file names
        """
        
        if isinstance(list_of_files,dict):
            for existing_file, new_file in list_of_files.iteritems():
                self.add_file(existing_file, new_file)
        else:
            for file_item in list_of_files:
                if isinstance(file_item,tuple):
                    self.add_file(file_item[0], file_item[1])
                else:
                    self.add_file(file_item)   
                    
    def add_figure(self,figure_name,fig=None,close_after=True,dpi=300):
        """
            Saves a matplotlib figure
        """
        if fig is None:
            fig_use = plt.gcf()
        else:
            fig_use = fig
        
        extention = os.path.splitext(figure_name)[1].lower()
        if extention.startswith('.'):
            raise ValueError('Images can not specify extention supported at this time')
        

        with self.add_temp_file(figure_name+".png") as output_png:
            fig_use.savefig(output_png,dpi=dpi,format='png',transparent=True)          
        #with self.add_temp_file(figure_name+".eps") as output_eps:
        #    fig_use.savefig(output_eps,dpi=dpi,format='eps',transparent=True)
          
            
        if close_after:
                plt.close(fig_use)
        
        
    def add_array(self,arary_name,array):
        import numpy as np
        
        extention = os.path.splitext(arary_name)[1].lower()
        if extention.startswith('.'):
            raise ValueError('arrays can not specify extention supported at this time')
            
        with self.add_temp_file(arary_name+".npy") as output_array:
            np.save(output_array,array)
        
    def add_image(self,file_name,image):
        """
            Saves an image
        """
        import skimage.io
        
        extention = os.path.splitext(file_name)[1].lower()
        if extention.startswith('.'):
            with self.add_temp_file(file_name) as output:
                skimage.io.imsave(output,image)    
        else:
            #http://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
            with self.add_temp_file(file_name+".png") as output_png:
                skimage.io.imsave(output_png,image)
            
    def set_name(self,name):
        folder = _config_file.get(ConfigParser.DEFAULTSECT,'output_location')
        file_name = name + self._timestamp  + ".zip"
        self._out_file = os.path.join(folder,file_name)    
        
        print "Will store output in" , self._out_file, "(unless changed)"
    
  
class FileExperiment(Experiment):
    """
        The File  Experiment class is a full log of an experiment in a zip file
    """
    def __init__(self,name):
        self._timestamp = datetime.now().strftime("_%b_%d_%Y__%H_%M_%S_%f")
        self._reccord = []
        self.is_alive = True
        
        self.set_name(name) 
    
    def __del__(self):        
        self.close()
        
    def close(self):
        """Closese the experiment file, and creates the final output"""
        #Only close if are not yet finished
        if self.is_alive:
            self.is_alive = False
            
            #create the text log 
            self._create_text_log()

            #create the arcive file
            self._create_final_arcive()
            
            #free the memory
            for rec in self._reccord:
                rec.close()
            del self._reccord
            
            
    def _create_text_log(self):
        """
            Create the fill text log
        """
        full_text_log = ''.join([
                                    "["+str(rec.time_stamp)+"] " +\
                                            rec.get_text_log() +"\n"  \
                                                for rec in self._reccord
                                ])
                                
        #add the text output 
        text_log = FileOutput("text_log.txt")
        with text_log.open_file() as f:
            f.write(full_text_log)
            
        self._reccord.append(text_log)        


    #creates the final arcive"
    def _create_final_arcive(self):
        """
            create the final arcive
        """
        with  zipfile.ZipFile(self._out_file,'w',zipfile.ZIP_DEFLATED) \
                                                                as arcive_file:
            for file_in_arcive in self._reccord:
                try:
                    file_in_arcive.add_to_archive(arcive_file)
                except:
                    print "Error adding item to arcive due to error: ", \
                                                           sys.exc_info()[0]
            
            
    def output(self,out):
        """
        Give some precreated output
        """
        if not self.is_alive:
            raise RuntimeError('Attempting to output to a closed experiment log')

        if not isinstance(out,ExperimentOutput):
            raise ValueError('Given non ExperimentOutput type '+str(type(out)))
        
        self._reccord.append(out)
    
    
 
    
    
        

###Interface for the commen experiment
_current_experiment = Experiment()

def create_experiment(name):
    """
        Creates a new common experiment with a given name. If an experiment 
        already exists, it may just rename it
    """
    global _current_experiment
    
    if not _current_experiment is None and isinstance(_current_experiment,FileExperiment):
        _current_experiment.set_name(name)
        
    else:
        _current_experiment = FileExperiment(name)
        

    
def current_experiment():
    """
        Returns the instance of the current experiment. 
        Will create a new experiment if none exists
    """
    
    return _current_experiment
    
def print_(text):
    """
        Prints a text message to the common experiment
    """
    current_experiment().print_(text)


def add_figure(figure_name,fig=None,close_after=True,dpi=300):
    """
        Saves a matplotlib figure
    """
    current_experiment().add_figure(figure_name,fig,close_after,dpi)

def add_image(file_name,image):
    """
        Saves an image
    """
    current_experiment().add_image(file_name,image)
    
def add_array(arary_name,array):
    """
        Saves an image
    """
    current_experiment().add_array(arary_name,array)    
    
def close():
    global _current_experiment
    _current_experiment.close();
    _current_experiment = None
        
####
        
####extra util to tee output when an experiment can't be used
#From http://shallowsky.com/blog/programming/python-tee.html
def tee_out():
    import sys
    
    class tee :
        def __init__(self, _fd1, _fd2) :
            self.fd1 = _fd1
            self.fd2 = _fd2
    
        def __del__(self) :
            if self.fd1 != sys.stdout and self.fd1 != sys.stderr :
                self.fd1.close()
            if self.fd2 != sys.stdout and self.fd2 != sys.stderr :
                self.fd2.close()
    
        def write(self, text) :
            self.fd1.write(text)
            self.fd2.write(text)
    
        def flush(self) :
            self.fd1.flush()
            self.fd2.flush()
    stdoutsav = sys.stdout
    outoutputlog = open("stdout.txt", "w",0)
    sys.stdout = tee(stdoutsav, outoutputlog)
    
    stderrsav = sys.stderr
    erroutputlog = open("stderr.txt", "w",0)
    sys.stderr = tee(stderrsav, erroutputlog)