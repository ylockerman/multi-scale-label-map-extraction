# multi-scale-label-map-extraction
Automatically find and labels all textures within an image. 

This is based on the source for the paper. 

["Multi-scale label-map extraction for texture synthesis"](http://graphics.cs.yale.edu/site/publications/multi-scale-label-map-extraction-texture-synthesis)
in the ACM Transactions on Graphics (TOG) - Proceedings of ACM SIGGRAPH 2016, Volume 35 Issue 4, July 2016 
by Lockerman, Y.D., Sauvage, B., Allegre, R., Dischler, J.M., Dorsey, J. and Rushmeier, H.



If you find it useful, please consider giving us credit or citing our paper.   

**Note: This repository is a work in progress. We are actively working on improving the quality 
of the quality of the code to transitioning the research code to produce a production-ready library.  
We are currently reorganizing the code, adding importers, and updating the documentation.**

**Feel free to contact us if you have any questions, have any comments, or need any help making use of this code.**

# Requirements 

* numpy
* scipy
* bottleneck
* joblib
* Mako
* msgpack_python
* pyamg
* pyopencl
* pytools
* scikit-image
* matplotlib
* scikit-learn

To output standalone executable you also need:
* PyInstaller 
* cx_Freeze
