# multi-scale-label-map-extraction
Automatically find and labels all textures within an image. 

This is based on the source for the paper 

["Multi-scale label-map extraction for texture synthesis"](http://graphics.cs.yale.edu/site/publications/multi-scale-label-map-extraction-texture-synthesis)
in the ACM Transactions on Graphics (TOG) - Proceedings of ACM SIGGRAPH 2016, Volume 35 Issue 4, July 2016 
by Lockerman, Y.D., Sauvage, B., Allegre, R., Dischler, J.M., Dorsey, J. and Rushmeier, H.

If you find it useful, please consider giving us credit or citing our paper.   

**Note that documentation is still a work in progress.**

# Requirements 

**Note that thtis is a partial list. If you find that something is missing or no longer needed, please let us know!**

* numpy
* scipy
* bottleneck
* joblib
* Mako
* msgpack_python
* pyamg
* pyopencl
* pytools
* scikits.ann
* sympy

To output standalone executable you also need:
* PyInstaller 
* cx_Freeze

ALso, some features use matplotlib
