/*
------------------------------------------------------------------------------ -
MIT License

Copyright(c) 2016 Yale University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is based on the source for the paper

"Multi-scale label-map extraction for texture synthesis"
in the ACM Transactions on Graphics(TOG) - Proceedings of ACM SIGGRAPH 2016,
Volume 35 Issue 4, July 2016
by
Lockerman, Y.D., Sauvage, B., Allegre,
R., Dischler, J.M., Dorsey, J.and Rushmeier, H.

http://graphics.cs.yale.edu/site/publications/multi-scale-label-map-extraction-texture-synthesis

If you find it useful, please consider giving us credit or citing our paper.
------------------------------------------------------------------------------ -

@author : Yitzchak David Lockerman
*/


#pragma once
#ifndef LABEL_MAP_H
#define LABEL_MAP_H

#include <vector>
#include <string>
#include <exception>

#include "region_map.h"
#include "region_io.h"
#include "matio.h"


template <typename ImageData>
HierarchicalRegionMap<ImageData> load_hierarchical_label_map(const std::string& file, int stride=0)
{
	mat_raii matfp = Mat_Open(file.c_str(), MAT_ACC_RDONLY);

	if (!matfp)
		throw std::exception((std::string("Unable to open MAT file: ") + file).c_str());


	AtomicRegionMap<ImageData> SLIC = load_atomic_region_map<ImageData>(matfp, "image_shape", "atomic_SLIC_rle", stride);

	HierarchicalRegionMap<ImageData> hierarchical_label_map = load_hierarchical_region_map<ImageData>(matfp, SLIC, "texture_tree");

	return hierarchical_label_map;
}



template <typename ImageData>
std::map<float, std::shared_ptr< CompoundRegionMap<ImageData> > > load_label_map_stack(const std::string& file, int stride = 0)
{
	mat_raii matfp = Mat_Open(file.c_str(), MAT_ACC_RDONLY);

	if (!matfp)
		throw std::exception((std::string("Unable to open MAT file: ") + file).c_str());


	AtomicRegionMap<ImageData> SLIC = load_atomic_region_map<ImageData>(matfp, "image_shape", "atomic_SLIC_rle", stride);

	std::map<float, std::shared_ptr< CompoundRegionMap<ImageData> > > region_map_stack = load_region_map_stack(matfp, SLIC, "texture_lists");

	return region_map_stack;
}




#endif