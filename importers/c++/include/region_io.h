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
#ifndef REGION_IO_H
#define REGION_IO_H

#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <map>
#include "region_map.h"

//////////////////////////////////Internal Use//////////////////////////////////////////////////////

struct matvar_t;


class matvar_raii : public std::unique_ptr<matvar_t, void (*)(matvar_t *)>
{
	matvar_raii();
public:
	matvar_raii(matvar_t* ptr);

	operator matvar_t* ();

};


struct _mat_t;

class mat_raii : public std::unique_ptr<_mat_t, int(*)(_mat_t *)>
{
	mat_raii();
public:
	mat_raii(_mat_t* ptr);

	operator _mat_t* ();

};

/*
Loads extra data that may of included in an array.
*/
void load_extra_data(matvar_t * struct_array, ExtraDataMap& out, std::string ignore_names[], int number_of_ignore_names);

/*
Loads a CompoundRegion from a struct_array. This is the fomat
used to store a region when it is not part of a tree.
*/
CompoundRegion load_region(matvar_t* struct_array, bool is_hierarchical);

/*
Loads a stack of compound region maps from a matlab variable.
*/
std::map<float, std::pair< std::vector<CompoundRegionPtr>, ExtraDataMap> > load_region_stack_from_cell_array(matvar_t* cell_array);

/*
This method loads a region list format used in our file.
*/
std::vector<CompoundRegionPtr> load_region_node_from_cell_array(matvar_t* cell_array);


/*
Loads a HierarchicalRegion set from the tree format used in our file.
*/
std::vector<HierarchicalRegionPtr> load_hierarchical_region_node_from_cell_array(matvar_t* cell_array);


struct image_size {
	size_t rows, cols, stride;
};

/*
Loads the size of the image from a mat file variable.
*/
image_size load_image_size(matvar_t *image_shape);

/*
Loads the atomic superpixels from a  rle encoded image
*/
std::valarray<size_t> load_atomic_regions_from_rle(matvar_t *atomic_region_rle, const image_size size);


template<typename ImageData>
AtomicRegionMap<ImageData> load_atomic_region_map(_mat_t* matfp, std::string image_shape_name = "image_shape", std::string atomic_region_rle_name = "atomic_SLIC_rle",
													size_t stride = 0)
{
	//Read the image shape
	matvar_raii image_shape = Mat_VarRead(matfp, image_shape_name.c_str());
    image_size is = load_image_size(image_shape);

	//Read the atomic superpixels
	matvar_raii  atomic_region_rle = Mat_VarRead(matfp, atomic_region_rle_name.c_str());
	std::valarray<size_t> atomic_regions = load_atomic_regions_from_rle(atomic_region_rle, is);

	if (stride == 0)
		stride = is.stride;

	return AtomicRegionMap<ImageData>(is.rows, is.cols, stride, atomic_regions);
}

template<typename ImageData>
std::map<float, std::shared_ptr< CompoundRegionMap<ImageData> > > load_region_map_stack(_mat_t* matfp, AtomicRegionMap<ImageData> atomic_region,
																				std::string region_list_name)
{
	matvar_raii hslic_list_var = Mat_VarRead(matfp, region_list_name.c_str());

	if (!hslic_list_var)
		throw std::exception((std::string("File does not include ") + region_list_name + ".").c_str());

	std::map<float, std::pair< std::vector<CompoundRegionPtr>, ExtraDataMap> > region_stack = load_region_stack_from_cell_array(hslic_list_var);

	std::map<float, std::shared_ptr< CompoundRegionMap<ImageData> > > output_stack;

	for (std::map<float, std::pair< std::vector<CompoundRegionPtr>, ExtraDataMap> >::iterator it = region_stack.begin(); it != region_stack.end(); it++)
	{
		output_stack[it->first] = std::make_shared<CompoundRegionMap<ImageData>> (atomic_region, it->second.first, it->second.second);
	}


	return output_stack;
}


template<typename ImageData> 
HierarchicalRegionMap<ImageData> load_hierarchical_region_map(_mat_t* matfp, std::shared_ptr<AtomicRegionMap<ImageData>> atomic_region,
	std::string region_tree_name)
{
	matvar_raii hslic_tree_var = Mat_VarRead(matfp, region_tree_name.c_str());

	if (!hslic_tree_var)
		throw std::exception((std::string("File does not include ") + region_tree_name + ".").c_str());

	std::vector<HierarchicalRegionPtr> root_region_list = load_hierarchical_region_node_from_cell_array(hslic_tree_var);

	return HierarchicalRegionMap<ImageData>(atomic_region, root_region_list);
}



template<typename ImageData>
HierarchicalRegionMap<ImageData> load_hierarchical_region_map(_mat_t* matfp, const AtomicRegionMap<ImageData>& atomic_region,
	std::string region_tree_name)
{
	matvar_raii hslic_tree_var = Mat_VarRead(matfp, region_tree_name.c_str());

	if (!hslic_tree_var)
		throw std::exception((std::string("File does not include ") + region_tree_name + ".").c_str());

	std::vector<HierarchicalRegionPtr> root_region_list = load_hierarchical_region_node_from_cell_array(hslic_tree_var);

	return HierarchicalRegionMap<ImageData>(atomic_region, root_region_list);
}

#endif