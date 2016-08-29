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

#include "HSLIC.h"
#include "matio.h"
#include <stdint.h>
#include <algorithm>  
#include <iostream>

int get_total_element_count(matvar_t* arry)
{
	int number_of_element = 1;
	for (int dim = 0; dim < arry->rank; dim++)
		number_of_element *= arry->dims[dim];
	
	return number_of_element;
}

/*
Returns a field within a struct_array. Note that the item should not be freed. 
*/
matvar_t* get_struct_field(matvar_t* struct_array, std::string name, bool optional=false)
{
	matvar_t* out = Mat_VarGetStructField(struct_array, (void*)name.c_str(), MAT_BY_NAME, 0);

	if (out == NULL || out->nbytes <= 0)
	{
		if (optional)
			return NULL;
		else
			throw std::exception( ("Struct is missing expected element: " + name).c_str());
	}

	return out;
}

/*
Loads a region (Superpixel or label) from a struct_array 
*/
void load_region(matvar_t* struct_array, Region& out)
{
	if (struct_array->data_type != MAT_T_STRUCT )
	{
		throw std::exception("cells should have struct array");
	}

	//load the scale. This will need to be a float or double. 
	matvar_t* scale = get_struct_field(struct_array, "scale");

	if (scale->data_type == MAT_T_SINGLE)
	{
		out.scale = *((float*)scale->data);
	}
	else if (scale->data_type == MAT_T_DOUBLE)
	{
		out.scale = *((double*)scale->data);
	}
	else
	{
		throw std::exception("Scale has unknown type");
	}

	//load the list of atomic superpixels.
	//This will need to be a array of 32 or 64 bit integers.
	matvar_t* list_of_atomic_superpixels = get_struct_field(struct_array, "list_of_atomic_superpixels");

	int number_of_atomic_superpixels = list_of_atomic_superpixels->nbytes / list_of_atomic_superpixels->data_size;
	out.atomic_superpixels.resize(number_of_atomic_superpixels);

	if (list_of_atomic_superpixels->data_type == MAT_T_INT32)
	{
		int32_t* atomic_list = (int32_t*)list_of_atomic_superpixels->data;
		std::copy(atomic_list, atomic_list + number_of_atomic_superpixels, out.atomic_superpixels.begin());
	}
	else if (list_of_atomic_superpixels->data_type == MAT_T_INT64)
	{
		int64_t* atomic_list = (int64_t*)list_of_atomic_superpixels->data;
		std::copy(atomic_list, atomic_list + number_of_atomic_superpixels, out.atomic_superpixels.begin());
	}
	else
	{
		throw std::exception("list_of_atomic_superpixels has unknown type");
	}
}

/*
This method loads a region from the tree format used in our file.
*/
void load_region_node_from_cell_array(matvar_t* cell_array, std::vector<RegionNode>& output)
{
	if (cell_array->data_type != MAT_T_CELL || cell_array->nbytes < 1)
		throw std::exception("Invalid tree data structure");

	int number_of_cells = get_total_element_count(cell_array);

	matvar_t **all_cells = Mat_VarGetCellsLinear(cell_array, 0, 1, number_of_cells);

	if (all_cells == NULL)
		throw std::exception("Invalid tree data structure");

	output.resize(number_of_cells);
	for (int i = 0; i < number_of_cells; i++)
	{
		try
		{
			//Find the elements of intrest and use them
			load_region(all_cells[i], output[i].region);

			//Recursivly load the subregions
			matvar_t * children = get_struct_field(all_cells[i], "children", true);

			if (children != NULL)
				load_region_node_from_cell_array(children, output[i].children_node);
		}
		catch (...)
		{
			free(all_cells);
			throw;
		}
	}


	free(all_cells);
}