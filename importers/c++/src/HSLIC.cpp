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

template<typename T>
void load_rle_data(T* data_array, size_t number_of_ellements, std::vector<int>& slic_indexes)
{
	auto itter = slic_indexes.begin();

	for (size_t elid = 0; elid < number_of_ellements; elid++)
	{
		T run_lenght = data_array[elid];
		T value = data_array[number_of_ellements + elid];

		itter = std::fill_n(itter, run_lenght, value);
	}

}


HSLIC::HSLIC(const std::string& file) throw(std::exception)
{
	//--------Open the mat file-----------------------------------------------------
	mat_t *matfp;
	matfp = Mat_Open(file.c_str(), MAT_ACC_RDONLY);
	if (NULL == matfp) {
		throw std::exception( (std::string("Unable to open MAT file: ") + file).c_str() );
	}


	//--------Read the shape of the image-------------------------------------------
	{
		matvar_t *image_shape = Mat_VarRead(matfp, "image_shape");
		if (image_shape == NULL)
		{
			Mat_Close(matfp);
			throw std::exception((std::string("File dose not include image_shape. Error while loading file: ") + file).c_str());
		}

		if (image_shape->rank != 2 || image_shape->isComplex ||
			image_shape->dims[0] != 1 || image_shape->dims[1] != 3)
		{
			Mat_VarFree(image_shape);
			Mat_Close(matfp);
			throw std::exception((std::string("image_shape is invalid. Error while loading file: ") + file).c_str());
		}

		if (image_shape->data_type == MAT_T_INT64)
		{
			int64_t* image_shape_data = (int64_t*)image_shape->data;
			rows = image_shape_data[0];
			cols = image_shape_data[1];
		}
		else if (image_shape->data_type == MAT_T_INT32)
		{
			int32_t* image_shape_data = (int32_t*)image_shape->data;
			rows = image_shape_data[0];
			cols = image_shape_data[1];
		}
		else
		{
			Mat_VarFree(image_shape);
			Mat_Close(matfp);
			throw std::exception((std::string("image_shape has unknown type. Error while loading file: ") + file).c_str());
		}

		Mat_VarFree(image_shape);
	}

	//--------Read the atomic superpixels-------------------------------------------
	{
		slic_indexes.resize(rows*cols);

		matvar_t *atomic_SLIC_rle = Mat_VarRead(matfp, "atomic_SLIC_rle");

		if (atomic_SLIC_rle == NULL)
		{
			Mat_Close(matfp);
			throw std::exception((std::string("File does not include atomic_SLIC_rle. Error while loading file: ") + file).c_str());
		}

		if (atomic_SLIC_rle->rank != 2 || atomic_SLIC_rle->isComplex ||
			atomic_SLIC_rle->dims[1] != 2)
		{
			Mat_VarFree(atomic_SLIC_rle);
			Mat_Close(matfp);
			throw std::exception((std::string("atomic_SLIC_rle is invalid. Error while loading file: ") + file).c_str());
		}

		if (atomic_SLIC_rle->data_type == MAT_T_INT64)
		{
			load_rle_data((int64_t*)atomic_SLIC_rle->data, atomic_SLIC_rle->dims[0], slic_indexes);
		}
		else if (atomic_SLIC_rle->data_type == MAT_T_INT32)
		{
			load_rle_data((int32_t*)atomic_SLIC_rle->data, atomic_SLIC_rle->dims[0], slic_indexes);
		}
		else
		{
			Mat_VarFree(atomic_SLIC_rle);
			Mat_Close(matfp);
			throw std::exception((std::string("atomic_SLIC_rle has unknown type. Error while loading file: ") + file).c_str());
		}

		Mat_VarFree(atomic_SLIC_rle);
	}

	//--------Compute the lookup table-----------------------
	int number_of_atomic_superpixels = *std::max_element(slic_indexes.begin(), slic_indexes.end()) + 1;
	atomic_lookup_table.resize(number_of_atomic_superpixels);

	for (int i = 0; i < slic_indexes.size(); i++)
		atomic_lookup_table[slic_indexes[i]].push_back(i);

	//--------Read the HSLIC tree-------------------------------------------
	{
		matvar_t *hslic_tree_var = Mat_VarRead(matfp, "HSLIC");

		if (hslic_tree_var == NULL)
		{
			Mat_Close(matfp);
			throw std::exception((std::string("File does not include HSLIC. Error while loading file: ") + file).c_str());
		}
		try
		{
			load_region_node_from_cell_array(hslic_tree_var, tree_root);
		}
		catch (...)
		{
			Mat_VarFree(hslic_tree_var);
			throw;
		}
		Mat_VarFree(hslic_tree_var);
	}


	Mat_Close(matfp);
}


int HSLIC::get_rows() const
{
	return rows;
}

int HSLIC::get_cols() const
{
	return cols;
}

int HSLIC::get_atomic_superpixels_count() const
{
	return atomic_lookup_table.size();
}


/*
Helper function for superpixels_at_scale. Will return the largest superpixels that are a scall less then "scale."
*/
void recursive_build_superpixel_list_at_scale(const std::vector<HSLICNode>& tree_list, float scale, std::vector<HSLICSuperpixel>& out)
{
	for (HSLICNode node : tree_list)
	{
		if (node.region.scale <= scale || node.children_node.size() == 0)
			out.push_back(node.region);
		else
			recursive_build_superpixel_list_at_scale(node.children_node, scale, out);
	}
}


/*
Returns a vector of HSLICSuperpixel containing all
the superpixels at a given scale
*/
std::vector<HSLICSuperpixel> HSLIC::superpixels_at_scale(float scale) const
{
	std::vector<HSLICSuperpixel> out;
	recursive_build_superpixel_list_at_scale(tree_root, scale, out);
	return out;
}


const std::vector<int>& HSLIC::get_atomic_superpixel_indicator()
{
	return slic_indexes;
}

/*Returns the superpixels at a given scale.
This is an indicator array in row major order */
std::vector<int> HSLIC::superpixels_indicator(const std::vector<HSLICSuperpixel>& superpixels) const
{
	std::vector<int> indicator(rows*cols,-1);
	for (int superpixel_id = 0; superpixel_id < superpixels.size(); superpixel_id++)
	{
		//Go through each atomic_superpixel of superpixel_id
		for (int atomic_superpixel : superpixels[superpixel_id].atomic_superpixels)
		{
			//Mark each index of each superpixel
			for (int index : atomic_lookup_table[atomic_superpixel])
				indicator[index] = superpixel_id;
		}
	}

	return indicator;
}



const std::vector<HSLICNode>& HSLIC::get_superpixel_trees()
{
	return tree_root;
}
