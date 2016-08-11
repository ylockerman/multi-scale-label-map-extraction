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

	//--------Read the base superpixels-------------------------------------------
	{
		slic_indexes.resize(rows*cols);

		matvar_t *base_SLIC_rle = Mat_VarRead(matfp, "base_SLIC_rle");

		if (base_SLIC_rle == NULL)
		{
			Mat_Close(matfp);
			throw std::exception((std::string("File dose not include base_SLIC_rle. Error while loading file: ") + file).c_str());
		}

		if (base_SLIC_rle->rank != 2 || base_SLIC_rle->isComplex ||
			base_SLIC_rle->dims[1] != 2)
		{
			Mat_VarFree(base_SLIC_rle);
			Mat_Close(matfp);
			throw std::exception((std::string("base_SLIC_rle is invalid. Error while loading file: ") + file).c_str());
		}

		if (base_SLIC_rle->data_type == MAT_T_INT64)
		{
			load_rle_data((int64_t*)base_SLIC_rle->data, base_SLIC_rle->dims[0], slic_indexes);
		}
		else if (base_SLIC_rle->data_type == MAT_T_INT32)
		{
			load_rle_data((int32_t*)base_SLIC_rle->data, base_SLIC_rle->dims[0], slic_indexes);
		}
		else
		{
			Mat_VarFree(base_SLIC_rle);
			Mat_Close(matfp);
			throw std::exception((std::string("base_SLIC_rle has unknown type. Error while loading file: ") + file).c_str());
		}

		Mat_VarFree(base_SLIC_rle);
	}



	Mat_Close(matfp);
}


int HSLIC::get_rows()
{
	return rows;
}

int HSLIC::get_cols()
{
	return cols;
}


const std::vector<int>& HSLIC::get_base_superpixels()
{
	return slic_indexes;
}


const std::vector<HSLICNode>& HSLIC::get_superpixel_trees()
{
	return root_slic;
}
