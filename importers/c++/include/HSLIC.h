#pragma once
#ifndef HSLIC_H
#define HSLIC_H


#include <vector>
#include <string>
#include <exception>
#include "region_tree.h"

typedef RegionNode HSLICNode;

/*
This class stores a tree of HSLIC data.
*/
class HSLIC
{
	int rows;
	int cols;
	std::vector<int> slic_indexes;
	std::vector<HSLICNode> root_slic;
public:

	/*
	Lodes the multiscale texture set from a file
	*/
	HSLIC(const std::string& file) throw(std::exception);

	/*
	Retruns the number of rows in the image
	*/
	int get_rows();

	/*
	Retruns the number of columns in the image
	*/
	int get_cols();

	/*
	Returns a row by col array which states what superpixel each pixel in the image belongs to.
	This is in Row major order
	*/
	const std::vector<int>& get_base_superpixels();

	/*
	Returns the set of multiscale texture trees.
	*/
	const std::vector<HSLICNode>& get_superpixel_trees();

};

#endif