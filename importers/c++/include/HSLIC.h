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
#ifndef HSLIC_H
#define HSLIC_H

#include <vector>
#include <string>
#include <exception>
#include "region_tree.h"

typedef RegionNode HSLICNode;
typedef Region HSLICSuperpixel;

/*
This class stores a tree of HSLIC data.
*/
class HSLIC
{
	//The number of rows in the image
	int rows;

	//The number of coloumns in the image
	int cols;

	//A row major order array of size (rows*cols) 
	//that indicates which atomic SLIC superpixel 
	//each pixel in the image belongs to.
	std::vector<int> slic_indexes;

	//A table that indicates which indexes in slic_indexes
	//represents each superpixels. 
	//
	//That is i is in atomic_lookup_table[j] if and only if
	//		slic_indexes[i] == j

	std::vector< std::vector<int> > atomic_lookup_table;

	//The roots of the the HSLIC trees
	std::vector<HSLICNode> tree_root;


public:

	/*
	Lodes the multiscale texture set from a file
	*/
	HSLIC(const std::string& file) throw(std::exception);

	/*
	Retruns the number of rows in the image
	*/
	int get_rows() const;

	/*
	Retruns the number of columns in the image
	*/
	int get_cols() const;

	/*
	Returns the number of atomic superpixels
	*/
	int get_atomic_superpixels_count() const;

	/*
	Returns a vector of HSLICSuperpixel containing all
	the superpixels at a given scale
	*/
	std::vector<HSLICSuperpixel> superpixels_at_scale(float scale) const;

	/*
	Returns a row by col array which states what superpixel each pixel in the image belongs to.
	This is in Row major order.
	*/
	const std::vector<int>& get_atomic_superpixel_indicator();

	/*
	Builds an indicator for an inputted list of superpixels.
	Each pixel in the indicator will give the index of the the input array where
	that pixel belongs to. The result is undefined if a pixel belongs to multiple 
	superpixels. 

	This is an indicator array in row major order.
	*/
	std::vector<int> HSLIC::superpixels_indicator(const std::vector<HSLICSuperpixel>& superpixels) const;

	/*
	Returns the set of multiscale texture trees.
	*/
	const std::vector<HSLICNode>& get_superpixel_trees();

};

#endif