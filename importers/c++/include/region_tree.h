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
#ifndef REGION_TREE_H
#define REGION_TREE_H

#include <vector>

/*
Describes a region in the image. In the case of superpixels that region is a 
continuous region. In the case of labels it can be any region.

The region is defined by a list of atomic suerpixels that's union composes
the region. 
*/
struct Region
{

	/*
	A list of super pixels that belong to our texture
	*/
	std::vector<int> atomic_superpixels;

	/*The scale of the node (negitive if not avalable)*/
	float scale;
};


/*
A Node in a tree of regions. This can be superpixels in HSLIC or 
labels for labelmaps

Each node contians a list of sp that are part of that regions,
as well as a list of lower scale regions
*/
struct RegionNode
{
	/*
	A list of lower scale textures that refine this texture.
	The superpixels in children nodes are all represented in
	this texture
	*/
	std::vector<RegionNode> children_node;

	/*
	The actual region data. 
	*/
	Region region;

};

//////////////////////////////////Internal Use//////////////////////////////////////////////////////

struct matvar_t;


/*
Loads a region (Superpixel or label) from a struct_array. This is the fomat 
used to store a region when it is not part of a tree. 
*/
void load_region(matvar_t* struct_array, Region& out);

/*
Loads a region from the tree format used in our file.
*/
void load_region_node_from_cell_array(matvar_t* cell_array, std::vector<RegionNode>& output);


#endif