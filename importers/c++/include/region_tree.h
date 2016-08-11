#pragma once
#ifndef REGION_TREE_H
#define REGION_TREE_H

#include <vector>


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
	A list of super pixels that belong to our texture
	*/
	std::vector<int> slic_pixels;

	/*The scale of the node (negitive if not avalable)*/
	float scale;
};




#endif