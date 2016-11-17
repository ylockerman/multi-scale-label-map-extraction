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
#include <memory>
#include <valarray>
#include <utility>
#include <algorithm>
#include <map>
#include <queue>

typedef std::valarray<float> ExtraData;
typedef std::map<std::string, ExtraData > ExtraDataMap;

template <typename ImageData_t, typename RegionKey_t, 
	typename AtomicRegionMap_t, typename AtomicRegionKey_t,
			typename const_iterator_t>
class AbstractRegionMap
{
public:
	typedef typename ImageData_t ImageData;
	typedef typename RegionKey_t RegionKey;
	typedef typename AbstractRegionMap AbstractRegionMap_type;
	typedef typename AtomicRegionMap_t AtomicRegionMap_type;
	typedef typename AtomicRegionKey_t AtomicRegionKey;
	typedef typename const_iterator_t const_iterator;

	/*Return a tile based on the key*/
	virtual std::indirect_array<ImageData> operator[](const RegionKey& key) = 0;

	/*Return a tile based on the key*/
	virtual const std::indirect_array<ImageData> operator[](const RegionKey& key) const = 0;

	/*Return a tile based on the key*/
	virtual std::indirect_array<ImageData> operator[](const std::vector<RegionKey>& keylist) = 0;

	/*Return a tile based on the key*/
	virtual const std::indirect_array<ImageData> operator[](const std::vector<RegionKey>& keylist) const = 0;

	/*Return the number of regions*/
	virtual size_t size() const = 0;

	/*Iterator pointing to the first key */
	virtual const_iterator begin() const  = 0;

	/*Iterator pointing to the last key*/
	virtual const_iterator end() const  = 0;

	/*Retrun a index from 0 to len for a key*/
	virtual size_t index_from_key(const RegionKey& key) const = 0;

	/*Return key number index*/
	virtual const RegionKey& key_from_index(size_t index) const = 0;

	/*Returns a new tile map with the same properyes for a diffrent image*/
	virtual std::shared_ptr<AbstractRegionMap> copy_map_for_image(std::shared_ptr<std::valarray<ImageData>> data, size_t stride = 0) const = 0;

	/*Returns a new tile map with the same properyes for a diffrent image*/
	virtual const std::shared_ptr<AbstractRegionMap> copy_map_for_image(const std::valarray<ImageData>& data, size_t stride = 0) const = 0;

	/*Returns a new tile map with the same properyes for a new image*/
	virtual const std::shared_ptr<AbstractRegionMap> copy_map_for_new_image(size_t stride = 0) const = 0;

	/*Returns the keyset*/
	virtual const std::vector<RegionKey>& get_key_set() = 0;

	/*Returns the atomic map that this map is based on.*/
	virtual AtomicRegionMap_type& get_atomic_map() = 0;

	/*Returns the atomic map that this map is based on.*/
	virtual const AtomicRegionMap_type& get_atomic_map() const = 0;

	/*        
		Returns the a list of the atomic keys that make up a  key of this map.
        
        This satisfies
        map.get_atomic_map()[map.get_atomic_keys(key)] == map[key]
	*/
	virtual std::vector<AtomicRegionKey> get_atomic_keys(const RegionKey& key) const = 0;


	/*
	Returns the number of rows in the image.
	*/
	virtual size_t get_rows() const = 0;

	/*
	Returns the number of cols in the image.
	*/
	virtual size_t get_cols() const = 0;


	/*
	Returns a reference to the to the image at a particular location.
	*/
	virtual std::slice_array<ImageData> value_at(int row, int col) = 0;

	/*
	Returns a reference to the to the image at a particular location.
	*/
	virtual const std::slice_array<ImageData> value_at(int row, int col) const = 0;

	/*
	Checks to see if the map has extra data given by the name name.
	*/
	virtual bool has_extra_data(std::string name) const = 0;

	/*
	Returns the extra data of a map (if it is available). Otherwise, throws an exception.
	*/
	virtual const ExtraData& get_extra_data(std::string name) const = 0;

	/*
	Checks to see if a given key has extra data given by the name name. 
	*/
	virtual bool has_extra_data(std::string name, const RegionKey& key) const = 0;

	/*
	Returns the extra data of a key (if it is available). Otherwise, throws an exception.
	*/
	virtual const ExtraData& get_extra_data(std::string name, const RegionKey& key) const = 0;
};

template <typename T>
void _concat(std::vector <std::valarray<T> > arrs, std::valarray<T>& out)
{
	size_t size = 0;

	for (const std::valarray<T>& arr : arrs)
		size += arr.size();

	out = std::valarray<T>(size);
	size_t out_pos = 0;

	for (const std::valarray<T>& arr : arrs)
	{
		out[std::slice(out_pos, arr.size())] = arr;
		out_pos += arr.size();
	}
}

template <typename S,typename T>
void _concat(std::vector <std::pair<S,std::valarray<T> > > arrs, std::valarray<T>& out)
{
	size_t size = 0;

	for (const std::pair<S, std::valarray<T> >& arr : arrs)
		size += arr.second.size();

	out = std::valarray<T>(size);
	size_t out_pos = 0;

	for (const std::pair<S, std::valarray<T> >& arr : arrs)
	{
		out[std::slice(out_pos, arr.second.size(), 1)] = arr.second;
		out_pos += arr.second.size();
	}
}

/*
An AtomicRegionMap resents a region map where all regions are atomic,
that is cannot be divided into sub regions. It is guaranteed that each
pixel belongs to one and only one region. 
*/
template <typename ImageData_t>
class AtomicRegionMap : 
			public AbstractRegionMap<ImageData_t, //ImageData_t
									 std::pair<size_t,std::valarray<size_t>>,  //RegionKey_t
									 AtomicRegionMap<ImageData_t>, //AtomicRegionMap_t
									 std::pair<size_t, std::valarray<size_t>>, //AtomicRegionKey_t
									 std::vector< std::pair<size_t, std::valarray<size_t> > >::const_iterator> //const_iterator
{
protected:
	/*The image data we map to*/
	std::shared_ptr< std::valarray<ImageData> > image;

	/*The size of each element in the image. */
	size_t stride;

	/*The size of the image (if 2d)*/
	size_t rows, cols;

	/*Which region each image element belongs to*/
	std::valarray<size_t> cluster_index;

	/*A lookup table for each region.*/
	std::vector< RegionKey > key_set;

	/*Extra data for the map*/
	ExtraDataMap extra_data;

	/*An array of extra data. Stores the data for each key index*/
	std::vector< ExtraDataMap > region_extra_data;

public:
	typedef typename ImageData ImageData;
	typedef typename RegionKey RegionKey;
	typedef typename AtomicRegionMap_type AtomicRegionMap_type;
	typedef typename AtomicRegionKey AtomicRegionKey;
	typedef typename const_iterator const_iterator;

	AtomicRegionMap(std::shared_ptr< std::valarray<ImageData> > image, size_t rows, size_t cols, size_t stride, 
						const std::valarray<size_t>& cluster_index, 
						const ExtraDataMap& edm = ExtraDataMap(),
						const std::vector< ExtraDataMap >& redm = std::vector< ExtraDataMap >() )
		: image(image), rows(rows), cols(cols), stride(stride), cluster_index(cluster_index), extra_data(edm), region_extra_data(redm)
	{
		size_t number_of_regions = cluster_index.max() + 1;
		key_set.resize(number_of_regions);

		std::vector< std::vector< size_t> > key_vector_table(number_of_regions);

		//Calculate a lookup table for each region. This allows us to access them in constant time. 
		for (size_t indx = 0; indx < cluster_index.size(); indx++)
		{
			//Each key needs to access all the elements.
			//This differs from python
			for (int offset = 0; offset < stride; offset++)
				key_vector_table[cluster_index[indx]].push_back(stride*indx+offset);
		}


		for (size_t indx = 0; indx < number_of_regions; indx++)
		{
			key_set[indx] = std::make_pair(indx, std::valarray<size_t>(key_vector_table[indx].size()) );
			std::copy(key_vector_table[indx].begin(), key_vector_table[indx].end(), std::begin(key_set[indx].second) );
		}
	}

	AtomicRegionMap(const std::valarray<ImageData>& image, size_t rows, size_t cols, size_t stride, 
						const std::valarray<size_t>& cluster_index, 
						const ExtraDataMap& edm = ExtraDataMap(),
						const std::vector< ExtraDataMap >& redm = std::vector< ExtraDataMap >())
		: AtomicRegionMap(std::shared_ptr< valarray<ImageData> >(new valarray<ImageData>(image)), rows,  cols, stride, cluster_index, edm, redm)
	{}

	AtomicRegionMap(size_t rows, size_t cols, size_t stride, const std::valarray<size_t>& cluster_index,
					  const ExtraDataMap& edm = ExtraDataMap(), const std::vector< ExtraDataMap >& redm = std::vector< ExtraDataMap >())
		: AtomicRegionMap(std::valarray<ImageData>(rows*cols*stride), rows, cols, stride,cluster_index, edm, redm)
	{}

	/*Return a tile based on the key*/
	std::indirect_array<ImageData> operator[](const RegionKey& key)
	{
		return (*image)[key.second];
	}

	/*Return a tile based on the key*/
	const std::indirect_array<ImageData> operator[](const RegionKey& key) const
	{
		return (*image)[key.second];
	}

	/*Return a tile based on the key*/
	std::indirect_array<ImageData> operator[](const std::vector<RegionKey>& keylist)
	{
		std::valarray<size_t> indexlist;
		_concat(keylist, indexlist);
		return (*image)[indexlist];
	}

	/*Return a tile based on the key*/
	const std::indirect_array<ImageData> operator[](const std::vector<RegionKey>& keylist) const
	{
		std::valarray<size_t> indexlist;
		_concat(keylist, indexlist);
		return (*image)[indexlist];
	}

	/*Return the number of regions*/
	virtual size_t size() const
	{
		return key_set.size();
	}

	/*Iterator pointing to the first key */
	const_iterator begin() const 
	{
		return key_set.begin();
	}

	/*Iterator pointing to the last key*/
	const_iterator end() const
	{
		return key_set.end();
	}

	/*Retrun a index from 0 to len for a key*/
	size_t index_from_key(const RegionKey& key) const
	{
		return key.first;
	}

	/*Return key number index*/
	const RegionKey& key_from_index(size_t index) const
	{
		return key_set[index];
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	template <typename ImageData_t>
	std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap, AtomicRegionKey, const_iterator> > 
			copy_map_for_image(std::shared_ptr<std::valarray<ImageData>> data, size_t stride = 0) const
	{
		if (stride == 0) stride = this->stride;

		return std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap, AtomicRegionKey, const_iterator> >
															(new AtomicRegionMap<ImageData_t>(data, rows, cols, stride, cluster_index));
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	template <typename ImageData_t>
	const std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap, AtomicRegionKey, const_iterator> > 
		copy_map_for_image(const std::valarray<ImageData>& data, size_t stride = 0) const
	{
		if (stride == 0) stride = this->stride;

		return std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap, AtomicRegionKey, const_iterator> >
																(new AtomicRegionMap<ImageData_t>(data, rows, cols, stride, cluster_index));
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	std::shared_ptr<AbstractRegionMap> copy_map_for_image(std::shared_ptr<std::valarray<ImageData>> data, size_t stride = 0) const
	{
		return copy_map_for_image<ImageData>(data, stride);
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	const std::shared_ptr<AbstractRegionMap> copy_map_for_image(const std::valarray<ImageData>& data, size_t stride = 0) const
	{
		return copy_map_for_image<ImageData>(data, stride);
	}

	/*Returns a new tile map with the same properyes for a new image*/
	template <typename ImageData_t>
	const std::shared_ptr<AbstractRegionMap> copy_map_for_new_image(size_t stride = 0) const
	{
		if (stride == 0) stride = this->stride;
		return copy_map_for_image<ImageData_t>(std::valarray<ImageData_t>(rows*cols*stride), stride);
	}

	/*Returns a new tile map with the same properyes for a new image*/
	const std::shared_ptr<AbstractRegionMap> copy_map_for_new_image(size_t stride = 0) const
	{
		if (stride == 0) stride = this->stride;
		return copy_map_for_image<ImageData>(std::valarray<ImageData>(rows*cols*stride), stride);
	}

	/*Returns the keyset*/
	const std::vector<RegionKey>& get_key_set()
	{
		return key_set;
	}

	/*Returns the atomic map that this map is based on.*/
	AtomicRegionMap_type& get_atomic_map()
	{
		return *this;
	}

	/*Returns the atomic map that this map is based on.*/
	const AtomicRegionMap_type& get_atomic_map() const
	{
		return *this;
	}

	/*
	Returns the a list of the atomic keys that make up a  key of this map.

	This satisfies
	map.get_atomic_map()[map.get_atomic_keys(key)] == map[key]
	*/
	std::vector<AtomicRegionKey> get_atomic_keys(const RegionKey& key) const
	{
		return std::vector < AtomicRegionKey >(1, key);
	}

	/*
		Returns the number of rows in the image. 
	*/
	size_t get_rows() const
	{
		return rows;
	}

	/*
		Returns the number of cols in the image. 
	*/
	size_t get_cols() const
	{
		return cols;
	}

	/*
	Returns A reference to the to the image at a particular location.  
	*/
	std::slice_array<ImageData> value_at(int row, int col)
	{
		return (*image)[std::slice(stride*(col + row*cols), stride, 1)];
	}

	/*
	Returns A reference to the to the image at a particular location.
	*/
	const std::slice_array<ImageData> value_at(int row, int col) const
	{
		return (*image)[std::slice(stride*(col + row*cols), stride, 1)];
	}

	/*
	Checks to see if the map has extra data given by the name name.
	*/
	bool has_extra_data(std::string name) const
	{
			return extra_data.find(name) != extra_data.end();
	}

	/*
	Returns the extra data of a map (if it is available). Otherwise, throws an exception.
	*/
	const ExtraData& get_extra_data(std::string name) const
	{
		ExtraDataMap::const_iterator it = extra_data.find(name);

		if (it == extra_data.end())
			throw std::exception(("Map does not contain extra data: " + name).c_str());

		return it->second;
	}

	/*
	Checks to see if a given key has extra data given by the name name.
	*/
	bool has_extra_data(std::string name, const RegionKey& key) const
	{
		return key.first < region_extra_data.size() &&
			region_extra_data[key.first].find(name) != region_extra_data[key.first].end();
	}

	/*
	Returns the extra data (if it is available). Otherwise, throws an exception.
	*/
	const ExtraData& get_extra_data(std::string name, const RegionKey& key) const
	{
		if (key.first > region_extra_data.size())
			throw std::exception("No extra data for key");

		ExtraDataMap::const_iterator it= region_extra_data[key.first].find(name);

		if (it == region_extra_data[key.first].end())
			throw std::exception(("Key does not contain extra data: " + name).c_str());

		return it->second;
	}

};


/*
Describes a region in the image. In the case of superpixels that region is a 
continuous region. In the case of labels it can be any region.

The region is defined by a list of atomic suerpixels that's union composes
the region. 
*/
struct CompoundRegion
{

	/*
	A list of super pixels that belong to our texture
	*/
	std::vector<size_t> atomic_superpixels;

	/*
	Any extra data we may have about this region.
	*/
	ExtraDataMap extra_data;
};

typedef std::shared_ptr<CompoundRegion> CompoundRegionPtr;

/*
A CompoundRegionMap consist of regions that are built by the union of
atomic regions.
*/
template <typename ImageData_t>
class CompoundRegionMap :
	public AbstractRegionMap<ImageData_t, //ImageData_t
	std::pair<size_t, CompoundRegionPtr >,  //RegionKey_t
	AtomicRegionMap<ImageData_t>, //AtomicRegionMap_t
	typename AtomicRegionMap<ImageData_t>::AtomicRegionKey,//AtomicRegionKey_t
	typename std::vector< std::pair<size_t, CompoundRegionPtr > >::const_iterator> //const_iterator
{
protected:
	AtomicRegionMap_type atomic_region_map;

	std::vector< RegionKey > key_set;

	/*Extra data for the map*/
	ExtraDataMap extra_data;

	CompoundRegionMap(AtomicRegionMap_type arm, const std::vector< RegionKey >& keys,
						const ExtraDataMap& edm = ExtraDataMap() )
		: atomic_region_map(arm), key_set(keys.begin(), keys.end()), extra_data(edm)
	{}




public:
	typedef typename ImageData ImageData;
	typedef typename RegionKey RegionKey;
	typedef typename AtomicRegionMap_type AtomicRegionMap_type;
	typedef typename AtomicRegionKey AtomicRegionKey;
	typedef typename const_iterator const_iterator;

	CompoundRegionMap(AtomicRegionMap_type arm, const std::vector< CompoundRegion >& regions,
		const ExtraDataMap& edm = ExtraDataMap() )
		: atomic_region_map(arm), key_set(regions.size()), extra_data(edm)
	{
		//Calculate a lookup table for each region. This allows us to access them in constant time. 
		for (size_t indx = 0; indx < regions.size(); indx++)
		{
			key_set[indx] = std::make_pair(indx, CompoundRegionPtr(new CompoundRegion(regions[indx])));
		}
	}

	CompoundRegionMap(AtomicRegionMap_type arm, const std::vector< CompoundRegionPtr >& regions,
		const ExtraDataMap& edm = ExtraDataMap() )
		: atomic_region_map(arm), key_set(regions.size()), extra_data(edm)
	{
		//Calculate a lookup table for each region. This allows us to access them in constant time. 
		for (size_t indx = 0; indx < regions.size(); indx++)
		{
			key_set[indx] = std::make_pair(indx, CompoundRegionPtr(new CompoundRegion( *regions[indx] ) ) );
		}
	}

	/*Return a tile based on the key*/
	std::indirect_array<ImageData> operator[](const RegionKey& key)
	{
		return atomic_region_map[get_atomic_keys(key)];
	}

	/*Return a tile based on the key*/
	const std::indirect_array<ImageData> operator[](const RegionKey& key) const
	{
		return atomic_region_map[get_atomic_keys(key)];
	}

	/*Return a tile based on the key*/
	std::indirect_array<ImageData> operator[](const std::vector<RegionKey>& keylist)
	{
		std::vector<AtomicRegionKey> all_keys;

		for (const RegionKey& key : keylist)
		{
			const std::vector<AtomicRegionKey>& attkey = get_atomic_keys(key);
			all_keys.insert(all_keys.end(), attkey.begin(), attkey.end());
		}
			
		return atomic_region_map[all_keys];
	}

	/*Return a tile based on the key*/
	const std::indirect_array<ImageData> operator[](const std::vector<RegionKey>& keylist) const
	{
		std::vector<AtomicRegionKey> all_keys;

		for (const RegionKey& key : keylist)
		{
			const std::vector<AtomicRegionKey>& attkey = get_atomic_keys(key);
			all_keys.insert(all_keys.end(), attkey.begin(), attkey.end());
		}

		return atomic_region_map[all_keys];
	}

	/*Return the number of regions*/
	virtual size_t size() const
	{
		return key_set.size();
	}

	/*Iterator pointing to the first key */
	const_iterator begin() const
	{
		return key_set.begin();
	}

	/*Iterator pointing to the last key*/
	 const_iterator end() const
	{
		 return key_set.end();
	}

	/*Retrun a index from 0 to len for a key*/
	virtual size_t index_from_key(const RegionKey& key) const
	{
		return key.first;
	}

	/*Return key number index*/
	const RegionKey& key_from_index(size_t index) const
	{
		return key_set[index];
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	template <typename ImageData_t>
	std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
								copy_map_for_image(std::shared_ptr<std::valarray<ImageData>> data, size_t stride = 0) const
	{
		return std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
			(new CompoundRegionMap<ImageData_t>(atomic_region_map.copy_map_for_image<ImageData_t>(data, stride)->get_atomic_map(), key_set));
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	template <typename ImageData_t>
	const std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
													copy_map_for_image(const std::valarray<ImageData>& data, size_t stride = 0) const
	{
		return std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
			(new CompoundRegionMap<ImageData_t>(atomic_region_map.copy_map_for_image<ImageData_t>(data, stride)->get_atomic_map(), key_set));
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	std::shared_ptr<AbstractRegionMap_type > copy_map_for_image(std::shared_ptr<std::valarray<ImageData>> data, size_t stride = 0) const
	{
		return copy_map_for_image<ImageData>(data, stride);
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	const std::shared_ptr<AbstractRegionMap_type > copy_map_for_image(const std::valarray<ImageData>& data, size_t stride = 0) const
	{
		return copy_map_for_image<ImageData>(data, stride);
	}

	/*Returns a new tile map with the same properyes for a new image*/
	template <typename ImageData_t>
	const std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> > copy_map_for_new_image(size_t stride = 0) const
	{
		return std::shared_ptr<AbstractRegionMap<typename ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
			(new CompoundRegionMap<typename ImageData_t>(atomic_region_map.copy_map_for_new_image<typename ImageData_t>(stride)->get_atomic_map(), key_set));
	}

	/*Returns a new tile map with the same properyes for a new image*/
	const std::shared_ptr<AbstractRegionMap> copy_map_for_new_image(size_t stride = 0) const
	{
		return this->copy_map_for_new_image<ImageData>(stride);
	}

	/*Returns the keyset*/
	virtual const std::vector<RegionKey>& get_key_set()
	{
		return key_set;
	}

	/*Returns the atomic map that this map is based on.*/
	virtual AtomicRegionMap_type& get_atomic_map()
	{
		return atomic_region_map;
	}

	/*Returns the atomic map that this map is based on.*/
	virtual const AtomicRegionMap_type& get_atomic_map() const
	{
		return atomic_region_map;
	}

	/*
	Returns the a list of the atomic keys that make up a  key of this map.

	This satisfies
	map.get_atomic_map()[map.get_atomic_keys(key)] == map[key]
	*/
	virtual std::vector<AtomicRegionKey> get_atomic_keys(const RegionKey& key) const
	{
		std::vector<AtomicRegionKey> att_key;
		for (size_t att_id : key.second->atomic_superpixels)
			att_key.push_back(atomic_region_map.key_from_index(att_id));

		return att_key;
	}


	/*
	Returns the number of rows in the image.
	*/
	size_t get_rows() const
	{
		return atomic_region_map.get_rows();
	}

	/*
	Returns the number of cols in the image.
	*/
	size_t get_cols() const
	{
		return atomic_region_map.get_cols();
	}


	/*
	Returns a reference to the to the image at a particular location.
	*/
	std::slice_array<ImageData> value_at(int row, int col)
	{
		return atomic_region_map.value_at(row, col);
	}

	/*
	Returns a reference to the to the image at a particular location.
	*/
	const std::slice_array<ImageData> value_at(int row, int col) const
	{
		return atomic_region_map.value_at(row, col);
	}

	/*
	Checks to see if the map has extra data given by the name name.
	*/
	bool has_extra_data(std::string name) const
	{
		return extra_data.find(name) != extra_data.end();
	}

	/*
	Returns the extra data of a map (if it is available). Otherwise, throws an exception.
	*/
	const ExtraData& get_extra_data(std::string name) const
	{
		ExtraDataMap::const_iterator it = extra_data.find(name);

		if (it == extra_data.end())
			throw std::exception(("Map does not contain extra data: " + name).c_str());

		return it->second;
	}

	/*
	Checks to see if a given key has extra data given by the name name.
	*/
	bool has_extra_data(std::string name, const RegionKey& key) const
	{
		return key.second->extra_data.find(name) != key.second->extra_data.end();
	}

	/*
	Returns the extra data (if it is available). Otherwise, throws an exception.
	*/
	const ExtraData& get_extra_data(std::string name, const RegionKey& key) const
	{
		ExtraDataMap::const_iterator it = key.second->extra_data.find(name);

		if (it == key.second->extra_data.end())
			throw std::exception(("Key does not contain extra data: " + name).c_str());

			return it->second;
	}


	template <typename ImageData_t> friend  class HierarchicalRegionMap;
};

/*
A Node in a tree of regions. This can be superpixels in HSLIC or 
labels for labelmaps

Each node contians a list of sp that are part of that regions,
as well as a list of lower scale regions
*/
struct HierarchicalRegion : public CompoundRegion
{
	/*
	A list of lower scale textures that refine this texture.
	The superpixels in children nodes are all represented in
	this texture
	*/
	std::vector< std::shared_ptr<HierarchicalRegion> > children_node;

	/*The scale of the node (negitive if not avalable)*/
	float scale;

public:
	HierarchicalRegion()
	{}

	HierarchicalRegion(const HierarchicalRegion& other)
		: CompoundRegion(other), children_node(other.children_node.size())
	{
		scale = other.scale;

		for (size_t indx = 0; indx < other.children_node.size(); indx++)
			children_node[indx] = std::shared_ptr<HierarchicalRegion>(new HierarchicalRegion(*other.children_node[indx]));
	}


	HierarchicalRegion & operator= (const HierarchicalRegion & other)
	{
		CompoundRegion::operator=(other);

		scale = other.scale;

		children_node.resize(other.children_node.size());
		for (size_t indx = 0; indx < other.children_node.size(); indx++)
			children_node[indx] = std::shared_ptr<HierarchicalRegion>(new HierarchicalRegion(*other.children_node[indx]));

		return *this;
	}
};

typedef std::shared_ptr<HierarchicalRegion> HierarchicalRegionPtr;

typedef float  scale_type;

/*
A HierarchicalRegionMap is a forest of regions across different scales.
Regions on larger scales are the union of regions on smaller scales.
*/
template <typename ImageData_t>
class HierarchicalRegionMap : public CompoundRegionMap<ImageData_t>
{
	std::vector<HierarchicalRegionPtr> root_table;
	std::map<scale_type, std::vector<HierarchicalRegionPtr> > regions_at_scale;
	std::vector<scale_type> scales_sorted;


	/*
	Helper function for superpixels_at_scale. Will return the largest superpixels that are a scall less then "scale."
	*/
	void _get_nodes_less_then_scale(const std::vector<HierarchicalRegionPtr>& tree_list,
										scale_type scale, std::vector<CompoundRegionPtr>& out)
	{
		for (HierarchicalRegionPtr node : tree_list)
		{
			if (node->scale <= scale || node->children_node.size() == 0)
				out.push_back(std::dynamic_pointer_cast<CompoundRegion>(node));
			else
				_get_nodes_less_then_scale(node->children_node, scale, out);
		}
	}

public:
	typedef typename ImageData ImageData;
	typedef typename RegionKey RegionKey;
	typedef typename AtomicRegionMap_type AtomicRegionMap_type;
	typedef typename AtomicRegionMap_type::RegionKey AtomicRegionKey;
	typedef typename const_iterator const_iterator;


	HierarchicalRegionMap(AtomicRegionMap_type arm, const std::vector< HierarchicalRegionPtr >& root_table,
								const ExtraDataMap& edm = ExtraDataMap())
		:CompoundRegionMap(arm, std::vector< CompoundRegion >(), edm), root_table(root_table.size())
	{
		std::queue<HierarchicalRegionPtr> too_look_at_queue;


		for (size_t indx = 0; indx < root_table.size(); indx++)
		{
			this->root_table[indx] = HierarchicalRegionPtr(new HierarchicalRegion(*root_table[indx]));
			too_look_at_queue.push(this->root_table[indx]);
		}

		int indx = 0;
		while (!too_look_at_queue.empty())
		{
			HierarchicalRegionPtr top = too_look_at_queue.front(); too_look_at_queue.pop();
			key_set.push_back(std::make_pair(indx++, top));
			regions_at_scale[top->scale].push_back(top);

			for (size_t indx = 0; indx < top->children_node.size(); indx++)
				too_look_at_queue.push(top->children_node[indx]);
		}

		//http://stackoverflow.com/questions/771453/copy-map-values-to-vector-in-stl
		for (std::map<scale_type, std::vector<HierarchicalRegionPtr> >::iterator it = regions_at_scale.begin(); 
																				it != regions_at_scale.end(); ++it) {
			scales_sorted.push_back(it->first);
		}

		std::sort(scales_sorted.begin(), scales_sorted.end());
	}


	const std::vector<scale_type> get_scales()
	{
		return scales_sorted;
	}


	/*Returns a new tile map with the same properyes for a diffrent image*/
	template <typename ImageData_t>
	std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
								copy_map_for_image(std::shared_ptr<std::valarray<ImageData>> data, size_t stride = 0) const
	{
		AtomicRegionMap_type am = atomic_region_map.copy_map_for_image<ImageData_t>(data, stride)->get_atomic_map();
		return std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
																				(new HierarchicalRegionMap<ImageData_t>(am, root_table));
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	template <typename ImageData_t>
	const std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
											copy_map_for_image(const std::valarray<ImageData>& data, size_t stride = 0) const
	{
		AtomicRegionMap_type am = atomic_region_map.copy_map_for_image<ImageData_t>(data, stride)->get_atomic_map();
		return std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> > 
																				(new HierarchicalRegionMap<ImageData_t>(am, root_table));
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	std::shared_ptr<AbstractRegionMap> copy_map_for_image(std::shared_ptr<std::valarray<ImageData>> data, size_t stride = 0) const
	{
		return copy_map_for_image<ImageData>(data, stride);
	}

	/*Returns a new tile map with the same properyes for a diffrent image*/
	const std::shared_ptr<AbstractRegionMap> copy_map_for_image(const std::valarray<ImageData>& data, size_t stride = 0) const
	{
		return copy_map_for_image<ImageData>(data, stride);
	}

	/*Returns a new tile map with the same properyes for a new image*/
	template <typename ImageData_t>
	const std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> > copy_map_for_new_image(size_t stride = 0) const
	{
		AtomicRegionMap_type am = atomic_region_map.copy_map_for_new_image<ImageData_t>(stride)->get_atomic_map();
		return std::shared_ptr<AbstractRegionMap<ImageData_t, RegionKey, AtomicRegionMap_type, AtomicRegionKey, const_iterator> >
			(new HierarchicalRegionMap<ImageData_t>(am, root_table));
	}

	/*Returns a new tile map with the same properyes for a new image*/
	const std::shared_ptr<AbstractRegionMap > copy_map_for_new_image(size_t stride = 0) const
	{
		return copy_map_for_new_image<ImageData>(stride);
	}

	CompoundRegionMap<ImageData_t> get_single_scale_map(scale_type scale)
	{
		std::vector<CompoundRegionPtr> single_scale_regions;
		_get_nodes_less_then_scale(root_table, scale, single_scale_regions);

		return CompoundRegionMap<ImageData_t>(atomic_region_map, single_scale_regions);
	}

};



#endif