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

This is a simple program to demonstrate loading feature vectors for HSLIC data.
Note that you must use the --output_features option when running multiscale_extraction.py 
to include the feature vectors in the output file. 
Usage: 
      HSLIC_features_demo input_file.mat image_index.csv features.csv scale
Where:
      input_file.mat is a label-map file
      image_index.csv Output file indicating which superpixel each pixel belongs too. 
	  features.csv Output file. The feature vector for each index. 
			Each row contains a different feature vector, in order of the indexes. 
      scale is the scale in pixels of the superpixels.

*/

#include "HSLIC.h"
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;


int main(int argc, char* argv[])
{
	//Print somthing of a useful message if we get an exception
	try 
	{
		//If we don't have enough args, give the usage infomation 
		if (argc < 3)
		{
			cout << "Usage: " << argv[0] << " input_file.mat image_index.csv features.csv scale" << endl
				<< "Will output two files that describe the texture variations for a single" << endl
				<< "selected scale. The first, image_index.csv, will give the index of each" << endl
				<< "superpixel at each location on the image. The second, features.csv, will" << endl
				<< "give the features for those superpixels. " << endl;
			return 1;
		}

		//Load the values from the arguments
		string filename_in = argv[1];
		string image_index_file = argv[2];
		string features_file = argv[3];
		float scale = stof(argv[4]);

		//Load the HSLIC file
		//Create a map to a blank image
		HierarchicalRegionMap<size_t> hslic = load_HLIC<size_t>(filename_in);
		CompoundRegionMap<size_t> single_scale = hslic.get_single_scale_map(scale);

		//Store the superpixel each pixel belongs to in the array
		for (const CompoundRegionMap<size_t>::RegionKey& key : single_scale)
			single_scale[key] = single_scale.index_from_key(key);

		//Output the indexes to the first file
		ofstream image_index(image_index_file);

		for (int r = 0; r < single_scale.get_rows(); r++)
		{
			for (int c = 0; c < single_scale.get_cols(); c++)
			{
				image_index << ((valarray<size_t>)single_scale.value_at(r, c))[0];
				if (c < single_scale.get_cols())
					image_index << ", ";
			}
			image_index << endl;

		}

		//Output the features to the second file. 
		ofstream features(features_file);

		for (size_t idx = 0; idx < single_scale.size(); idx++)
		{
			const CompoundRegionMap<size_t>::RegionKey& key = single_scale.key_from_index(idx);

			if (!single_scale.has_extra_data("feature", key))
			{
				cout << "Not all features stored in " << filename_in << endl;
				cout << "Did you remember to use --output_features when running multiscale_extraction.py?" << endl;
				features << "<Not in file>";
			}
			else
			{
				ExtraData ed = single_scale.get_extra_data("feature", key);

				for (size_t data_id = 0; data_id < ed.rows; data_id++)
				{
					features << ed.data[data_id];
					if (data_id < ed.rows)
						features << ", ";
				}
			}

			features << endl;
		}

	}
	catch (exception e)
	{
		cerr << "Error encounted: " << e.what() << endl;
	}
	catch (...)
	{
		cerr << "Error encounted!";
	}

}