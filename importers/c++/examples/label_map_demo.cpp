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

This is a simple program to demonstrate loading of label map data. Given a label-map file and a scale
it will output a ppm file with each superpixel given a random color.

Usage:
label_map_demo input_file.mat output_file_%f.ppm [--tree]
Where:
input_file.mat is a label-map file
output_file_%f.ppm is the location to store the resulting image. 
%f will be replaced with the scale of each image.
If --tree is included the map will form a hierarchal tree, otherwise the
labels at each level will be independent.
*/

#include "label_map.h"
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

#include "example_utils.h"

int main(int argc, char* argv[])
{
	//Print somthing of a useful message if we get an exception
	try
	{
		//If we don't have enough args, give the usage infomation 
		if (argc < 3 || argc > 4)
		{
			cout << "Usage: " << argv[0] << " input_file.mat output_file_%f.ppm [--tree]" << endl
				<< "Will output a ppm file with the label map at each scale." << endl
				<< "If --tree is included the map will form a hierarchal tree, otherwise the" << endl
				<< "labels at each level will be independent." << endl
				<< "The color will be chosen to make the image a little easier to read. However," << endl
				<< "note that the colors are arbitrary and useful for visualization only. " << endl;
			return 1;
		}

		//Load the values from the arguments
		string filename_in = argv[1];
		string filename_out = argv[2];

		size_t index_of_template = filename_out.find("%f");
		if (index_of_template == string::npos) {
			cout << "The output file must include the location template %f" << endl;
		}

		bool use_tree = false;
		if (argc == 4)
		{
			if (string(argv[3]) == "--tree")
				use_tree = true;
			else
				cout << "Unknown option "<< argv[3]  << endl;
		}


		if (use_tree)
		{
			//Load the HSLIC file
			//Create a map to a blank image
			HierarchicalRegionMap<color> hlabels = load_hierarchical_label_map<color>(filename_in, 3);

			std::vector<float> all_scales = hlabels.get_scales();

			for (float scale : all_scales)
			{
				cout << "Saving scale " << scale << endl;
				output_PPM(filename_out.replace(index_of_template, 2, to_string(scale)), hlabels.get_single_scale_map(scale));
			}
		}
		else
		{
			std::map<float, std::shared_ptr< CompoundRegionMap<color> > > label_stack =  load_label_map_stack<color>(filename_in, 3);

			for (const std::pair<float, std::shared_ptr< CompoundRegionMap<color> > >& elm : label_stack)
			{
				cout << "Saving scale:" << elm.first << endl;
				output_PPM(filename_out.replace(index_of_template, 2, to_string(elm.first)), *elm.second);

				//Demo of using extra data
				if (elm.second->has_extra_data("F"))
				{
					ExtraData F = elm.second->get_extra_data("F");
					cout << "	Has F matrix:";
					for (size_t i = 0; i < F.size(); i++)
						cout << F[i] << " ";
					cout << endl;
				}

				if (elm.second->has_extra_data("G"))
				{
					ExtraData G = elm.second->get_extra_data("G");
					cout << "	Has G matrix:";
					for (size_t i = 0; i < G.size(); i++)
						cout << G[i] << " ";
					cout << endl;
				}
			}
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

	char c;
	cin >> c;

}