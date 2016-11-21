#ifndef EXAMPLE_UTILS_H
#define EXAMPLE_UTILS_H

typedef unsigned char byte;
struct color {
	byte r, g, b;
};


//output to PPM format
//See http://netpbm.sourceforge.net/doc/ppm.html
template<typename RegionMap> 
void output_PPM(std::string filename, const RegionMap& rmap)
{

	//Get the array of superpixels at the scale of interest.
	std::shared_ptr<typename RegionMap::AbstractRegionMap_type> regions = rmap.copy_map_for_new_image<color>();

	//Create a random color for each superpixel, and fill in the image to that color 
	for (const CompoundRegionMap<color>::RegionKey& key : *regions)
	{
		//std::valarray<color> data = ;

		color c;
		c.r = (byte)(rand() % 255);
		c.g = (byte)(rand() % 255);
		c.b = (byte)(rand() % 255);

		(*regions)[key] = c;

		//for(int c = 0; c < 3; c++)
		//	data[std::slice(c, data.size(), 3)] = (byte)(rand() % 255);

		//(*regions)[key] = data;
	}
	ofstream output(filename);

	//P3 is the magic number of an ascii PPM.
	output << "P3" << endl;

	//width height and then max color
	output << regions->get_cols() << " " << regions->get_rows() << " 255" << endl;

	//PPM output each color in row major order. We format it in rows to make it easer to read
	for (int r = 0; r < regions->get_rows(); r++)
	{
		for (int c = 0; c < regions->get_cols(); c++)
		{
			const valarray<color>& color = regions->value_at(r, c);
			output << (int)color[0].r << " " << (int)color[0].g << " " << (int)color[0].b << " ";
		}
		output << endl;
	}

	output.close();
}

#endif