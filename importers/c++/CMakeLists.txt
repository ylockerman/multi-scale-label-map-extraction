cmake_minimum_required (VERSION 2.8.11)
project (LABEL_MAP_IO)

include_directories(include)
include_directories(include_matio)

add_library(label_map_io src_matio/endian.c src_matio/inflate.c src_matio/io.c 
						 src_matio/mat.c    src_matio/mat4.c    src_matio/mat5.c 
						 src_matio/mat73.c  src_matio/matvar_cell.c 
						 src_matio/matvar_struct.c src_matio/read_data.c 
						 src_matio/snprintf.c 
						 include/HSLIC.h include/region_map.h
						 src/region_io.cpp include/region_io.h
						 include/label_map.h)

target_include_directories(label_map_io PRIVATE 
								${CMAKE_CURRENT_SOURCE_DIR}/include_matio)

add_executable (HSLIC_demo examples/HSLIC_demo.cpp examples/example_utils.h)
target_link_libraries (HSLIC_demo LINK_PUBLIC label_map_io)

add_executable (label_map_demo examples/label_map_demo.cpp examples/example_utils.h)
target_link_libraries (label_map_demo LINK_PUBLIC label_map_io)

add_executable (HSLIC_features_demo examples/HSLIC_features_demo.cpp examples/example_utils.h)
target_link_libraries (HSLIC_features_demo LINK_PUBLIC label_map_io)
