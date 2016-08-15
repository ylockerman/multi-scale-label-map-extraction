cmake_minimum_required (VERSION 2.8.11)
project (LABEL_MAP_IO)

include_directories(include)
include_directories(include_matio)

add_library(label_map_io src_matio/endian.c src_matio/inflate.c src_matio/io.c 
						 src_matio/mat.c    src_matio/mat4.c    src_matio/mat5.c 
						 src_matio/mat73.c  src_matio/matvar_cell.c 
						 src_matio/matvar_struct.c src_matio/read_data.c 
						 src_matio/snprintf.c 
						 src/HSLIC.cpp include/HSLIC.h 
						 src/region_tree.cpp include/region_tree.h)

target_include_directories(label_map_io PRIVATE 
								${CMAKE_CURRENT_SOURCE_DIR}/include_matio)

add_executable (HSLIC_demo src/HSLIC_demo.cpp)
target_link_libraries (HSLIC_demo LINK_PUBLIC label_map_io)
