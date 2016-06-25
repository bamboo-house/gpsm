#ifndef __GPSM_GRAPH_IO_H__
#define __GPSM_GRAPH_IO_H__

#include "gutil.h"

namespace gpsm{
namespace graphio {

	/* Read graph from binary file to memory */
	bool readBinary(GPGraph* graph, char* fileName, bool debug = false);

	/* Write in-memory graph to binary file */
	bool writeBinary(GPGraph* graph, char* fileName, bool debug = false);

	/* Read graph from text file to memory */
	bool readText(GPGraph* graph, char* fileName, bool debug = false);

	/* Write in-memory graph to text file */
	bool writeText(GPGraph* graph, char* fileName, bool debug = false);

	/* Copy graphs */
	bool copy(GPGraph* dest, GPGraph* src, CopyType type);
}}

#endif