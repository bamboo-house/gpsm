#ifndef __GPSM_GRAPH_IO_H__
#define __GPSM_GRAPH_IO_H__

#include "gutil.h"
#include "csr_graph.cuh"

namespace gpsm{
namespace graph{

	/* Read graph from binary file to memory */
	GPGraph* readBinary(const char* fileName, bool debug = false);

	/* Write in-memory graph to binary file */
	bool writeBinary(GPGraph* graph, const char* fileName, bool debug = false);

	/* Read graph from text file to memory */
	GPGraph* readText(const char* fileName, bool debug = false);

	/* Write in-memory graph to text file */
	bool writeText(GPGraph* graph, const char* fileName, bool debug = false);
}}

#endif