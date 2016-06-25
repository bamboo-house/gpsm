#include "graphio.h"

namespace gpsm {
namespace graphio {
	//---------------------------------------------------------------------------
	void getLabels(GPGraph* graph) { // get statistics of node labels
		if (graph->numNodes > 0) {
			graph->numLabels = 0;

			FOR_LIMIT(i, graph->numNodes) graph->numLabels = std::max(graph->numLabels, graph->nodeLabels[i]);
			graph->numLabels++;

			graph->labelSizes = (int*)malloc(graph->numLabels * sizeof(int));
			CHECK_POINTER(graph->labelSizes);

			FOR_LIMIT(i, graph->numLabels) graph->labelSizes[i] = 0;
			FOR_LIMIT(i, graph->numNodes) graph->labelSizes[graph->nodeLabels[i]]++;

			graph->maxLabelSize = 0;
			FOR_LIMIT(i, graph->numLabels)graph->maxLabelSize = std::max(graph->maxLabelSize, graph->labelSizes[i]);
		}
	}
	//---------------------------------------------------------------------------
	void getInEdges(GPGraph* graph) { // get information about incoming nodes and edges
		if (graph->numNodes > 0) {
			graph->inOffsets = (int*)malloc((graph->numNodes + 1) * sizeof(int));
			CHECK_POINTER(graph->inOffsets);

			graph->inEdges = (int*)malloc(graph->numEdges * sizeof(int));
			CHECK_POINTER(graph->inEdges);

			// calculate in-offsets of nodes
			FOR_LIMIT(i, graph->numNodes)
				FOR_RANGE(j, graph->outOffsets[i], graph->outOffsets[i + 1]) {
				int node2 = graph->outEdges[j];
				graph->inOffsets[node2]++;
			}
			graph->inOffsets[graph->numNodes] = 0;

			// prefix sum
			FOR_LIMIT(i, graph->numNodes) graph->inOffsets[i + 1] += graph->inOffsets[i];

			// calculate in-endpoints
			FOR_LIMIT_REV(i, graph->numNodes)
				FOR_RANGE(j, graph->outOffsets[i], graph->outOffsets[i + 1]) {
				int node = graph->outEdges[j];
				int offset = --graph->inOffsets[node];
				graph->inEdges[offset] = i;
			}
		}
	}
	//---------------------------------------------------------------------------
	bool readBinary(GPGraph* graph, char* fileName, bool debug) {
		FILE* fp = fopen(fileName, "rb");
		if (fp == NULL) return false;

		if (debug) {
			printf("Loading graph from binary file ...\n");
		}

		// read statistics
		fread(&graph->numNodes, sizeof(int), 1, fp);

		// read contents
		graph->nodeLabels = (int*)malloc(graph->numNodes * sizeof(int));
		CHECK_POINTER(graph->nodeLabels);

		graph->outOffsets = (int*)malloc((graph->numNodes + 1) * sizeof(int));
		CHECK_POINTER(graph->outOffsets);

		fread(graph->nodeLabels, sizeof(int), graph->numNodes, fp);
		fread(graph->outOffsets, sizeof(int), graph->numNodes + 1, fp);

		graph->numEdges = graph->outOffsets[graph->numNodes];
		graph->outEdges = (int*)malloc(graph->numEdges * sizeof(int));
		CHECK_POINTER(graph->outEdges);

		fread(graph->outEdges, sizeof(int), graph->numEdges, fp);

		// close file
		fclose(fp);

		// calculate label statistics
		getLabels(graph);

		// calculate incoming information
		getInEdges(graph);

		return true;
	}
	//---------------------------------------------------------------------------
	bool writeBinary(GPGraph* graph, char* fileName, bool debug) {
		FILE* fp = fopen(fileName, "wb");
		if (fp == NULL) return false;

		if (debug) {
			printf("Writing graph to binary file ...\n");
		}

		// write statistics
		fwrite(&graph->numNodes, sizeof(int), 1, fp);

		// write contents
		fwrite(graph->nodeLabels, sizeof(int), graph->numNodes, fp);
		fwrite(graph->outOffsets, sizeof(int), graph->numNodes + 1, fp);
		fwrite(graph->outEdges, sizeof(int), graph->numEdges, fp);

		// close file
		fclose(fp);

		return true;
	}
	//---------------------------------------------------------------------------
	bool readStatistics(GPGraph* graph, char* fileName, bool debug) { // read graph statistics from text format
		std::ifstream in(fileName);

		if (in.is_open()) {
			if (debug) {
				printf("Reading graph statistics from text file ...\n");
			}

			std::string line;
			while (getline(in, line)) {
				if (line[0] == 'v') graph->numNodes++;
				else if (line[0] == 'e') graph->numEdges++;
			}

			in.close();
			return true;
		}

		return false;
	}
	//---------------------------------------------------------------------------
	bool readNodes(GPGraph* graph, char* fileName, bool debug) { // read node information from text format
		std::ifstream in(fileName);

		if (in.is_open()) {
			if (debug) {
				printf("Reading node information from text file ...\n");
			}

			std::string line;
			while (getline(in, line)) {
				if (line[0] == 'v') { // get node information
					std::istringstream iss(line);

					std::string type;
					int node;
					int label;

					if (!(iss >> type >> node >> label)) {
						printf("Graph format errors!...\n");
						return false;
					} // error

					if (node >= graph->numNodes) {
						printf("Graph format errors!...\n");
						return false;
					} // error

					graph->nodeLabels[node] = label;
				}
				else if (line[0] == 'e') { // get edge information

					std::istringstream iss(line);
					std::string type;
					int node1;
					int node2;

					if (!(iss >> type >> node1 >> node2)) {
						printf("Graph format errors!...\n");
						return false;
					} // error

					if (node1 >= graph->numNodes || node2 >= graph->numNodes) {
						printf("Graph format errors!...\n");
						return false;
					} // error

					graph->outOffsets[node1]++;
					graph->inOffsets[node2]++;
				}
			}

			in.close();

			return true;
		}

		return false;
	}
	//---------------------------------------------------------------------------
	bool readEdges(GPGraph* graph, char* fileName, bool debug) { // get edge information from text format
		std::ifstream in(fileName);

		if (in.is_open()) {
			if (debug) {
				printf("Reading graph contents from text file ...\n");
			}

			// prefix sum
			FOR_LIMIT(i, graph->numNodes) {
				graph->inOffsets[i + 1] += graph->inOffsets[i];
				graph->outOffsets[i + 1] += graph->outOffsets[i];
			}

			std::string line;
			while (getline(in, line)) {
				if (line[0] == 'e') { // get edge information
					std::istringstream iss(line);
					std::string type;
					int node1;
					int node2;

					if (!(iss >> type >> node1 >> node2)) {
						printf("Graph format errors!...\n");
						return false;
					} // error

					graph->outEdges[--graph->outOffsets[node1]] = node2;
					graph->inEdges[--graph->inOffsets[node2]] = node1;
				}
			}

			in.close();
			return true;
		}

		return false;
	}
	//---------------------------------------------------------------------------
	bool readText(GPGraph* graph, char* fileName, bool debug) {

		if (readStatistics(graph, fileName, debug) == false) return false;

		// init node and edge arrays
		graph->nodeLabels = (int*)malloc(graph->numNodes * sizeof(int));
		CHECK_POINTER(graph->nodeLabels);

		graph->outOffsets = (int*)malloc((graph->numNodes + 1) * sizeof(int));
		CHECK_POINTER(graph->outOffsets);
		FILL(graph->outOffsets, graph->numNodes + 1, 0);

		graph->inOffsets = (int*)malloc((graph->numNodes + 1) * sizeof(int));
		CHECK_POINTER(graph->inOffsets);
		FILL(graph->inOffsets, graph->numNodes + 1, 0);

		graph->outEdges = (int*)malloc(graph->numEdges * sizeof(int));
		CHECK_POINTER(graph->outEdges);

		graph->inEdges = (int*)malloc(graph->numEdges * sizeof(int));
		CHECK_POINTER(graph->inEdges);

		// get node information
		if (readNodes(graph, fileName, debug) == false) return false;

		// get label information
		getLabels(graph);

		// get edge information
		return readEdges(graph, fileName, debug);
	}
	//---------------------------------------------------------------------------
	bool writeText(GPGraph* graph, char* fileName, bool debug) { // read graph data from binary file
		std::ofstream out(fileName);
		if (out.is_open()) {
			FOR_LIMIT(i, graph->numNodes) out << "v " << i << " " << graph->nodeLabels[i] << std::endl;

			FOR_LIMIT(i, graph->numNodes)
				FOR_RANGE(j, graph->outOffsets[i], graph->outOffsets[i + 1])
				out << "e " << i << " " << graph->outEdges[j] << std::endl;

			out.close();
			return true;
		}

		return false;
	}
	//---------------------------------------------------------------------------
	bool copy(GPGraph* dest, GPGraph* src, CopyType type) {
		if ((type == CopyType::HOST_TO_DEVICE || type == CopyType::HOST_TO_HOST)
			&& src->dataPos != DataPosition::MEM) return NULL;

		if (type == CopyType::DEVICE_TO_HOST && src->dataPos != DataPosition::GPU) return NULL;

		dest->numNodes = src->numNodes;
		dest->numEdges = src->numEdges;
		dest->numLabels = src->numLabels;
		dest->maxLabelSize = src->maxLabelSize;

		switch (type)
		{
		case HOST_TO_DEVICE:
			dest->dataPos = DataPosition::GPU;

			CUDA_SAFE_CALL(cudaMalloc(&dest->nodeLabels, src->numNodes * sizeof(int)));
			CUDA_SAFE_CALL(cudaMalloc(&dest->outOffsets, (src->numNodes + 1) * sizeof(int)));
			CUDA_SAFE_CALL(cudaMalloc(&dest->inOffsets, (src->numNodes + 1) * sizeof(int)));
			CUDA_SAFE_CALL(cudaMalloc(&dest->outEdges, src->numEdges * sizeof(int)));
			CUDA_SAFE_CALL(cudaMalloc(&dest->inEdges, src->numEdges * sizeof(int)));
			CUDA_SAFE_CALL(cudaMalloc(&dest->labelSizes, src->numLabels * sizeof(int)));

			CUDA_SAFE_CALL(cudaMemcpy(dest->nodeLabels, src->nodeLabels, src->numNodes * sizeof(int),
				cudaMemcpyHostToDevice));

			CUDA_SAFE_CALL(cudaMemcpy(dest->outOffsets, src->outOffsets, (src->numNodes + 1) * sizeof(int),
				cudaMemcpyHostToDevice));

			CUDA_SAFE_CALL(cudaMemcpy(dest->inOffsets, src->inOffsets, (src->numNodes + 1) * sizeof(int),
				cudaMemcpyHostToDevice));

			CUDA_SAFE_CALL(cudaMemcpy(dest->outEdges, src->outEdges, src->numEdges * sizeof(int),
				cudaMemcpyHostToDevice));

			CUDA_SAFE_CALL(cudaMemcpy(dest->inEdges, src->inEdges, src->numEdges * sizeof(int),
				cudaMemcpyHostToDevice));

			CUDA_SAFE_CALL(cudaMemcpy(dest->labelSizes, src->labelSizes, src->numLabels * sizeof(int),
				cudaMemcpyHostToDevice));

			return true;
		case HOST_TO_HOST:
			dest->dataPos = DataPosition::MEM;

			dest->nodeLabels = (int*)malloc(src->numNodes * sizeof(int));
			CHECK_POINTER(dest->nodeLabels);

			dest->outOffsets = (int*)malloc((src->numNodes + 1) * sizeof(int));
			CHECK_POINTER(dest->outOffsets);

			dest->inOffsets = (int*)malloc((src->numNodes + 1) * sizeof(int));
			CHECK_POINTER(dest->inOffsets);

			dest->outEdges = (int*)malloc(src->numEdges * sizeof(int));
			CHECK_POINTER(dest->outEdges);

			dest->inEdges = (int*)malloc(src->numEdges * sizeof(int));
			CHECK_POINTER(dest->inEdges);

			dest->labelSizes = (int*)malloc(src->numLabels * sizeof(int));
			CHECK_POINTER(dest->labelSizes);

			memcpy(dest->nodeLabels, src->nodeLabels, src->numNodes * sizeof(int));
			memcpy(dest->outOffsets, src->outOffsets, (src->numNodes + 1) * sizeof(int));
			memcpy(dest->inOffsets, src->inOffsets, (src->numNodes + 1) * sizeof(int));
			memcpy(dest->outEdges, src->outEdges, src->numEdges * sizeof(int));
			memcpy(dest->inEdges, src->inEdges, src->numEdges * sizeof(int));
			memcpy(dest->labelSizes, src->labelSizes, src->numLabels * sizeof(int));
			
			return true;
		case DEVICE_TO_HOST:
			break;
		default:
			break;
		}

		return false;
	}
}}