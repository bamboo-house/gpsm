#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "cutil.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace gpsm {

	// define graph structure used in gpsm
	template <typename VertexId, typename LabelId, typename SizeT>
	struct GPGraph {

		SizeT num_nodes; // number of nodes
		SizeT num_edges; // number of edges
		SizeT num_labels; // number of labels
		SizeT max_nodes_per_label; // maximum number of nodes per label

		SizeT* nodes_per_labels; // numbers of nodes per labels
		LabelId* node_labels; // labels of nodes (each node has a single label for simplication)

		SizeT* out_offsets; // points to first endpoint of outgoing edges starting from nodes
		VertexId* out_endpoints; // endpoints of outgoing edges

		SizeT* in_offsets; // points to first endpoint of incoming edges ending at nodes
		VertexId* in_endpoints; // endpoints of incoming edges

		bool in_dev; // indicate if graph is stored in device memory or main memory

		GPGraph(bool dev = false) { // constructor
			num_nodes = 0;
			num_edges = 0;
			num_labels = 0;
			max_nodes_per_label = 0;

			nodes_per_labels = NULL;
			node_labels = NULL;
			out_offsets = NULL;
			out_endpoints = NULL;
			in_offsets = NULL;
			in_endpoints = NULL;
			in_dev = dev;
		}

		void ReadBinary(char* fileName, bool debug = false) { // read graph from binary file
			if (debug) {
				printf("Loading graph from binary file ...\n");
			}

			FILE* fp = fopen(fileName, "rb");
			if (fp != NULL) {
				fread(&num_nodes, sizeof(SizeT), 1, fp);
				fread(&num_edges, sizeof(SizeT), 1, fp);
				fread(&num_labels, sizeof(SizeT), 1, fp);

				node_labels = (LabelId*) malloc(num_nodes * sizeof(LabelId));

				out_offsets = (SizeT*) malloc((num_nodes + 1) * sizeof(SizeT));
				out_endpoints = (VertexId*)malloc(num_edges * sizeof(VertexId));



				fclose(fp);
			}
		}

		void Release() { // release memory
			if (in_dev) {
				if (num_nodes > 0) {
					CUDA_SAFE_CALL(cudaFree(node_labels));
					CUDA_SAFE_CALL(cudaFree(out_offsets));
					CUDA_SAFE_CALL(cudaFree(in_offsets));
				}

				if (num_edges > 0) {
					CUDA_SAFE_CALL(cudaFree(out_endpoints));
					CUDA_SAFE_CALL(cudaFree(in_endpoints));
				}

				if (num_labels > 0) {
					CUDA_SAFE_CALL(cudaFree(nodes_per_labels));
				}
			}
			else {
				if (num_nodes > 0) {
					free(node_labels);
					free(out_offsets);
					free(in_offsets);
				}

				if (num_edges > 0) {
					free(out_endpoints);
					free(in_endpoints);
				}

				if (num_labels > 0) {
					free(nodes_per_labels);
				}
			}
		}

		~GPGraph() {
			Release();
		}
	};
}