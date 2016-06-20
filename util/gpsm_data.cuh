#pragma once

#include "gutil.h"

namespace gpsm {

	// define graph structure used in gpsm
	template <typename VertexId, typename LabelId, typename SizeT>
	struct GPGraph {

		//---------------------------------------------------------------------------
		// Declare properties
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

		//---------------------------------------------------------------------------
		// Constructor
		GPGraph(bool dev = false) { 
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
		//---------------------------------------------------------------------------
		// Get statistics of node labels
		void GetLabelStatistic() {
			if (num_nodes > 0) {
				num_labels = 0;
				for (int i = 0; i < num_nodes; i++) if (node_labels[i] > num_labels)
						num_labels = node_labels[i];

				num_labels++;

				nodes_per_labels = (SizeT*)malloc(num_labels * sizeof(SizeT));
				CHECK_POINTER(nodes_per_labels);
				for (int i = 0; i < num_labels; i++) nodes_per_labels[i] = 0;

				for (int i = 0; i < num_nodes; i++) nodes_per_labels[node_labels[i]]++;

				max_nodes_per_label = 0;
				for (int i = 0; i < num_labels; i++) if (nodes_per_labels[i] > max_nodes_per_label)
					max_nodes_per_label = nodes_per_labels[i];
			}
		}
		//---------------------------------------------------------------------------
		// Get information about incoming nodes and edges
		void GetIncomingEdges() {
			if (num_nodes > 0) {
				in_offsets = (SizeT*)malloc((num_nodes + 1) * sizeof(SizeT));
				CHECK_POINTER(in_offsets);

				in_endpoints = (VertexId*)malloc(num_edges * sizeof(VertexId));
				CHECK_POINTER(in_endpoints);

				SizeT* degrees = (SizeT*)malloc(num_nodes * sizeof(SizeT));
				CHECK_POINTER(degrees);


			}
		}
		//---------------------------------------------------------------------------
		// Read graph data from binary file
		void ReadBinary(char* fileName, bool debug = false) {
			if (debug) {
				printf("Loading graph from binary file ...\n");
			}

			FILE* fp = fopen(fileName, "rb");
			if (fp != NULL) {

				// read statistics
				fread(&num_nodes, sizeof(SizeT), 1, fp);

				// read contents
				node_labels = (LabelId*) malloc(num_nodes * sizeof(LabelId));
				CHECK_POINTER(node_labels);
				out_offsets = (SizeT*) malloc((num_nodes + 1) * sizeof(SizeT));
				CHECK_POINTER(out_offsets);

				fread(node_labels, sizeof(LabelId), num_nodes, fp);
				fread(out_offsets, sizeof(SizeT), num_nodes + 1, fp);

				num_edges = out_offsets[num_nodes];
				out_endpoints = (VertexId*)malloc(num_edges * sizeof(VertexId));
				CHECK_POINTER(out_endpoints);

				fread(out_endpoints, sizeof(VertexId), num_edges, fp);

				// close file
				fclose(fp);

				// calculate label statistics
				GetLabelStatistic();

				// calculate incoming information
				GetIncomingEdges();
			}
		}
		//---------------------------------------------------------------------------
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
		//---------------------------------------------------------------------------
		// Destructor
		~GPGraph() {
			Release();
		}
	};
}