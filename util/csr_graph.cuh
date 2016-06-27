#ifndef __GPSM_GRAPH_H__
#define __GPSM_GRAPH_H__

#include "gutil.h"

namespace gpsm {
namespace graph {
	struct GPGraph {
		int numNodes; // number of nodes
		int numEdges; // number of edges
		int numLabels; // number of labels
		int maxLabelSize; // maximum number of nodes per label

		int* labelSizes; // numbers of nodes per labels
		int* nodeLabels; // labels of nodes (each node has a single label for simplification)

		int* outOffsets; // points to first endpoint of outgoing edges starting from nodes
		int* outEdges; // endpoints of outgoing edges

		int* inOffsets; // points to first endpoint of incoming edges ending at nodes
		int* inEdges; // endpoints of incoming edges

		short dataPos; // indicate if graph is stored in device memory or main memory

		GPGraph() { // new graph
			numNodes = 0;
			numEdges = 0;
			numLabels = 0;
			maxLabelSize = 0;

			labelSizes = NULL;
			nodeLabels = NULL;
			outOffsets = NULL;
			outEdges = NULL;
			inOffsets = NULL;
			inEdges = NULL;
			dataPos = DataPosition::MEM;
		}

		~GPGraph() { // release memory
			if (dataPos == DataPosition::GPU) {
				if (numNodes > 0) {
					CUDA_SAFE_CALL(cudaFree(nodeLabels));
					CUDA_SAFE_CALL(cudaFree(outOffsets));
					CUDA_SAFE_CALL(cudaFree(inOffsets));
				}

				if (numEdges > 0) {
					CUDA_SAFE_CALL(cudaFree(outEdges));
					CUDA_SAFE_CALL(cudaFree(inEdges));
				}

				if (numLabels > 0) {
					CUDA_SAFE_CALL(cudaFree(labelSizes));
				}
			}
			else if (dataPos == DataPosition::MEM) {
				if (numNodes > 0) {
					free(nodeLabels);
					free(outOffsets);
					free(inOffsets);
				}

				if (numEdges > 0) {
					free(outEdges);
					free(inEdges);
				}

				if (numLabels > 0) {
					free(labelSizes);
				}
			}
		}

		GPGraph* copy(CopyType type) {
			if ((type == CopyType::HOST_TO_DEVICE || type == CopyType::HOST_TO_HOST)
				&& dataPos != DataPosition::MEM) return NULL;

			if (type == CopyType::DEVICE_TO_HOST && dataPos != DataPosition::GPU) return NULL;

			GPGraph* dest = new GPGraph();

			dest->numNodes = numNodes;
			dest->numEdges = numEdges;
			dest->numLabels = numLabels;
			dest->maxLabelSize = maxLabelSize;

			switch (type)
			{
			case HOST_TO_DEVICE:
				dest->dataPos = DataPosition::GPU;

				CUDA_SAFE_CALL(cudaMalloc(&dest->nodeLabels, numNodes * sizeof(int)));
				CUDA_SAFE_CALL(cudaMalloc(&dest->outOffsets, (numNodes + 1) * sizeof(int)));
				CUDA_SAFE_CALL(cudaMalloc(&dest->inOffsets, (numNodes + 1) * sizeof(int)));
				CUDA_SAFE_CALL(cudaMalloc(&dest->outEdges, numEdges * sizeof(int)));
				CUDA_SAFE_CALL(cudaMalloc(&dest->inEdges, numEdges * sizeof(int)));
				CUDA_SAFE_CALL(cudaMalloc(&dest->labelSizes, numLabels * sizeof(int)));

				CUDA_SAFE_CALL(cudaMemcpy(dest->nodeLabels, nodeLabels, numNodes * sizeof(int),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(dest->outOffsets, outOffsets, (numNodes + 1) * sizeof(int),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(dest->inOffsets, inOffsets, (numNodes + 1) * sizeof(int),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(dest->outEdges, outEdges, numEdges * sizeof(int),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(dest->inEdges, inEdges, numEdges * sizeof(int),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(dest->labelSizes, labelSizes, numLabels * sizeof(int),
					cudaMemcpyHostToDevice));
				CUDA_SAFE_CALL(cudaDeviceSynchronize());
				break;
			case DEVICE_TO_HOST:
				dest->dataPos = DataPosition::MEM;

				dest->nodeLabels = new int[numNodes];
				CHECK_POINTER(dest->nodeLabels);

				dest->outOffsets = new int[numNodes + 1];
				CHECK_POINTER(dest->outOffsets);

				dest->inOffsets = new int[numNodes + 1];
				CHECK_POINTER(dest->inOffsets);

				dest->outEdges = new int[numEdges];
				CHECK_POINTER(dest->outEdges);

				dest->inEdges = new int[numEdges];
				CHECK_POINTER(dest->inEdges);

				dest->labelSizes = new int[numNodes];
				CHECK_POINTER(dest->labelSizes);

				CUDA_SAFE_CALL(cudaMemcpy(dest->nodeLabels, nodeLabels, numNodes * sizeof(int),
					cudaMemcpyDeviceToHost));

				CUDA_SAFE_CALL(cudaMemcpy(dest->outOffsets, outOffsets, (numNodes + 1) * sizeof(int),
					cudaMemcpyDeviceToHost));

				CUDA_SAFE_CALL(cudaMemcpy(dest->inOffsets, inOffsets, (numNodes + 1) * sizeof(int),
					cudaMemcpyDeviceToHost));

				CUDA_SAFE_CALL(cudaMemcpy(dest->outEdges, outEdges, numEdges * sizeof(int),
					cudaMemcpyDeviceToHost));

				CUDA_SAFE_CALL(cudaMemcpy(dest->inEdges, inEdges, numEdges * sizeof(int),
					cudaMemcpyDeviceToHost));

				CUDA_SAFE_CALL(cudaMemcpy(dest->labelSizes, labelSizes, numLabels * sizeof(int),
					cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL(cudaDeviceSynchronize());
				break;
			case HOST_TO_HOST:
				dest->dataPos = DataPosition::MEM;

				dest->nodeLabels = new int[numNodes];
				CHECK_POINTER(dest->nodeLabels);

				dest->outOffsets = new int[numNodes+1];
				CHECK_POINTER(dest->outOffsets);

				dest->inOffsets = new int[numNodes + 1];
				CHECK_POINTER(dest->inOffsets);

				dest->outEdges = new int[numEdges];
				CHECK_POINTER(dest->outEdges);

				dest->inEdges = new int[numEdges];
				CHECK_POINTER(dest->inEdges);

				dest->labelSizes = new int[numNodes];
				CHECK_POINTER(dest->labelSizes);

				memcpy(dest->nodeLabels, nodeLabels, numNodes * sizeof(int));
				memcpy(dest->outOffsets, outOffsets, (numNodes + 1) * sizeof(int));
				memcpy(dest->inOffsets, inOffsets, (numNodes + 1) * sizeof(int));
				memcpy(dest->outEdges, outEdges, numEdges * sizeof(int));
				memcpy(dest->inEdges, inEdges, numEdges * sizeof(int));
				memcpy(dest->labelSizes, labelSizes, numLabels * sizeof(int));
			default:
				break;
			}

			return dest;
		}
	};
}}

#endif