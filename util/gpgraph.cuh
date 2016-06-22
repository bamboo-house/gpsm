#pragma once

#include "gutil.h"

using namespace std;

namespace gpsm {

	// define graph structure used in gpsm
	template <typename VertexId, typename LabelId, typename SizeT>
	struct GPGraph {

		SizeT numNodes; // number of nodes
		SizeT numEdges; // number of edges
		SizeT numLabels; // number of labels
		SizeT maxLabelSize; // maximum number of nodes per label

		SizeT* labelSizes; // numbers of nodes per labels
		LabelId* nodeLabels; // labels of nodes (each node has a single label for simplication)

		SizeT* outOffsets; // points to first endpoint of outgoing edges starting from nodes
		VertexId* outEdges; // endpoints of outgoing edges

		SizeT* inOffsets; // points to first endpoint of incoming edges ending at nodes
		VertexId* inEdges; // endpoints of incoming edges

		DataPosition dataPos; // indicate if graph is stored in device memory or main memory

		//---------------------------------------------------------------------------
		void _GetLabels() { // get statistics of node labels
			if (numNodes > 0) {
				numLabels = 0;

				FOR_LIMIT(i, numNodes) numLabels = max(numLabels, nodeLabels[i]);
				numLabels++;

				labelSizes = (SizeT*)malloc(numLabels * sizeof(SizeT));
				CHECK_POINTER(labelSizes);

				FOR_LIMIT(i, numLabels) labelSizes[i] = 0;
				FOR_LIMIT(i, numNodes) labelSizes[nodeLabels[i]]++;

				maxLabelSize = 0;
				FOR_LIMIT(i, numLabels) maxLabelSize = max(maxLabelSize, labelSizes[i]);
			}
		}
		//---------------------------------------------------------------------------
		void _GetInEdges() { // get information about incoming nodes and edges
			if (numNodes > 0) {
				inOffsets = (SizeT*)malloc((numNodes + 1) * sizeof(SizeT));
				CHECK_POINTER(inOffsets);

				inEdges = (VertexId*)malloc(numEdges * sizeof(VertexId));
				CHECK_POINTER(inEdges);

				// calculate in-offsets of nodes
				FOR_LIMIT(i, numNodes)
					FOR_RANGE(j, outOffsets[i], outOffsets[i + 1]) {
					int node2 = outEdges[j];
					inOffsets[node2]++;
				}
				inOffsets[numNodes] = 0;

				// prefix sum
				FOR_LIMIT(i, numNodes) inOffsets[i + 1] += inOffsets[i];

				// calculate in-endpoints
				FOR_LIMIT_REV(i, numNodes)
					FOR_RANGE(j, outOffsets[i], outOffsets[i + 1]) {
					int node = outEdges[j];
					int offset = --inOffsets[node];
					inEdges[offset] = i;
				}
			}
		}
		//---------------------------------------------------------------------------
		void _ReadStatistics(char* fileName, bool debug = false) { // read graph statistics from text format
			ifstream in(fileName);

			if (in.is_open()) {
				if (debug) {
					printf("Reading graph statistics from text file ...\n");
				}

				string line;
				while (getline(in, line)) {
					if (line[0] == 'v') numNodes++;
					else if (line[0] == 'e') numEdges++;
				}

				in.close();
			}
		}
		//---------------------------------------------------------------------------
		void _ReadNodes(char* fileName, bool debug = false) { // read node information from text format
			ifstream in(fileName);

			if (in.is_open()) {
				if (debug) {
					printf("Reading node information from text file ...\n");
				}

				string line;
				while (getline(in, line)) {
					if (line[0] == 'v') { // get node information
						istringstream iss(line);

						string type;
						VertexId node;
						LabelId label;

						if (!(iss >> type >> node >> label)) {
							printf("Graph format errors!...\n");
							exit(1);
						} // error

						if (node >= numNodes) {
							printf("Graph format errors!...\n");
							exit(1);
						} // error

						nodeLabels[node] = label;
					}
					else if (line[0] == 'e') { // get edge information

						istringstream iss(line);
						string type;
						VertexId node1;
						VertexId node2;

						if (!(iss >> type >> node1 >> node2)) {
							printf("Graph format errors!...\n");
							exit(1);
						} // error

						if (node1 >= numNodes || node2 >= numNodes) {
							printf("Graph format errors!...\n");
							exit(1);
						} // error

						outOffsets[node1]++;
						inOffsets[node2]++;
					}
				}

				in.close();
			}
		}
		//---------------------------------------------------------------------------
		void _GetEdges(char* fileName, bool debug = false) { // get edge information from text format
			ifstream in(fileName);

			if (in.is_open()) {
				if (debug) {
					printf("Reading graph contents from text file ...\n");
				}

				// prefix sum
				FOR_LIMIT(i, numNodes) {
					inOffsets[i + 1] += inOffsets[i];
					outOffsets[i + 1] += outOffsets[i];
				}

				string line;
				while (getline(in, line)) {
					if (line[0] == 'e') { // get edge information
						istringstream iss(line);
						string type;
						VertexId node1;
						VertexId node2;

						if (!(iss >> type >> node1 >> node2)) {
							printf("Graph format errors!...\n");
							exit(1);
						} // error

						outEdges[--outOffsets[node1]] = node2;
						inEdges[--inOffsets[node2]] = node1;
					}
				}

				in.close();
			}
		}
		//---------------------------------------------------------------------------
		void _Release() { // release memory
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
		//---------------------------------------------------------------------------
		GPGraph(DataPosition pos = DataPosition::MEM) { // constructor
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
			dataPos = pos;
		}
		//---------------------------------------------------------------------------
		~GPGraph() { // destructor
			_Release();
		}
		//---------------------------------------------------------------------------
		void ReadBinary(char* fileName, bool debug = false) { // read graph data from binary file
			FILE* fp = fopen(fileName, "rb");
			if (fp != NULL) {
				if (debug) {
					printf("Loading graph from binary file ...\n");
				}

				// read statistics
				fread(&numNodes, sizeof(SizeT), 1, fp);

				// read contents
				nodeLabels = (LabelId*)malloc(numNodes * sizeof(LabelId));
				CHECK_POINTER(nodeLabels);
				outOffsets = (SizeT*)malloc((numNodes + 1) * sizeof(SizeT));
				CHECK_POINTER(outOffsets);

				fread(nodeLabels, sizeof(LabelId), numNodes, fp);
				fread(outOffsets, sizeof(SizeT), numNodes + 1, fp);

				numEdges = outOffsets[numNodes];
				outEdges = (VertexId*)malloc(numEdges * sizeof(VertexId));
				CHECK_POINTER(outEdges);

				fread(outEdges, sizeof(VertexId), numEdges, fp);

				// close file
				fclose(fp);

				// calculate label statistics
				_GetLabels();

				// calculate incoming information
				_GetInEdges();
			}
		}
		//---------------------------------------------------------------------------
		void WriteBinary(char* fileName, bool debug = false) { // write graph data to binary file
			FILE* fp = fopen(fileName, "wb");
			if (fp != NULL) {
				if (debug) {
					printf("Writing graph to binary file ...\n");
				}

				// write statistics
				fwrite(&numNodes, sizeof(SizeT), 1, fp);

				// write contents
				fwrite(nodeLabels, sizeof(LabelId), numNodes, fp);
				fwrite(outOffsets, sizeof(SizeT), numNodes + 1, fp);
				fwrite(outEdges, sizeof(VertexId), numEdges, fp);

				// close file
				fclose(fp);
			}
		}
		//---------------------------------------------------------------------------
		void ReadText(char* fileName, bool debug = false) { // read graph data from binary file
			// get graph statistics
			_ReadStatistics(fileName, debug);

			// init node and edge arrays
			nodeLabels = (LabelId*)malloc(numNodes * sizeof(LabelId));
			CHECK_POINTER(nodeLabels);

			outOffsets = (SizeT*)malloc((numNodes + 1) * sizeof(SizeT));
			CHECK_POINTER(outOffsets);
			FILL(outOffsets, numNodes + 1, 0);

			inOffsets = (SizeT*)malloc((numNodes + 1) * sizeof(SizeT));
			CHECK_POINTER(inOffsets);
			FILL(inOffsets, numNodes + 1, 0);

			outEdges = (VertexId*)malloc(numEdges * sizeof(VertexId));
			CHECK_POINTER(outEdges);

			inEdges = (VertexId*)malloc(numEdges * sizeof(VertexId));
			CHECK_POINTER(inEdges);

			// get node information
			_ReadNodes(fileName, debug);

			// get label information
			_GetLabels();

			// get edge information
			_GetEdges(fileName, debug);
		}
		//---------------------------------------------------------------------------
		void WriteText(char* fileName, bool debug = false) { // read graph data from binary file
			ofstream out(fileName);
			if (out.is_open()) {
				FOR_LIMIT(i, numNodes) out << "v " << i << " " << nodeLabels[i] << endl;

				FOR_LIMIT(i, numNodes)
					FOR_RANGE(j, outOffsets[i], outOffsets[i + 1])
					out << "e " << i << " " << outEdges[j] << endl;

				out.close();
			}
		}
		//---------------------------------------------------------------------------
		GPGraph* Copy(CopyType type) {
			if ( (type == CopyType::HOST_TO_DEVICE || type == CopyType::HOST_TO_HOST)
				 && dataPos != DataPosition::MEM) return NULL;

			if (type == CopyType::DEVICE_TO_HOST && dataPos != DataPosition::GPU) return NULL;
			
			GPGraph* clone = (GPGraph*)malloc(sizeof(GPGraph));
			CHECK_POINTER(clone);

			clone->numNodes = numNodes;
			clone->numEdges = numEdges;
			clone->numLabels = numLabels;
			clone->maxLabelSize = maxLabelSize;

			switch (type)
			{
			case HOST_TO_DEVICE:
				clone->dataPos = DataPosition::GPU;

				CUDA_SAFE_CALL(cudaMalloc(&clone->nodeLabels, numNodes * sizeof(LabelId)));
				CUDA_SAFE_CALL(cudaMalloc(&clone->outOffsets, (numNodes + 1) * sizeof(SizeT)));
				CUDA_SAFE_CALL(cudaMalloc(&clone->inOffsets, (numNodes + 1) * sizeof(SizeT)));
				CUDA_SAFE_CALL(cudaMalloc(&clone->outEdges, numEdges * sizeof(VertexId)));
				CUDA_SAFE_CALL(cudaMalloc(&clone->inEdges, numEdges * sizeof(VertexId)));
				CUDA_SAFE_CALL(cudaMalloc(&clone->labelSizes, numLabels * sizeof(SizeT)));

				CUDA_SAFE_CALL(cudaMemcpy(clone->nodeLabels, nodeLabels, numNodes * sizeof(LabelId),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(clone->outOffsets, outOffsets, (numNodes + 1) * sizeof(SizeT),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(clone->inOffsets, inOffsets, (numNodes + 1) * sizeof(SizeT),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(clone->outEdges, outEdges, numEdges * sizeof(VertexId),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(clone->inEdges, inEdges, numEdges * sizeof(VertexId),
					cudaMemcpyHostToDevice));

				CUDA_SAFE_CALL(cudaMemcpy(clone->labelSizes, labelSizes, numLabels * sizeof(SizeT),
					cudaMemcpyHostToDevice));

				break;
			case HOST_TO_HOST:
				clone->dataPos = DataPosition::MEM;

				clone->nodeLabels = (LabelId*)malloc(numNodes * sizeof(LabelId));
				CHECK_POINTER(clone->nodeLabels);

				clone->outOffsets = (SizeT*)malloc((numNodes + 1) * sizeof(SizeT));
				CHECK_POINTER(clone->outOffsets);

				clone->inOffsets = (SizeT*)malloc((numNodes + 1) * sizeof(SizeT));
				CHECK_POINTER(clone->inOffsets);

				clone->outEdges = (VertexId*)malloc(numEdges * sizeof(VertexId));
				CHECK_POINTER(clone->outEdges);

				clone->inEdges = (VertexId*)malloc(numEdges * sizeof(VertexId));
				CHECK_POINTER(clone->inEdges);

				clone->labelSizes = (SizeT*)malloc(numLabels * sizeof(SizeT));
				CHECK_POINTER(clone->labelSizes);

				memcpy(clone->nodeLabels, nodeLabels, numNodes * sizeof(LabelId));
				memcpy(clone->outOffsets, outOffsets, (numNodes + 1) * sizeof(SizeT));
				memcpy(clone->inOffsets, inOffsets, (numNodes + 1) * sizeof(SizeT));
				memcpy(clone->outEdges, outEdges, numEdges * sizeof(VertexId));
				memcpy(clone->inEdges, inEdges, numEdges * sizeof(VertexId));
				memcpy(clone->labelSizes, labelSizes, numLabels * sizeof(SizeT));
				break;
			case DEVICE_TO_HOST:
				break;
			default:
				break;
			}

			return clone;
		}
		//---------------------------------------------------------------------------
		void Print() { // print graph
			cout << "Node count " << numNodes << endl;
			cout << "Edge count " << numEdges << endl;
			cout << "Label count " << numLabels << endl;
			cout << "Max nodes per label " << maxLabelSize << endl;
			FOR_LIMIT(i, numNodes) {
				cout << "Node " << i << "(" << nodeLabels[i] << ")" << ":";
				FOR_RANGE(j, outOffsets[i], outOffsets[i + 1]) {
					cout << " " << outEdges[j];
				}
				cout << endl;
			}
		}
	};
}