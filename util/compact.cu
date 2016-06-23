#include "scan.h"
#include "gutil.h"

namespace gpsm {

	__global__ void scatter(const bool *in, int *out, const int len) {
		const int threadId = GTID;
		if (threadId < len) {
			out[threadId] = in[threadId] == false ? 0 : 1;
		}
	}

	int compact(bool* d_inArr, int* d_outArr, int numRecords) {
		uint blocksPerGrid = (numRecords + BLOCK_SIZE - 1) / BLOCK_SIZE;
		
		int* d_tmp;
		CUDA_SAFE_CALL(cudaMalloc(&d_tmp, numRecords * sizeof(int)));


		CUDA_SAFE_CALL(cudaFree(d_tmp));
	}
}