#pragma once

#ifndef __GPSM_SCAN_H__
#define __GPSM_SCAN_H__

namespace gpsm {
	namespace scan {
	
		template <typename VertexId, typename SizeT>
		struct GPScan {
			VertexId** g_scanBlockSums;
			SizeT g_numEltsAllocated = 0;
			SizeT g_numLevelsAllocated = 0;
			
			int prefixSum(VertexId* d_inArr, VertexId* d_outArr, SizeT numRecords);
			/// test
		};

		// Stream compaction on the device
		int compact(bool* d_inArr, int* d_outArr, int numRecords);
	}
}

#endif
