#include "../util/gutil.h"
#include "../util/csr_graph.cuh"

namespace gpsm {
namespace filter {

	struct GPSpec {
		int numNodes;

		bool** candidateSets;
		
	};

	bool filterNodes();

}}