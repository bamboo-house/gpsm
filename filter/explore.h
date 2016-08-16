#include "../util/gutil.h"
#include "../util/graph.h"

namespace gpsm {
namespace filter {

	struct GPSpec {
		int numNodes;

		bool** candidateSets;
		int* candidateNodes;

	};

	bool filterNodes();

}}