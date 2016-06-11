#pragma once

template <typename VertexId, typename SizeT>
struct GPGraph {
	SizeT num_nodes;
	SizeT num_edges;

	SizeT* out_offsets;
	VertexId* out_indices;

	SizeT* in_offsets;
	VertexId* in_indices;
	
	bool in_device;

	void deleteGraph() {
		if (in_device) {

		}
		else {

		}
	}
};