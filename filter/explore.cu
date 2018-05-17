#include "explore.h"
#include <queue>

namespace gpsm {
namespace filter {
	//---------------------------------------------------------------------------
	struct Node
	{
		int id;
		float scoreIn;
		float scoreOut;

		Node(){
			id = 0;
			scoreIn = 0;
			scoreOut = 0;
		}
	};
	//---------------------------------------------------------------------------
	struct CompareNode : public std::binary_function<Node*, Node*, bool>
	{
		bool operator()(const Node* lhs, const Node* rhs) const
		{
			return std::max(lhs->scoreIn, lhs->scoreOut) < std::max(rhs->scoreIn, rhs->scoreOut);
		}
	};
	//---------------------------------------------------------------------------
	int getScore(	graph::GPGraph* query, 
					graph::GPGraph* graph, 
					Node* scores)
	{

		float factor = 1, avgSize = 0; // scale factor for refine node scores
		FOR_LIMIT(i, graph->numLabels) avgSize += graph->labelSizes[i];
		avgSize /= graph->numLabels;

		while (avgSize / factor > 1000.0) factor *= 10.0;

		int root = 0;
		float maxScore = 0;

		FOR_LIMIT(i, query->numNodes) {
			float labelSize = query->labelSizes[query->nodeLabels[i]] / factor;
			int inDegree = query->inOffsets[i + 1] - query->inOffsets[i];
			int outDegree = query->outOffsets[i + 1] - query->outOffsets[i];

			scores[i].id = i;
			scores[i].scoreIn = inDegree / labelSize;
			scores[i].scoreOut = outDegree / labelSize;

			if (std::max(scores[i].scoreIn, scores[i].scoreOut) > maxScore) {
				maxScore = std::max(scores[i].scoreIn, scores[i].scoreOut);
				root = i;
			}
		}

		return root;
	}
	//---------------------------------------------------------------------------
	bool checkEmptyLabel(graph::GPGraph* query, graph::GPGraph* graph) {
		FOR_LIMIT(i, query->numNodes) {
			int label = query->nodeLabels[i];
			if (label > graph->numLabels || graph->nodeLabels[label] == 0) return false;
		}

		return true;
	}
	//---------------------------------------------------------------------------
	bool initCandidateNodes(graph::GPGraph* query, 
							graph::GPGraph* d_query, 
							graph::GPGraph* graph,
							graph::GPGraph* d_graph) 
	{
		// check if there is any label in the query graph having zero-size
		if (checkEmptyLabel(query, graph) == false) return false;
		
		// calculate node scores
		Node* scores = (Node*)malloc(sizeof(Node) * query->numNodes);
		CHECK_POINTER(scores);
		int nodeId = getScore(query, graph, scores); // return the root node

		// monitor filtered nodes
		bool* visited = (bool*)malloc(sizeof(bool) * query->numNodes);
		CHECK_POINTER(visited);
		FILL(visited, graph->numNodes, false);


		return true;
	}
	//---------------------------------------------------------------------------

}
}
