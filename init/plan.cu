#include "plan.h"
#include <queue>
#include "../util/graph.h"

namespace gpsm {
namespace init {
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
	int getScore(graph::GPGraph * query, graph::GPGraph * graph, Node* scores) {
		
		float factor = 1, avgSize = 0; // scale factor for refine node scores
		FOR_LIMIT(i, graph->numLabels) avgSize += graph->labelSizes[i];
		avgSize /= graph->numLabels;

		while (avgSize / factor > 1000.0) factor *= 10.0;

		int firstNodeId = 0;
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
				firstNodeId = i;
			}
		}

		return firstNodeId;
	}
	//---------------------------------------------------------------------------
	void initPlan(GPPlan* plan, int size) {
		plan->numNodes = size;

		plan->nodes = (int*)malloc(sizeof(int) * size);
		CHECK_POINTER(plan->nodes);

		plan->scanIn = (bool*)malloc(sizeof(bool) * size);
		CHECK_POINTER(plan->scanIn);

		plan->scanOut = (bool*)malloc(sizeof(bool) * size);
		CHECK_POINTER(plan->scanOut);
	}
	//---------------------------------------------------------------------------
	bool getPlan(graph::GPGraph* query, graph::GPGraph* graph, GPPlan* plan) {
		FOR_LIMIT(i, query->numNodes) {
			int label = query->nodeLabels[i];
			if (label > graph->numLabels || graph->nodeLabels[label] == 0) return false;
		}

		Node* scores = (Node*)malloc(sizeof(Node) * query->numNodes);
		CHECK_POINTER(scores);

		bool* visited = (bool*)malloc(sizeof(bool) * query->numNodes);
		CHECK_POINTER(visited);
		FILL(visited, graph->numNodes, false);

		initPlan(plan, query->numNodes);

		int nodeId = getScore(query, graph, scores);
		visited[nodeId] = true;

		std::priority_queue<Node*, std::vector<Node*>, CompareNode > pq;

		plan->numNodes = 0;
		while (pq.empty() == false) {
			Node* node = pq.top();
			nodeId = node->id;

			FOR_RANGE(i, query->outOffsets[nodeId], query->outOffsets[nodeId + 1]) {
				int id = query->outEdges[i];
				if (visited[id] == false) {
					visited[id] = true;
				}
			}


			pq.pop();
		}


		free(visited);
		free(scores);

		return true;
	}
}}
