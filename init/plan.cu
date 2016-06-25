#include "plan.h"
#include <queue>

namespace gpsm {
namespace init {
	//---------------------------------------------------------------------------
	struct Node 
	{
		int id;
		float score;
		Direction direct;

		Node(){
			id = 0;
			score = 0;
		}
	};
	//---------------------------------------------------------------------------
	struct CompareNode : public std::binary_function<Node*, Node*, bool>
	{
		bool operator()(const Node* lhs, const Node* rhs) const
		{
			return lhs->score < rhs->score;
		}
	};
	//---------------------------------------------------------------------------
	int getScore(GPGraph* query, GPGraph* graph, Node* inScores, Node* outScores) {
		
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

			inScores[i].id = i;
			inScores[i].direct = Direction::IN;
			inScores[i].score = inDegree / labelSize;

			outScores[i].id = i;
			outScores[i].direct = Direction::OUT;
			outScores[i].score = outDegree / labelSize;

			if (inScores[i].score > maxScore) {
				maxScore = inScores[i].score;
				firstNodeId = i;
			}

			if (outScores[i].score > maxScore) {
				maxScore = outScores[i].score;
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
	bool getPlan(GPGraph* query, GPGraph* graph, GPPlan* plan) {
		FOR_LIMIT(i, query->numNodes) {
			int label = query->nodeLabels[i];
			if (label > graph->numLabels || graph->nodeLabels[label] == 0) return false;
		}

		Node* inScores = (Node*)malloc(sizeof(Node) * query->numNodes);
		CHECK_POINTER(inScores);

		Node* outScores = (Node*)malloc(sizeof(Node) * query->numNodes);
		CHECK_POINTER(outScores);

		bool* visited = (bool*)malloc(sizeof(bool) * query->numNodes);
		CHECK_POINTER(visited);
		FILL(visited, graph->numNodes, false);

		initPlan(plan, query->numNodes);

		int nodeId = getScore(query, graph, inScores, outScores);
		visited[nodeId] = true;

		std::priority_queue<Node*, std::vector<Node*>, CompareNode > pq;

		if (inScores[nodeId].score > outScores[nodeId].score) pq.push(&inScores[nodeId]);
		else pq.push(&outScores[nodeId]);

		plan->numNodes = 0;
		while (pq.empty() == false) {
			Node* node = pq.top();
			nodeId = node->id;

			FOR_RANGE(i, query->outOffsets[nodeId], query->outOffsets[nodeId + 1]) {
				int id = query->outEdges[i];
				if (visited[id] == false) {
					
				}
			}


			pq.pop();
		}


		free(visited);
		free(inScores);
		free(outScores);

		return true;
	}
}}