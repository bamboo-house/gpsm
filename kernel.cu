#include "util/graph_io.h"
#include "util/large_node.cuh"
#include "util/gutil.h"
#include "util/Timer.h"
#include <iostream>
#include <vector>
//---------------------------------------------------------------------------
using namespace gpsm;
//---------------------------------------------------------------------------
Timer timer;

#define CMT_LOAD "load"
#define CMT_INFO "graph"
#define CMT_CLEAR "clear"
#define CMT_EXEC "exec"
#define CMT_HELP "help"
#define CMT_EXIT "exit"

std::vector<std::string> allowed() {
	std::vector<std::string> all;
	all.push_back(CMT_LOAD);
	all.push_back(CMT_INFO);
	all.push_back(CMT_CLEAR);
	all.push_back(CMT_EXEC);
	all.push_back(CMT_HELP);
	all.push_back(CMT_EXIT);
	return all;
}
//---------------------------------------------------------------------------
bool isAllowed(std::string command, std::vector<std::string> all)
{
	FOR_LIMIT(i, all.size())
		if (command == all[i]) return true;

	return false;
}
//---------------------------------------------------------------------------
void showGraph(graph::GPGraph* db)
{
	if (db != NULL) {
		std::cout << "Node count: " << db->numNodes << std::endl;
		std::cout << "Edge count: " << db->numEdges << std::endl;
		std::cout << "Label count: " << db->numLabels << std::endl;
		std::cout << "Maximum nodes per label: " << db->maxLabelSize << std::endl;
	}
	else std::cout << "No graph data loaded to main memory\n";
}
//---------------------------------------------------------------------------
graph::GPGraph* db = NULL;
graph::GPGraph* dev_db = NULL;
graph::GPExpNode* exn = NULL;
graph::GPExpNode* dev_exn = NULL;

void deleteData() {
	if (db != NULL) { delete db; db = NULL; }
	if (dev_db != NULL) { delete dev_db; dev_db = NULL; }

	if (exn != NULL) { delete exn; exn = NULL; }
	if (dev_exn != NULL) { delete dev_exn; dev_exn = NULL; }
}
//---------------------------------------------------------------------------
graph::GPGraph* loadGraph(std::string line) {
	graph::GPGraph* db = NULL;
	
	std::istringstream iss(line);
	std::string com;
	std::string format;
	std::string file;

	if (!(iss >> com >> format >> file)) std::cout << "No input file or format specified; use 'help' for more information\n";
	else {
		if (format == "-t" || format == "-b") {
			timer.start();
			if (format == "-t") db = graph::readText(file.c_str());
			else db = graph::readBinary(file.c_str());
			timer.stop();

			if (db != NULL) {
				std::cout << "Loaded graph data in " << timer.getElapsedTimeInMilliSec() << " ms\n";
				showGraph(db);
			}
			else std::cout << "Could not load data from '" + file + "'\n";
		}
		else std::cout << "No graph format specified; use 'help' for more information\n";
	}

	return db;
}
//---------------------------------------------------------------------------
void loadDatabase(std::string line) {
	// clear all existing data
	std::cout << "Removing existing data graph ... ";
	deleteData();
	std::cout << "\n";

	// load graph to main memory
	db = loadGraph(line);

	if (db != NULL) {
		// find high-degree nodes from database graph
		exn = new graph::GPExpNode();
		exn->extract(db);

		timer.start();
		/*dev_exn = exn->copy(CopyType::HOST_TO_DEVICE);*/
		dev_exn = exn->copy(HOST_TO_DEVICE);
		/*dev_db = db->copy(CopyType::HOST_TO_DEVICE);*/
		dev_db = db->copy(HOST_TO_DEVICE);
		timer.stop();

		if (dev_db == NULL) std::cout << "Could not copy data to GPU; check device infomation\n";
		else std::cout << "Copied data to GPU memory in " << timer.getElapsedTimeInMilliSec() << " ms\n";
	}
}
//---------------------------------------------------------------------------
void executeQuery(std::string line) {
	if (db == NULL) {
		std::cout << "No database graph specified\n"; return;
	}

	// load a query graph
	graph::GPGraph* qr = loadGraph(line);
	if (qr == NULL) return;

	timer.start();
	// copy a query graph to device memory
	/*graph::GPGraph* dev_qr = qr->copy(CopyType::HOST_TO_DEVICE);*/
	graph::GPGraph* dev_qr = qr->copy(HOST_TO_DEVICE);
	
	// allocate intermediate data
	//TODO: match 


	timer.stop();

	delete dev_qr;
	delete qr;
}
//---------------------------------------------------------------------------
int main()
{
	std::vector<std::string> allCom = allowed();

	while (true) {
		std::cout << "gpsm> ";

		std::string line;
		getline(std::cin, line);

		if (line.empty()) continue;

		std::istringstream iss(line);
		std::string com;
		if (!(iss >> com)) continue;

		if (isAllowed(com, allCom) == false) {
			std::cout << "'" << com << "' is not recognized as an internal command; use 'help' for more information\n";
			continue;
		}

		// load database graph to main memory and device memory
		if (com == CMT_LOAD) loadDatabase(line);
		else if (com == CMT_EXEC) executeQuery(line);
		else if (com == CMT_INFO) showGraph(db);
		else if (com == CMT_CLEAR) deleteData();
		else if (com == CMT_EXIT) break;

		std::cout << std::endl;
	}

	deleteData();

	return 0;
}

