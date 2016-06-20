
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util\gpgraph.cuh"
#include <iostream>

using namespace gpsm;

int main()
{
	GPGraph<int,int,int> graph;
	graph.ReadBinary("G:\\data\\data.dat");
	graph.Print();

	return 0;
}