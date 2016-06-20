#ifndef _GPSM_COMMON_H_
#define _GPSM_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iostream>
#include "cutil.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
	    }} while(0)

#define NP2(n)              do {                    \
    n--;                                            \
    n |= n >> 1;                                    \
    n |= n >> 2;                                    \
    n |= n >> 4;                                    \
    n |= n >> 8;                                    \
    n |= n >> 16;                                   \
    n ++; } while (0) 

#define FOR_LIMIT(i, len)			for(int i=0; i<len; i++)

#define FOR_LIMIT_REV(i, len)		for(int i=len-1; i>=0; i--)

#define FOR_RANGE(i, begin, end)		for(int i=begin; i<end; i++)

#define FOR_RANGE_REV(i, begin, end)	for(int i=end-1; i>=begin; i--)

#define FILL(arr, size, val)	for(int i=0; i< size; i++) arr[i] = val

enum {
	/* supported copy types */
	HOST_TO_DEVICE = 0,
	HOST_TO_HOST,
	DEVICE_TO_HOST
};

#endif