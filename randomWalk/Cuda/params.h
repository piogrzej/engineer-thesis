/*
 * params.h
 *
 *  Created on: 26 wrz 2016
 *      Author: mknap
 */

#ifndef PARAMS_H_
#define PARAMS_H_

#include "cuda_runtime.h"

#include <math.h>

#define NODES_NUMBER 4

__device__ __host__ int nodesCountAtLevel(int level)
{
  return int((1 - powf(NODES_NUMBER,level)) /
	 (1 - level));
}

struct Params
{
  //CUDA PARAMS
        const int THREAD_PER_BLOCK = 128;
	      int WARP_SIZE; // inicjalizowane przez cudaInit
	      int WARPS_PER_BLOCK;
	      int SHARED_MEM_SIZE;

 // RANDOM WALK params

    	const int QUAD_TREE_CHILD_NUM = NODES_NUMBER;
    	const int MAX_LEVEL = 8;
    	const int MIN_RECT_IN_NODE = 16;
    	const int MAX_NUM_NODES = (1 - pow(QUAD_TREE_CHILD_NUM,MAX_LEVEL)) /
			          (1 - QUAD_TREE_CHILD_NUM);

/*
 * MIN_RECT_IN_NODE - inicjalnie root ma wszystkie recty rozdysponowuje je do osiagniecia tej liczby
 * MAX_DEPTH - jeden z warunków zakończenia rekurencji tworzenia drzewa,
 * 	       drzewo nie bedzie miało więcej poziomów niż ta liczba
 * MAX_NUM_NODES - z góry możemy wyliczyć maksymalną ilość węzłów QuadTree
 * 	           Liczba wszystkich wezlow Suma:4^n (c.g. ,n-głebokość drzewa)
 *  	           Dla MAX_DEPTH = 8 wynosi 21845
 */
} params;

#endif /* PARAMS_H_ */
