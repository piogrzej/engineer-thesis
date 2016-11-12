#include "mainkernels.h"
#include "helper_cuda.h"
#include "../utils/Timer.h"

floatingPoint countAvg(unsigned int output[],int ITER_NUM)
{
    floatingPoint out=0;

    for(unsigned int i=0; i<ITER_NUM;++i)
    {
        out += output[i];
    }

    return out/ITER_NUM;
}

floatingPoint getAvgPathLenCUDA(char* path, int ITER_NUM,int RECT_ID,bool measure)
{
    //tworzenie drzewa
    QuadTreeManager* qtm = randomWalkCudaInit(path,measure);
    //alokowanie pamieci na wynik
    unsigned int output[ITER_NUM];
    unsigned int* d_output;
    printf("Test: %s watkow: %d\n", path,ITER_NUM);
    unsigned int outputSize = ITER_NUM * sizeof(unsigned int);
    if(true==measure)
	{
		Timer::getInstance().start("_RandomWalkCuda Total");
	}
    checkCudaErrors(cudaMalloc((void **)&d_output,outputSize));
    randomWalkCudaWrapper(1,ITER_NUM,qtm,RECT_ID,d_output,time(NULL));
    checkCudaErrors(cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost));
    if(true==measure)
	{
		Timer::getInstance().stop("_RandomWalkCuda Total");
	}
    freeQuadTreeManager(qtm);
    cudaFree(d_output);
    cudaDeviceReset();
    return countAvg(output,ITER_NUM);
}
