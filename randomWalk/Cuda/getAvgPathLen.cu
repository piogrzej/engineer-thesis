#include "mainkernels.h"

floatingPoint countAvg(unsigned int output[],int ITER_NUM)
{
    floatingPoint out=0;

    for(unsigned int i=0; i<ITER_NUM;++i)
    {
        out += output[i];
    }

    return out/ITER_NUM;
}

floatingPoint getAvgPathLen(char* path, int ITER_NUM,int RECT_ID)
{
    //tworzenie drzewa
    QuadTreeManager* qtm = randomWalkCudaInit(path);
    //alokowanie pamieci na wynik
    unsigned int output[ITER_NUM];
    unsigned int* d_output;
    unsigned int outputSize = ITER_NUM * sizeof(unsigned int);
    cudaMalloc((void **)&d_output,outputSize);
    randomWalkCudaWrapper(ITER_NUM,1,qtm,RECT_ID,d_output,time(NULL));
    cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost);
    freeQuadTreeManager(qtm);
    cudaFree(d_output);

    return countAvg(output,ITER_NUM);
}
