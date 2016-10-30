#include "mainkernels.h"


floatingPoint countAvg(floatingPoint output[],int ITER_NUM)
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
    floatingPoint output[ITER_NUM];
    floatingPoint* d_output;
    unsigned int outputSize = ITER_NUM * sizeof(floatingPoint);
    cudaMalloc((void **)&d_output,outputSize);
    randomWalkCudaWrapper(ITER_NUM,1,qtm,RECT_ID,d_output,time(NULL));
    cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost);
    //sprzatanie
    freeQuadTreeManager(qtm);
    free(d_output);

    return countAvg(output,ITER_NUM);
}
