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

floatingPoint getAvgPathLenCUDA(char* path, int ITER_NUM,int RECT_ID)
{
    //tworzenie drzewa
    QuadTreeManager* qtm = randomWalkCudaInit(path);
    //alokowanie pamieci na wynik
    unsigned int output[ITER_NUM];
    unsigned int* d_output;
    unsigned int outputSize = ITER_NUM * sizeof(unsigned int);
    cudaMalloc((void **)&d_output,outputSize);
    randomWalkCudaWrapper(1,ITER_NUM,qtm,RECT_ID,d_output,time(NULL));
    cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost);
    freeQuadTreeManager(qtm);
    cudaFree(d_output);
    cudaDeviceReset();

	//tworzenie drzewa

	QuadTreeManager* qtm = randomWalkCudaInit(path);
	for(int i=1; i<101; ++i)
	{
		//alokowanie pamieci na wynik
		unsigned int output[ITER_NUM];
		for(int j=0; j<ITER_NUM;++j){
			output[j]=0;
		}
		unsigned int* d_output;
		unsigned int outputSize = ITER_NUM * sizeof(unsigned int);
		cudaMalloc((void **)&d_output,outputSize);
		randomWalkCudaWrapper(1,ITER_NUM,qtm,RECT_ID,d_output,time(NULL));
		cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost);
		freeQuadTreeManager(qtm);
		cudaFree(d_output);

		printf("%d: %f\n",i,countAvg(output,ITER_NUM));
	}
    return 1;
}
