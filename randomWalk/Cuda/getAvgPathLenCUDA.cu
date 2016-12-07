#include "mainkernels.h"
#include "helper_cuda.h"
#include "../utils/Timer.h"
#include "../utils/RandGen.h"

#include <time.h>
#include <map>		//map
#include <string> 	//to_string
#include <iostream>	//ofstream

floatingPoint countAvg(unsigned int output[],int ITER_NUM)
{
    floatingPoint out=0;

    for(unsigned int i=0; i<ITER_NUM;++i)
    {
        out += output[i];
    }

    return out/ITER_NUM;
}

void saveOutput(d_Rect rectOutput[],int ITER_NUM)
{
	std::string timestamp = std::to_string((unsigned long)time(NULL));
	std::string filename = timestamp+"wyniki.txt";
	std::ofstream out(filename);
	out << "[GPU]\n";
	std::map<d_Rect,int> m;
	for(int i=0; i<ITER_NUM; ++i)
	{
		m[rectOutput[i]]++;
	}
	for(auto const &entry : m)
	{
		if(-1==entry.first.topLeft.x && -1== entry.first.topLeft.y && -1== entry.first.bottomRight.x && -1== entry.first.bottomRight.y)
		{
			out << "Ile scieżek wyszło poza granice warstwy: "<<(float)entry.second/(float)ITER_NUM*100 << "%\n";
		}
		else
		{
			out << entry.first.topLeft.x <<" "<< entry.first.topLeft.y <<" "<<
					entry.first.bottomRight.x<<" "<<entry.first.bottomRight.y <<": "<<
					((float)entry.second/(float)ITER_NUM)*100<<"% (" << entry.second <<")\n";
		}
	}
	printf("Wynik zapisany do %s\n",filename.c_str());
}

floatingPoint getAvgPathLenCUDA(char* path, int ITER_NUM,int RECT_ID,bool measure,int layerID)
{
    //tworzenie drzewa
    QuadTreeManager* qtm = randomWalkCudaInit(path,measure,RECT_ID,layerID);
    //alokowanie pamieci na wynik
    unsigned int output[ITER_NUM];
    unsigned int* d_output;
    d_Rect rectOutput[ITER_NUM];
    d_Rect* d_rectOutput;
    unsigned int outputSize = ITER_NUM * sizeof(unsigned int);
    unsigned int rectOutputSize = ITER_NUM * sizeof(d_Rect);
    if(true==measure)
	{
		Timer::getInstance().start("_RandomWalkCuda Total");
	}
    checkCudaErrors(cudaMalloc((void **)&d_output,outputSize));
    checkCudaErrors(cudaMalloc((void **)&d_rectOutput,rectOutputSize));
    randomWalkCudaWrapper(ITER_NUM,qtm,d_output,d_rectOutput,time(NULL));
    checkCudaErrors(cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(rectOutput,d_rectOutput,rectOutputSize,cudaMemcpyDeviceToHost));
    if(true==measure)
	{
		Timer::getInstance().stop("_RandomWalkCuda Total");
	}
    freeQuadTreeManager(qtm);
    cudaFree(d_output);
    cudaFree(d_rectOutput);
    cudaDeviceReset();
    saveOutput(rectOutput,ITER_NUM);
    return countAvg(output,ITER_NUM);
}
