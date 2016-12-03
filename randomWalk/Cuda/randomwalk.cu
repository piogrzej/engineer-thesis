#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>

#include "mainkernels.h"
#include "d_parser.h"
#include "../utils/Logger.h"
#include "../utils/Timer.h"
#include "../utils/RandGen.h"

#define MAX_THREADS_BLOCK 1024

QuadTreeManager* randomWalkCudaInit(char* path,bool measure,int RECT_ID,int layerID)
{
	d_Parser parser("<<");
	if(true==measure)
	{
		TimeLogger::getInstance() << "RandomWalkCuda \nTest: " << path << "\n";
		Timer::getInstance().start("TotalTime");
		Timer::getInstance().start("deviceParser");
		parser.parse(path);
		Timer::getInstance().stop("deviceParser");
	}
	else
	{
		parser.parse(path);
	}
    const std::vector<d_Rect>& layer = parser.getLayerAt(layerID);
    d_Rect const& spaceSize = parser.getLayerSize(layerID);

    QuadTreeManager* treeMng;
    if(true==measure)
	{
		Timer::getInstance().start("Create Tree");
		treeMng = createQuadTree(layer,spaceSize,RECT_ID,true);
		Timer::getInstance().stop("Create Tree");
	}
    else
    {
    	treeMng = createQuadTree(layer,spaceSize,RECT_ID,true);
    }

    return treeMng;
}

__device__ int d_getIndex(REAL64_t intg[NSAMPLE + 1], floatingPoint rand){
    for (int i = 0; i <= NSAMPLE; ++i)
    {
        if (intg[i] <= rand && intg[i + 1] > rand)
            return i;
    }
    return -1;
}

__global__ void randomWalkCuda(QuadTreeManager* quadTreeMn,
        unsigned int *output,
        d_Rect* d_rectOutput,
        dTreePtr** stack,
        RandGen* gen,
        int threadInBlock,
        unsigned long long randomSeed=time(NULL))
{
	int id = (blockIdx.x * threadInBlock) + threadIdx.x;
    d_QuadTree* root = quadTreeMn->root;
    gen->initCudaPointers(id);
	root->setStack(stack);
	quadTreeMn->threadInBlock = threadInBlock;

	//printf("\t%d \n",threadIdx.x);
    /*inicjalizacja silnika random*/
    curandState_t state;
    curand_init(randomSeed*(threadIdx.x+1), /* the seed controls the sequence of random values that are produced */
            blockIdx.x, /* the sequence number is only important with multiple cores */
            0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            &state);
    //przyklad uzycia: printf("%f\n",curand_uniform (&state));, genruje float w zakresie od 0-1
    d_Rect rectOutput;
    point2 p;
    floatingPoint r;
    bool isCollison,broken=false;
    output[id] = 0;
    root->createStack(id,quadTreeMn->maxlevel * 3 + 1); // tyle wystarczy do trawersowania po drzewie
    d_Rect start = quadTreeMn->start;

    d_Rect square = root->createGaussianSurfFrom(start, 1.5);

    do
    {
    	int ind = gen->nextIndex(threadIdx.x);
        r = curand_uniform(&state);
        p = square.getPointFromNindex(ind, NSAMPLE);
       // printf("%f    %f   %d\n",p.x,p.y,ind);
        if(false == root->isInBounds(p))
        {
            broken = true;
            break;
        }
        square = root->drawBiggestSquareAtPoint(p);
        isCollison = root->checkCollisons(p, rectOutput);

        if(!(rectOutput == quadTreeMn->start))
        	output[id]=1;
    }
    while (false == isCollison);

    if(!(rectOutput == quadTreeMn->start))
    	output[id]=1;
    else
    	output[id]=0;

    if(false == broken)
    	d_rectOutput[id] = rectOutput;
    else
    	d_rectOutput[id] = d_Rect();

    root->freeStack(id);
}

void randomWalkCudaWrapper(int threads,QuadTreeManager* quadTree, unsigned int *output,d_Rect* d_rectOutput,RandGen &gen,unsigned long long randomSeed)
{
	dTreePtr** stack;
	RandGen* dGen;
    checkCudaErrors(cudaMalloc((void **)&(gen.indexPtrs),sizeof(int)* threads));
    checkCudaErrors(cudaMalloc((void **)&dGen,sizeof(RandGen)));
    checkCudaErrors(cudaMalloc((void **)&stack,sizeof(dTreePtr**) * threads + 1));
    checkCudaErrors(cudaMemcpy(dGen,&gen,sizeof(RandGen),cudaMemcpyHostToDevice));


    int blockCount     = (int)ceil(threads / double(MAX_THREADS_BLOCK));
    int threadsInBlock =  threads/ blockCount;

    randomWalkCuda<<<blockCount,threadsInBlock>>>(quadTree,output,d_rectOutput,stack,dGen,threadsInBlock,randomSeed);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(stack));
}
