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

// TO DO: brzydkie kopiowanie, trzeba poprawić
// TO DO: wykrywanie ilosci threadow, thread/block, (cudaDeviceProp)
QuadTreeManager* randomWalkCudaInit(char* path,bool measure)
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
    const std::vector<d_Rect>& layer = parser.getLayerAt(0); // na razie 0 warstwa hardcode
    d_Rect const& spaceSize = parser.getLayerSize(0);

    QuadTreeManager* treeMng;
    if(true==measure)
	{
		Timer::getInstance().start("Create Tree");
		treeMng = createQuadTree(layer,spaceSize,true);
		Timer::getInstance().stop("Create Tree");
	}
    else
    {
    	treeMng = createQuadTree(layer,spaceSize,true);
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

__global__ void randomWalkCuda(QuadTreeManager* quadTreeMn,int RECT_ID,
        unsigned int *output,dTreePtr** stack,
        RandGen* gen,
        unsigned long long randomSeed=time(NULL))
{
    d_QuadTree* root = quadTreeMn->root;

	if(threadIdx.x == 0)
	{
	    gen->initPtrs();
		root->setStack(stack);
	}
	__syncthreads();
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
    bool isCollison;
    output[threadIdx.x]=0;
    root->createStack(threadIdx.x,quadTreeMn->maxlevel * 3 + 1); // tyle wystarczy do trawersowania po drzewie
    d_Rect start = quadTreeMn->rects[8];
    if(threadIdx.x == 0)
         printf("square: %f %f %f %f\n",start.topLeft.x,start.topLeft.y,
                                        start.bottomRight.x,start.bottomRight.y);

    d_Rect square = root->createGaussianSurfFrom(start, 1.5);

    do
    {
        r = curand_uniform(&state);
        p = square.getPointFromNindex(d_getIndex(quadTreeMn->d_intg,r), NSAMPLE);
        //printf("%d %d\n",(int)p.x,(int)p.y);
        if(false == root->isInBounds(p))
        {
            //broken = true;
            break;
        }
        square = root->drawBiggestSquareAtPoint(p);
        isCollison = root->checkCollisons(p, rectOutput);
        ++output[threadIdx.x];
    }
    while (false == isCollison);
    //if(threadIdx.x == 827)
    printf("%d square: %f %f %f %f\n",threadIdx.x,rectOutput.topLeft.x,rectOutput.topLeft.y,
                                       rectOutput.bottomRight.x,rectOutput.bottomRight.y);
    root->freeStack(threadIdx.x);

    if(threadIdx.x == 0)
        gen->freeStck();
}

void randomWalkCudaWrapper(int dimThread,QuadTreeManager* quadTree, int RECT_ID,unsigned int *output,RandGen &gen,unsigned long long randomSeed)
{
	dTreePtr** stack;
	RandGen* dGen;
    checkCudaErrors(cudaMalloc((void **)&dGen,sizeof(RandGen)));
    checkCudaErrors(cudaMalloc((void **)&stack,sizeof(dTreePtr**) * dimThread));
    checkCudaErrors(cudaMemcpy(dGen,&gen,sizeof(RandGen),cudaMemcpyHostToDevice));

    randomWalkCuda<<<1,dimThread>>>(quadTree,RECT_ID,output,stack,dGen,randomSeed);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(stack));
}
