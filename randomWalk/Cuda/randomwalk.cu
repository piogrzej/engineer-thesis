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

// TO DO: brzydkie kopiowanie, trzeba poprawić
// TO DO: wykrywanie ilosci threadow, thread/block, (cudaDeviceProp)
QuadTreeManager* randomWalkCudaInit(char* path,bool measure,int layer_id)
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
    const std::vector<d_Rect>& layer = parser.getLayerAt(layer_id); // na razie 0 warstwa hardcode
    d_Rect const& spaceSize = parser.getLayerSize(layer_id);

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

__global__ void randomWalkCuda(QuadTreeManager* quadTreeMn,int RECT_ID,unsigned int *output,dTreePtr** stack, unsigned long long randomSeed=time(NULL))
{
    d_QuadTree* root = quadTreeMn->root;

	if(threadIdx.x == 0)
		root->setStack(stack);

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
    d_Rect start = quadTreeMn->rects[RECT_ID];

    d_Rect square = root->createGaussianSurfFrom(quadTreeMn->rects[RECT_ID], 1.5);

    do
    {
        r = curand_uniform(&state);
        p = square.getPointFromNindex(d_getIndex(quadTreeMn->d_intg, r), NSAMPLE);
        if(false == root->isInBounds(p))
        {
            //broken = true;
            break;
        }
       // printf("square: %f %f %f %f\n",square.topLeft.x,square.topLeft.y,square.bottomRight.x,square.bottomRight.y);
        square = root->drawBiggestSquareAtPoint(p);
        isCollison = root->checkCollisons(p, rectOutput);
        if(!(rectOutput==quadTreeMn->rects[RECT_ID]))
        	output[threadIdx.x]=1;;
    }
    while (false == isCollison);
    root->freeStack(threadIdx.x);
}

void randomWalkCudaWrapper(int dimBlck,int dimThread,QuadTreeManager* quadTree, int RECT_ID,unsigned int *output,unsigned long long randomSeed)
{
	dTreePtr** stack;
    checkCudaErrors(cudaMalloc((void **)&stack,sizeof(dTreePtr**) * dimThread));
    randomWalkCuda<<<dimBlck,dimThread>>>(quadTree,RECT_ID,output,stack,randomSeed);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(stack));
}
