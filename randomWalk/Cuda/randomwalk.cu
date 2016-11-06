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
QuadTreeManager* randomWalkCudaInit(char* path)
{
    ErrorLogger::getInstance() >> "Random Walk CUDA\n";
    Timer::getInstance().start("Parser");
    d_Parser parser("<<");
    parser.parse(path);
    const std::vector<d_Rect>& layer = parser.getLayerAt(0); // na razie 0 warstwa hardcode
    d_Rect const& spaceSize = parser.getLayerSize(0);
    Timer::getInstance().stop("Parser");
    QuadTreeManager* treeMng = createQuadTree(layer,spaceSize,false);

    Timer::getInstance().printResults();
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

__global__ void randomWalkCuda(QuadTreeManager* quadTreeMn,int RECT_ID,unsigned int *output, unsigned int randomSeed=time(NULL))
{
	//printf("\t%d \n",threadIdx.x);
	printf("printf w randomWalkCuda id watku:%d \n",threadIdx.x);
    /*inicjalizacja silnika random*/
    curandState_t state;
    curand_init(randomSeed, /* the seed controls the sequence of random values that are produced */
            blockIdx.x, /* the sequence number is only important with multiple cores */
            0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            &state);
    //przyklad uzycia: printf("%f\n",curand_uniform (&state));, genruje float w zakresie od 0-1
    d_Rect rectOutput;
    point2 p;
    floatingPoint r;
    bool isCollison;
    output[threadIdx.x]=0;
    printf("\t%d rects count: %d \n",threadIdx.x,quadTreeMn->root->rectCount());
    d_Rect temp;
    p.x=10;
    p.y=10;
    quadTreeMn->root->checkCollisons(p,temp);
    //printf("\t%d rects count: %d \n",threadIdx.x,quadTreeMn->rectsCount);
    //d_Rect startRect = quadTreeMn->rects[RECT_ID];
    //printf("\t\t%d createGuassianSquere: %f,%f %f,%f\n",threadIdx.x,startRect.topLeft.x,startRect.topLeft.y,startRect.bottomRight.x,startRect.bottomRight.y);
    d_Rect square = quadTreeMn->root->createGaussianSurfFrom(quadTreeMn->rects[RECT_ID], 1.5);
    printf("\t\t%d createGuassianSquere: %f,%f %f,%f\n",threadIdx.x,square.topLeft.x,square.topLeft.y,square.bottomRight.x,square.bottomRight.y);
    /*
    do
    {
        r = curand_uniform(&state);
        p = square.getPointFromNindex(d_getIndex(quadTreeMn->d_intg, r), NSAMPLE);
        if(false == quadTreeMn->root->isInBounds(p))
        {
            //broken = true;
            break;
        }
        square = quadTreeMn->root->drawBiggestSquareAtPoint(p);
        isCollison = quadTreeMn->root->checkCollisons(p, rectOutput);
        ++output[threadIdx.x];
    }
    while (false == isCollison);*/
}

void randomWalkCudaWrapper(int dimBlck,int dimThread,QuadTreeManager* quadTree, int RECT_ID,unsigned int *output,unsigned int randomSeed)
{
    randomWalkCuda<<<dimBlck,dimThread>>>(quadTree,RECT_ID,output,randomSeed);
}
