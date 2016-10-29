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
#include "Parser.h"
#include "Logger.h"
#include "Timer.h"

#define NSAMPLE 200

// TO DO: brzydkie kopiowanie, trzeba poprawić
// TO DO: wykrywanie ilosci threadow, thread/block, (cudaDeviceProp)
QuadTreeManager* randomWalkCudaInit(char* path, int ITER_NUM, int RECT_ID)
{
    ErrorLogger::getInstance() >> "Random Walk CUDA\n";
    Timer::getInstance().start("Parser");
    Parser parser(path, "<<");
    const std::vector<d_Rect>& layer = parser.getLayerAt(0); // na razie 0 warstwa hardcode
    d_Rect const& spaceSize = parser.getLayerSize(0);
    Timer::getInstance().stop("Parser");
    QuadTreeManager* treeMng = createQuadTree(layer,spaceSize,false);


    Timer::getInstance().printResults();
    //tworzenie i kopiowanie intg do pamieci device
    REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE + 1];
    UINT32_t Nsample = NSAMPLE;
    precompute_unit_square_green(g,dgdx,dgdy,intg,NSAMPLE);
    int sizeOfIntg = (NSAMPLE + 1)*sizeof(REAL64_t);
    cudaMalloc((void **)&(treeMng->d_intg),sizeOfIntg);
    cudaMemcpy((treeMng->d_intg),intg,sizeOfIntg,cudaMemcpyHostToDevice);
    //--------------------------------------------
    return treeMng;
}

__device__ int getIndex(REAL64_t intg[NSAMPLE + 1], floatingPoint rand){
    for (int i = 0; i <= NSAMPLE; ++i)
    {
        if (intg[i] <= rand && intg[i + 1] > rand)
            return i;
    }
}

__global__ void randomWalkCuda(QuadTreeManager* quadTreeMn,int RECT_ID,unsigned int *output, unsigned int randomSeed=time(NULL))
{
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
    int index;
    bool isCollison;
    d_Rect square = quadTreeMn->root->createGaussianSurfFrom(R, 1.5);
    output[blockIdx.x]=0;

    bool broken = false;

    do
    {
        r = curand_uniform(&state);
        p = square.getPointFromNindex(getIndex(quadTree->d_intg, r), NSAMPLE);
        isCollison = quadTreeMn->root->checkCollisons(p, rectOutput);
        ++output[blockIdx.x];
    }
    while (false == isCollison);
}
