
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Parser.h"
#include "d_quadtree.h"
#include "params.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <vector>

__global__ void createQuadTree(d_QuadTree* nodes, RectCuda* rects, Params* d_params);

bool initCuda(int argc, char **argv)
{

    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return false;
    }
    int cuda_device = findCudaDevice(argc, (const char **)argv);
        cudaDeviceProp deviceProps;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
        params.WARP_SIZE = deviceProps.warpSize;
        params.WARPS_PER_BLOCK = params.THREAD_PER_BLOCK / params.WARP_SIZE;
        return true;

    int cdpCapable = (deviceProps.major == 3 && deviceProps.minor >= 5) || deviceProps.major >=4;

    printf("GPU: %s ma (SM %d.%d)\n", deviceProps.name, deviceProps.major, deviceProps.minor);

    if (!cdpCapable)
    {
        std::cerr << "RandomWalk potrzebuje SM 3.5 lub wyzszej do CUDA Dynamic Parallelism.\n" << std::endl;
        return false;
    }
    return true;
}

// TO DO: brzydkie kopiowanie, trzeba poprawić
// TO DO: wykrywanie ilosci threadow, thread/block, (cudaDeviceProp)
void randomWalkCUDA(char* path, int ITER_NUM, int RECT_ID)
{
    Params* d_params;
    d_QuadTree* d_nodes;
    Parser parser(path, "<<");
    const std::vector<RectHost>& layer = parser.getLayerAt(0); // na razie 0 warstwa hardcode
    RectHost const& spaceSize = parser.getLayerSize(0);

    RectCuda rects[layer.size()];
    RectCuda* d_rects;
    for(int i = 0; i < layer.size(); i++)
      {
        RectHost const& r = layer[i];
        rects[i] = RectCuda(r.topLeft.x,r.bottomRight.x,r.topLeft.y,r.bottomRight.y);
      }
    int rectTableSize = sizeof(RectCuda)*layer.size();
    int nodesTableSize= sizeof(d_QuadTree)*params.MAX_NUM_NODES;
    int sharedMemorySize = params.QUAD_TREE_CHILD_NUM * params.WARPS_PER_BLOCK * sizeof(int);
    d_QuadTree root(0,0,params.MAX_NUM_NODES);

    checkCudaErrors(cudaMalloc((void**)&d_rects,rectTableSize));
    checkCudaErrors(cudaMalloc((void**)&d_nodes,nodesTableSize));
    checkCudaErrors(cudaMalloc((void**)&d_params,sizeof(Params)));
    checkCudaErrors(cudaMemcpy(d_rects,&rects,rectTableSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_nodes,&root,sizeof(d_QuadTree), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_params,&params,sizeof(Params), cudaMemcpyHostToDevice));

    printf("test %d\n",params.WARP_SIZE);
    createQuadTree<<<1,params.THREAD_PER_BLOCK,sharedMemorySize>>>(d_nodes,d_rects,d_params);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&params,d_params,sizeof(Params), cudaMemcpyDeviceToHost));
    printf("test %d\n",params.WARP_SIZE);

}

__global__ void createQuadTree(d_QuadTree* nodes, RectCuda* rects, Params* params)
{
  extern __shared__ int sharedMemory[]; // trzymamy tu ilość rectów w danym węźle w każdym warpie
  const int warpId = threadIdx.x / warpSize; // Id warpa w bloku
  const int laneId = threadIdx.x & warpSize; // Id watku w warpie
  const int nodeId = blockIdx.x; // id node wzgledam parenta
  d_QuadTree &node = nodes[nodeId]; // 1 blok to jeden wezel
  int rectCount = node.rectCount();
  point2 center = node.getCenter();
  // Kazdy warp (32thready) bedzie wykonywany jednoczesnie, rozdzielamy na nasze warpy
  // robote po rowno
  int rectsPerWarp = (node.rectCount() + params->WARPS_PER_BLOCK - 1) / params->WARPS_PER_BLOCK;
  int nodeRangeBegin = node.startRectOff() + warpId * rectsPerWarp; // kazdy warp dostaje swoj przedzial rectow
  int nodeRangeEnd = min(nodeRangeBegin + rectsPerWarp,node.endRectOff()); // zeby nie przekroczyc swojego zakresu

  // przekonwertuj 1-d tablice do 2-d - latwiejsze operacje
  volatile int *rectsCountNode[4]; // volatile bo adresy do shared memory, ktore inne thready beda zmieniac
  for(int i = 0; i < d_params->QUAD_TREE_CHILD_NUM; ++i)
    {
      rectsCountNode[i] = (volatile int*) &sharedMemory[i * params->WARPS_PER_BLOCK];
    }

  if(laneId == 0) // czyscimy śmieci po ostatnich wywolaniach
    {
#pragma unroll
      for(int i = 0; i < params->QUAD_TREE_CHILD_NUM; ++i)
	rectsCountNode[i][warpId] = 0;
    }

  //Liczymy ilosc rectow w kazdym wezle-dziecku wszystkimi dostepnymi watkami,a co
  //kilka cudowych funkcji, ciekawe sa: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-vote-functions
  // for chodzi dopoki sa jakies aktywne thready, sprawdzamy 32 recty jednoczesnie na jednym SM,
  for(int it = nodeRangeBegin + laneId; __any(it < nodeRangeEnd) ; it += warpSize)
    {
      bool isActive = it < nodeRangeEnd;
      RectCuda rect  = isActive ? rects[it] : Rect(0.,0.,0.,0.); // jesli nieaktywny zerujemy zeby nam nic nie psul
      int rectsMatches = __popc(__ballot(isActive && rect.topLeft.x < center.x &&
						     rect.bottmRight.x < center.x &&
						     rect.topLeft.y < center.y &&
						     rect.bottmRight.y < center.y));

      if(rectsMatches > 0 && laneId == 0) // 1 watek dodaje wyniki calego warpa + optymalizacja
	{
	  rectsCountNode[NODE_ID::UP_LEFT] += rectsMatches;
	}

      rectsMatches = __popc(__ballot(isActive && rect.topLeft.x >= center.x &&
						     rect.bottmRight.x >= center.x &&
						     rect.topLeft.y < center.y &&
						     rect.bottmRight.y < center.y));

      if(rectsMatches > 0 && laneId == 0)
	{
	  rectsCountNode[NODE_ID::UP_RIGHT] += rectsMatches;
	}

      rectsMatches = __popc(__ballot(isActive && rect.topLeft.x < center.x &&
						     rect.bottmRight.x < center.x &&
						     rect.topLeft.y >= center.y &&
						     rect.bottmRight.y >= center.y));

      if(rectsMatches > 0 && laneId == 0) // 1 watek dodaje wyniki calego warpa + optymalizacja
	{
	  rectsCountNode[NODE_ID::DOWN_LEFT] += rectsMatches;
	}
       rectsMatches = __popc(__ballot(isActive && rect.topLeft.x >= center.x &&
						     rect.bottmRight.x >= center.x &&
						     rect.topLeft.y >= center.y &&
						     rect.bottmRight.y >= center.y));

      if(rectsMatches > 0 && laneId == 0) // 1 watek dodaje wyniki calego warpa + optymalizacja
	{
	  rectsCountNode[NODE_ID::DOWN_RIGHT] += rectsMatches;
	}
    }

  __syncthreads(); // czekamy aż inne warpy skoncza


  // nie skoczone

  printf("warpId: %d  threadId: %d \n", threadIdx.x / params->WARP_SIZE, threadIdx.x);


}
