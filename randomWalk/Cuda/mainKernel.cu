

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

#include "../Parser.h"
#include "d_quadtree.h"
#include "params.h"

__global__ void createQuadTree(d_QuadTree* nodes, RectCuda** rects, Params* d_params);

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
        params.SHARED_MEM_SIZE = params.QUAD_TREE_CHILD_NUM * params.WARPS_PER_BLOCK * sizeof(int);

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

    RectCuda* rects = new RectCuda[layer.size()];
    RectCuda* d_rects[2]; // potrzebujemy dodatkowy buffor pomocniczy, dlatego 2
    for(int i = 0; i < layer.size(); i++)
      {
        RectHost const& r = layer[i];
        rects[i] = RectCuda(r.topLeft.x,r.bottomRight.x,r.topLeft.y,r.bottomRight.y);
      }
    int rectTableSize = sizeof(RectCuda)*layer.size();
    int nodesTableSize= sizeof(d_QuadTree)*params.MAX_NUM_NODES;
    d_QuadTree root(0,0,params.MAX_NUM_NODES);

    checkCudaErrors(cudaMalloc((void**)&d_rects[0],rectTableSize));
    checkCudaErrors(cudaMalloc((void**)&d_rects[1],rectTableSize));
    checkCudaErrors(cudaMalloc((void**)&d_nodes,nodesTableSize));
    checkCudaErrors(cudaMalloc((void**)&d_params,sizeof(Params)));
    checkCudaErrors(cudaMemcpy(d_rects[0],&rects,rectTableSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_nodes,&root,sizeof(d_QuadTree), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_params,&params,sizeof(Params), cudaMemcpyHostToDevice));

    printf("test %d\n",params.WARP_SIZE);
    createQuadTree<<<1,params.THREAD_PER_BLOCK,params.SHARED_MEM_SIZE>>>(d_nodes,d_rects,d_params);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&params,d_params,sizeof(Params), cudaMemcpyDeviceToHost));
    printf("test %d\n",params.WARP_SIZE);

}

__global__ void createQuadTree(d_QuadTree* nodes, RectCuda** rects, Params* params)
{
  extern __shared__ int sharedMemory[]; // trzymamy tu ilość rectów w danym węźle w każdym warpie
  const int warpId = threadIdx.x / warpSize; // Id warpa w bloku
  const int laneId = threadIdx.x % warpSize; // Id watku w warpie
  const int nodeId = blockIdx.x; // id node wzgledam parenta
  d_QuadTree &node = nodes[nodeId]; // 1 blok to jeden wezel
  int rectCount = node.rectCount();

  if(node.getLevel() >= params->MAX_LEVEL || rectCount <= params->MIN_RECT_IN_NODE) // dwa warunki zakonczenia albo okreslona ilosc poziomow albo satysfakcjonujace nas rozdrobnienie
    {
      if((node.getLevel() % 2) > 0) // jesli zakonczymy na nie parzystym levelu dobre posortowane recty beda w zlej tablicy ,trzeba skopiowac
	{
	  int end = node.endRectOff();
	  for(int it = node.startRectOff() + threadIdx.x; it < end ; it += params->THREAD_PER_BLOCK)
	    if(it < end)
	      rects[0] = rects[1];
	}
      return;
    }

  point2 center = node.getCenter();
  const RectCuda* roRects = rects[node.getLevel() % 2]; // read only rects
  RectCuda* sortedRects = rects[(node.getLevel() + 1) % 2]; //
  // Kazdy warp (32thready) bedzie wykonywany jednoczesnie, rozdzielamy na nasze warpy
  // robote po rowno
  int rectsPerWarp = (node.rectCount() + params->WARPS_PER_BLOCK - 1) / params->WARPS_PER_BLOCK;
  int nodeRangeBegin = node.startRectOff() + warpId * rectsPerWarp; // kazdy warp dostaje swoj przedzial rectow
  int nodeRangeEnd = min(nodeRangeBegin + rectsPerWarp,node.endRectOff()); // zeby nie przekroczyc swojego zakresu

  // przekonwertuj 1-d tablice do 2-d - latwiejsze operacje
  volatile int *rectsCountNode[NODES_NUMBER]; // volatile bo adresy do shared memory, ktore inne thready beda zmieniac
  for(int i = 0; i < params->QUAD_TREE_CHILD_NUM; ++i)
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
      RectCuda rect  = isActive ? roRects[it] : RectCuda(0.,0.,0.,0.); // jesli nieaktywny zerujemy zeby nam nic nie psul
      bool TLx = rect.topLeft.x    < center.x;
      bool TLy = rect.topLeft.y    < center.y;
      bool BRx = rect.bottmRight.x < center.x;
      bool BRy = rect.bottmRight.y < center.y;

#pragma unroll
      for(int i = 0; i < params->QUAD_TREE_CHILD_NUM; i++)// petla tylko do skrocenia kodu, ale unrollem preprocesor ja wyciagnie na normalny kod
	{
	  bool xMask = !(i % 2); // prawdziwe dla 0 , 2
	  bool yMask = i < 2;	// prawdziew dla 0 , 1
	  bool pred = (TLx && BRx && xMask) && ( TLy && BRy && yMask);

	  int rectsMatches = __popc(__ballot(isActive && pred));

	  if(rectsMatches > 0 && laneId == 0) // 1 watek dodaje wyniki calego warpa + optymalizacja
	      rectsCountNode[i] += rectsMatches;

	}
    }

  __syncthreads(); // czekamy, niech wszystkie chlopaki skoncza

  // Redukcja , typowa akcja do synchronizacji danych
  // Tylko QUAD_TREE_CHILD_NUM watkow mozemy zaangazowac max
  if(warpId < params->QUAD_TREE_CHILD_NUM)
    {
      int rectCount =  laneId < params->WARPS_PER_BLOCK ? rectsCountNode[warpId][laneId] : 0;

#pragma unroll
      for(int offset = 1; laneId < params->WARPS_PER_BLOCK; offset *= 2)
	{
	//  int countPerWarp = __shfl_up(rectCount, offset,params->WARPS_PER_BLOCK);

	 // if(laneId >= offset)
	 //     rectCount += countPerWarp;
	}
      if(laneId < params->WARPS_PER_BLOCK)
	rectsCountNode[warpId][laneId] = rectCount;
    }
  __syncthreads(); // czekamy, niech te 4 chlopaki skoncza



  if(warpId == 0)
    {
      int sum = rectsCountNode[NODE_ID::UP_LEFT][params->WARPS_PER_BLOCK - 1];
#pragma unroll
      for(int nodeId = 1; nodeId < params->WARPS_PER_BLOCK; ++nodeId)
	{
	  int tmp = rectsCountNode[nodeId][params->WARPS_PER_BLOCK -1 ];

	  if(laneId < params->WARPS_PER_BLOCK)
	    rectsCountNode[nodeId][laneId] += sum;
	  sum += tmp;
	}
    }
  __syncthreads();

  // Sorttowanie rectow,

  int laneMask = (1 << laneId) - 1; // maska np: laneId (0-32): dla id 6- maska: 111111B


  for(int it = nodeRangeBegin + laneId; __any(it < nodeRangeEnd) ; it += warpSize)
    {
      bool isActive = it < nodeRangeEnd; // pracujace tylko te ktore sa w zakresie jeszcze
      RectCuda rect  = isActive ? roRects[it] : RectCuda(0.,0.,0.,0.); // jesli nieaktywny zerujemy zeby nam nic nie psul
      bool TLx = rect.topLeft.x    < center.x;
      bool TLy = rect.topLeft.y    < center.y;
      bool BRx = rect.bottmRight.x < center.x;
      bool BRy = rect.bottmRight.y < center.y;

#pragma unroll
      for(int i = 0; i < params->QUAD_TREE_CHILD_NUM; i++) // petla tylko do skrocenia kodu, ale unrollem preprocesor ja wyciagnie na normalny kod
	{
	  bool xMask = !(i % 2); // prawdziwe dla 0 , 2
	  bool yMask = i < 2;	// prawdziew dla 0 , 1
	  bool pred = (TLx && BRx && xMask) && ( TLy && BRy && yMask);
	  int threadsResult = __ballot(pred);
	  int dest = rectsCountNode[i][warpId] + __popc(threadsResult & laneMask);// ballot zlicza nam z calego warp, a my chcemy tylko do konkretnego threada, stad maska

	  if(pred)
	    sortedRects[dest] = rect;

	  if(laneId == 0)
	    rectsCountNode[i][warpId] += __popc(threadsResult);
	}
    }

  __syncthreads();

  if(threadIdx.x == 0) // jeden blokowy watek ustala dzieci, ich indeksy, itd.
    {
        int nodesAtLevel = nodesCountAtLevel(node.getLevel());
	d_QuadTree* startNodeAtLevel = &nodes[nodesAtLevel]; // wskaznik na pierwszy wezel w tym poziomie
	int childCount = params->QUAD_TREE_CHILD_NUM;
	int childIndex = childCount * node.getId();
	const RectCuda& bounds = node.getBounds();

#pragma unroll
	for(int i = 0; i < params->QUAD_TREE_CHILD_NUM; i++)
	  {
	    startNodeAtLevel[childIndex + i].setId(childCount * node.getId() * 2 *i);
	    startNodeAtLevel[childIndex + i].setLevel(node.getLevel() + 1);
	    node.setChild(i,nodesAtLevel + childIndex + i);
	  }

	startNodeAtLevel[childIndex + NODE_ID::UP_LEFT].setLBounds(RectCuda(bounds.topLeft,center));
	startNodeAtLevel[childIndex + NODE_ID::DOWN_RIGHT].setLBounds(RectCuda(center,bounds.bottmRight));
	startNodeAtLevel[childIndex + NODE_ID::UP_RIGHT].setLBounds(RectCuda(center.x,bounds.bottmRight.x,
	                                                                     bounds.topLeft.y,center.y));
	startNodeAtLevel[childIndex + NODE_ID::DOWN_LEFT].setLBounds(RectCuda(bounds.topLeft.x,center.x,
	                                                                      center.y,bounds.bottmRight.y));

	startNodeAtLevel[childIndex + NODE_ID::UP_LEFT].setOff(node.startRectOff(),rectsCountNode[0][warpId]);
	startNodeAtLevel[childIndex + NODE_ID::UP_RIGHT].setOff(rectsCountNode[0][warpId],rectsCountNode[1][warpId]);
	startNodeAtLevel[childIndex + NODE_ID::DOWN_LEFT].setOff(rectsCountNode[1][warpId],rectsCountNode[2][warpId]);
	startNodeAtLevel[childIndex + NODE_ID::DOWN_RIGHT].setOff(rectsCountNode[2][warpId],rectsCountNode[3][warpId]);


	//createQuadTree<<<childCount,childCount * params->THREAD_PER_BLOCK ,
//	childCount * params->THREAD_PER_BLOCK * sizeof(int)>>>(nodes,rects,params);
    }
  // nie skonczone

  printf("warpId: %d  threadId: %d \n", threadIdx.x / params->WARP_SIZE, threadIdx.x);

}


/*
 *      /* int rectsMatches = __popc(__ballot(isActive && TLx && BRx && TLy && BRy));

      if(rectsMatches > 0 && laneId == 0) // 1 watek dodaje wyniki calego warpa + optymalizacja
	  rectsCountNode[NODE_ID::UP_LEFT] += rectsMatches;

      rectsMatches = __popc(__ballot(isActive && !TLx && !BRx && TLy && BRy)); // zlicza wszystkie watki ktore maja rect w tym sektorze

      if(rectsMatches > 0 && laneId == 0)
	  rectsCountNode[NODE_ID::UP_RIGHT] += rectsMatches;


      rectsMatches = __popc(__ballot(isActive && TLx && BRx && !TLy && !BRy));

      if(rectsMatches > 0 && laneId == 0)
	  rectsCountNode[NODE_ID::DOWN_LEFT] += rectsMatches;


       rectsMatches = __popc(__ballot(isActive && !TLx && !BRx && !TLy && !BRy));

      if(rectsMatches > 0 && laneId == 0)
	  rectsCountNode[NODE_ID::DOWN_RIGHT] += rectsMatches;*/

