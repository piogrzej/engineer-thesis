#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>

#include "helper_cuda.h"
#include "mainkernels.h"
#include "../utils/Logger.h"
#include "../utils/Timer.h"
#include "../green/green.h"


__global__ void createQuadTreeKernel(d_QuadTree* nodes, d_Rect* rects, Params* d_params,int lvlMul);
bool checkQuadTree(const d_QuadTree *nodes,int idx,d_Rect *rects, int& count);

QuadTreeManager* createQuadTree(const std::vector<d_Rect>& layer,d_Rect const& spaceSize,bool doCheck)
{
  Params params;
  params.WARP_SIZE = 32;
  params.WARPS_PER_BLOCK = params.THREAD_PER_BLOCK / params.WARP_SIZE;
  params.SHARED_MEM_SIZE = (params.QUAD_TREE_CHILD_NUM + 1) * params.WARPS_PER_BLOCK * sizeof(int); // dzieci i rodzic

  Timer::getInstance().start("Kopiowanie zasobów do CUDA");
  d_Rect* d_rects;
  d_QuadTree* nodes, *d_nodes;
  Params* d_params;
  params.TOTAL_RECT = layer.size();
 /* for(int i = 0; i < params.TOTAL_RECT; i++)
    {
	  printf("rect: %d %d\n",(int)layer[i].topLeft.x,(int)layer[i].topLeft.y);
    }*/
  size_t rectTableSize = sizeof(d_Rect)*layer.size();
  size_t nodesTableSize= sizeof(d_QuadTree) * params.MAX_NUM_NODES;
  d_QuadTree root(0,0,params.TOTAL_RECT);
  nodes = (d_QuadTree*)malloc(nodesTableSize);
  root.setBounds(spaceSize);
  checkCudaErrors(cudaMalloc((void**)&d_rects,rectTableSize * 2)); // potrzebujemy bufora dlatego 2
  checkCudaErrors(cudaMalloc((void**)&d_nodes,nodesTableSize));
  checkCudaErrors(cudaMalloc((void**)&d_params,sizeof(Params)));
  checkCudaErrors(cudaMemcpy(d_rects,&layer.front(),rectTableSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_nodes,&root,sizeof(d_QuadTree), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_params,&params,sizeof(Params), cudaMemcpyHostToDevice));
  Timer::getInstance().stop("Kopiowanie zasobów do CUDA");

  Timer::getInstance().start("Create Tree CUDA");
  createQuadTreeKernel<<<1,params.THREAD_PER_BLOCK,params.SHARED_MEM_SIZE>>>(d_nodes,d_rects,d_params,0);
  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();
  Timer::getInstance().stop("Create Tree CUDA");
 // ErrorLogger::getInstance() >> "Tworzenie drzewa zakończone pomyślnie\n";

  if(doCheck)
  {
      ErrorLogger::getInstance() >> "Sprawdzanie drzewa. \n";// Max nodes: " >> params.MAX_NUM_NODES >> "\n";

      d_Rect* rects = (d_Rect*)malloc(sizeof(d_Rect)*params.TOTAL_RECT);
      checkCudaErrors(cudaMemcpy(nodes,d_nodes,nodesTableSize, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(&params,d_params,sizeof(Params), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(rects,d_rects,rectTableSize, cudaMemcpyDeviceToHost));
      int count;
      bool result = checkQuadTree(nodes,0,rects,count);
      result ? ErrorLogger::getInstance() >> "Stworzono pomyślnie\n":
               ErrorLogger::getInstance() >> "Błąd tworzenia drzewa\n";
  }
  QuadTreeManager* qm = new QuadTreeManager();
  QuadTreeManager* d_tree;

  qm->nodes = d_nodes;
  qm->root = d_nodes;
  qm->rects = d_rects;
  qm->rectsCount = layer.size();
/*
  //tworzenie i kopiowanie intg do pamieci device
  REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE + 1];
  precompute_unit_square_green(g,dgdx,dgdy,intg,NSAMPLE);
  int sizeOfIntg = (NSAMPLE + 1)*sizeof(REAL64_t);
  REAL64_t *d_intg;
  cudaMalloc((void **)&d_intg,sizeOfIntg);
  cudaMemcpy(d_intg,intg,sizeOfIntg,cudaMemcpyHostToDevice);
  qm->d_intg = d_intg;
  cudaMalloc((void**)d_tree,sizeof(QuadTreeManager));
  cudaMemcpy(d_tree,qm,sizeof(QuadTreeManager),cudaMemcpyHostToDevice);
  delete qm;*/

  return d_tree;
}

/* Ogólny flow:
 * 1. Sprawdzamy czy warunki końcowe nie są spełnione, jeśli tak to kopiujemy jesli trzeba wszystko koncowej tablicy rectow
 * 2. Zliczamy ile rectow pasuje do ktorego wezla
 * 3. Łączymy wyniki wszystkich warpow
 * 4. Wyliczamy pozycje kazdego recta, tak aby znajdowal sie w przedziale swojego wezla
 * 5. Tworzymy wezly dzieci, wyznaczamy ich przedzialy w tablicy rectow
 * 6. Wywolujemy funkcje rekurencyjna dla nowo powstalych wezlow
*/
__global__ void createQuadTreeKernel(d_QuadTree* nodes, d_Rect* rects, Params* params,int lvlMul)
{
  extern __shared__ int sharedMemory[]; // trzymamy tu ilość rectów w danym węźle w każdym warpie
  const int warpId = threadIdx.x / warpSize; // Id warpa w bloku
  const int laneId = threadIdx.x % warpSize; // Id watku w warpie
  const int nodeId = blockIdx.x + lvlMul * NODES_NUMBER; // id node wzgledam parenta
 // if(threadIdx.x == 0)
 //     printf("nodId: %d\n",nodeId);
  d_QuadTree &node = nodes[nodeId]; // 1 blok to jeden wezel
  int rectCount = node.rectCount();
  d_Rect* buffer[2]; buffer[node.getLevel() % 2] = rects; buffer[(node.getLevel() + 1) % 2] = &rects[params->TOTAL_RECT];
  const d_Rect* roRects = buffer[0]; // read only rects
  d_Rect* sortedRects = buffer[1];

  //if(threadIdx.x == 0)
  //printf("block %d id: %d node: %d %d %d %d lvl: %d count %d\n", nodeId,node.getId(),(int)node.getBounds().topLeft.x,(int)node.getBounds().topLeft.y,
		//			  (int)node.getBounds().bottomRight.x,(int)node.getBounds().bottomRight.y,node.getLevel(),node.rectCount());


  if(node.getLevel() >= params->MAX_LEVEL || rectCount <= params->MIN_RECT_IN_NODE) // dwa warunki zakonczenia albo okreslona ilosc poziomow albo satysfakcjonujace nas rozdrobnienie
    {
      if((node.getLevel() % 2) != 0) // jesli zakonczymy na nie parzystym levelu dobre posortowane recty beda w zlej tablicy ,trzeba skopiowac
	    {
           int it = node.startRectOff(), end = node.endRectOff();
           int total = params->TOTAL_RECT;
           int threadsNum = params->THREAD_PER_BLOCK;
           for (it += threadIdx.x ; it < end ; it += threadsNum)
             {
               if (it < end)
                   rects[it] = rects[total + it];
             }
    	}
      //if(threadIdx.x == 0)
        //    printf("koniec id %d    s  %d e  %d\n",node.getId(),node.startRectOff(),node.endRectOff());
      return;
    }

  float2 center = node.getCenter();

  // Kazdy warp (32thready) bedzie wykonywany jednoczesnie, rozdzielamy na nasze warpy
  // robote po rowno
  int rectsPerWarp = max(params->WARP_SIZE,(node.rectCount() + params->WARPS_PER_BLOCK - 1) / params->WARPS_PER_BLOCK);
  int nodeRangeBegin = node.startRectOff() + warpId * rectsPerWarp; // kazdy warp dostaje swoj przedzial rectow
  int nodeRangeEnd = min(nodeRangeBegin + rectsPerWarp,node.endRectOff()); // zeby nie przekroczyc swojego zakresu

  // przekonwertuj 1-d tablice do 2-d - latwiejsze operacje
  volatile int *rectsCountNode[NODES_NUMBER + 1]; // volatile bo adresy do shared memory, ktore inne thready beda zmieniac
  for(int i = 0; i < params->QUAD_TREE_CHILD_NUM + 1; ++i)
    {
      rectsCountNode[i] = (volatile int*) &sharedMemory[i * params->WARPS_PER_BLOCK];
    }
  if( laneId == 0) // czyscimy śmieci po ostatnich wywolaniach
    {
#pragma unroll
      for(int i = 0; i < params->QUAD_TREE_CHILD_NUM + 1; ++i)
	rectsCountNode[i][warpId] = 0;
    }

    __syncthreads();
  //Liczymy ilosc rectow w kazdym wezle-dziecku wszystkimi dostepnymi watkami,a co
  //kilka cudowych funkcji, ciekawe sa: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-vote-functions
  // for chodzi dopoki sa jakies aktywne thready, sprawdzamy 32 recty jednoczesnie na jednym SM,
  for(int it = nodeRangeBegin + laneId; __any(it < nodeRangeEnd) ; it += warpSize)
    {
      bool isActive = it < nodeRangeEnd;
      d_Rect rect  = isActive ? roRects[it] : d_Rect(0.,0.,0.,0.); // jesli nieaktywny zerujemy zeby nam nic nie psul
      bool TLx = rect.topLeft.x    < center.x;
      bool TLy = rect.topLeft.y    < center.y;
      bool BRx = rect.bottomRight.x < center.x;
      bool BRy = rect.bottomRight.y < center.y;

      int rectsMatches = __popc(__ballot(isActive && TLx && BRx && TLy && BRy));

      if(rectsMatches > 0 && laneId == 0) // 1 watek dodaje wyniki calego warpa + optymalizacja
	rectsCountNode[NODE_ID::UP_LEFT][warpId] += rectsMatches;

      rectsMatches = __popc(__ballot(isActive && !TLx && !BRx && TLy && BRy)); // zlicza wszystkie watki ktore maja rect w tym sektorze

      if(rectsMatches > 0 && laneId == 0)
	rectsCountNode[NODE_ID::UP_RIGHT][warpId] += rectsMatches;


      rectsMatches = __popc(__ballot(isActive && TLx && BRx && !TLy && !BRy));

      if(rectsMatches > 0 && laneId == 0)
	rectsCountNode[NODE_ID::DOWN_LEFT][warpId] += rectsMatches;


       rectsMatches = __popc(__ballot(isActive && !TLx && !BRx && !TLy && !BRy));

      if(rectsMatches > 0 && laneId == 0)
	rectsCountNode[NODE_ID::DOWN_RIGHT][warpId] += rectsMatches;
    }

  __syncthreads(); // czekamy, niech wszystkie chlopaki skoncza
  // Redukcja , typowa akcja do synchronizacji danych między wrapami
  // Tylko QUAD_TREE_CHILD_NUM watkow mozemy zaangazowac max
  if(warpId < params->QUAD_TREE_CHILD_NUM)
    {
      int rectCount =  laneId < params->WARPS_PER_BLOCK ? rectsCountNode[warpId][laneId] : 0;

#pragma unroll
      for(int offset = 1; offset < params->WARPS_PER_BLOCK; offset *= 2)
	{
	  int countPerWarp = __shfl_up(rectCount, offset,params->WARPS_PER_BLOCK);

	  if(laneId >= offset)
	     rectCount += countPerWarp;
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
	 rectsCountNode[params->WARPS_PER_BLOCK][laneId] = sum;
    }
  __syncthreads();

// zapisujemy ilosc miejsc dla kazdego z dzieci noda wraz z jego startowym offsetem
  if(threadIdx.x < (NODES_NUMBER +  1) * params->WARPS_PER_BLOCK)
    {
      int val = threadIdx.x == 0 ? 0 : sharedMemory[threadIdx.x - 1];
      val += node.startRectOff();
      sharedMemory[threadIdx.x] = val;
    }
/*
  __syncthreads();
    if(threadIdx.x == 0)
      {
	for(int i = 0; i < 5; ++i)
	  {
	    for(int j = 0; j < 4; ++j)
	      {
		printf("%d ",rectsCountNode[i][j]);
	      }
	    printf("\n");
	  }

     }
    __syncthreads();*/

  // Sorttowanie rectow, przenosimy recty tak aby zanajdowaly sie w zakresie danego noda

  int laneMask = (1 << laneId) - 1; // maska np: laneId (0-32): dla id 6- maska: 111111B

  for(int it = nodeRangeBegin + laneId; __any(it < nodeRangeEnd) ; it += warpSize)
    {
      bool isActive = it < nodeRangeEnd; // pracujace tylko te ktore sa w zakresie jeszcze
      d_Rect rect  = isActive ? roRects[it] : d_Rect(0.,0.,0.,0.); // jesli nieaktywny zerujemy zeby nam nic nie psul
      bool TLx = rect.topLeft.x    < center.x;
      bool TLy = rect.topLeft.y    < center.y;
      bool BRx = rect.bottomRight.x < center.x;
      bool BRy = rect.bottomRight.y < center.y;

      bool pred = isActive && TLx && BRx  &&  TLy && BRy;
      int threadsResult = __ballot(pred);
      int dest = rectsCountNode[NODE_ID::UP_LEFT][warpId] +
		__popc(threadsResult & laneMask);// ballot zlicza nam z calego warp, a my chcemy tylko do konkretnego threada, stad maska

      if(pred)
	{
	  //printf("0 dest: %d rect: %d %d\n",dest,(int)rect.topLeft.x,(int)rect.topLeft.y);
	sortedRects[dest] = rect;
	}
      if(laneId == 0)
	rectsCountNode[NODE_ID::UP_LEFT][warpId] += __popc(threadsResult);

      pred = isActive && !TLx && !BRx  &&  TLy && BRy;
      threadsResult = __ballot(pred);
      dest = rectsCountNode[NODE_ID::UP_RIGHT][warpId] +
		__popc(threadsResult & laneMask);// ballot zlicza nam z calego warp, a my chcemy tylko do konkretnego threada, stad maska

      if(pred)
	{
	 // printf("1 dest: %d rect: %d %d\n",dest,(int)rect.topLeft.x,(int)rect.topLeft.y);
	         sortedRects[dest] = rect;
	}
      if(laneId == 0)
	rectsCountNode[NODE_ID::UP_RIGHT][warpId] += __popc(threadsResult);


      pred = isActive && TLx && BRx  &&  !TLy && !BRy;
      threadsResult = __ballot(pred);
      dest = rectsCountNode[NODE_ID::DOWN_LEFT][warpId] +
		__popc(threadsResult & laneMask);// ballot zlicza nam z calego warp, a my chcemy tylko do konkretnego threada, stad maska

      if(pred)
	{
	  //printf("2 dest: %d rect: %d %d\n",dest,(int)rect.topLeft.x,(int)rect.topLeft.y);
	sortedRects[dest] = rect;
	}

      if(laneId == 0)
	{
	  rectsCountNode[NODE_ID::DOWN_LEFT][warpId] += __popc(threadsResult);
	}


      pred =  isActive && !TLx && !BRx  &&  !TLy && !BRy;
      threadsResult = __ballot(pred);
      dest = rectsCountNode[NODE_ID::DOWN_RIGHT][warpId] +
		__popc(threadsResult & laneMask);// ballot zlicza nam z calego warp, a my chcemy tylko do konkretnego threada, stad maska

      if(pred)
	{
	  //printf("3 dest: %d rect: %d %d\n",dest,(int)rect.topLeft.x,(int)rect.topLeft.y);
	  sortedRects[dest] = rect;
	}

      if(laneId == 0)
	rectsCountNode[NODE_ID::DOWN_RIGHT][warpId] += __popc(threadsResult);

      //wszystkie inne ktore nie pasuja do zadnego z powyzszych zostawiamy u rodzica
      pred = (TLx && !BRx) || (TLy && !BRy);
      threadsResult = __ballot(pred && isActive);
      dest = rectsCountNode[4][warpId] + __popc(threadsResult & laneMask);

      if(pred && isActive)
	{
	 // printf("4 dest: %d rect: %d %d\n",(int)dest,(int)rect.topLeft.x,(int)rect.topLeft.y);
	  sortedRects[dest] = rect;
	}

      if(laneId == 0)
	rectsCountNode[4][warpId] += __popc(threadsResult);
    }
   // if(threadIdx.x == 0)
	//printf("center %d %d \n",(int)center.x,(int)center.y);

 /* __syncthreads();
  if(threadIdx.x == 0)
   {
	for(int i = 0; i < 5; ++i)
	  {
	    for(int j = 0; j < 4; ++j)
	      {
		printf("%d ",rectsCountNode[i][j]);
	      }
	    printf("\n");
	  }

    }*/
  __syncthreads();

  /*
  if(threadIdx.x == 0 && blockIdx.x == 0)
    {
	  printf("%d level %d , ro %d rw %d \n",params->TOTAL_RECT,node.getLevel(),node.getLevel() % 2,(node.getLevel() + 1) % 2);

	         for(int i = 0; i < params->TOTAL_RECT; i++)
	  	{
	  	  printf("   rect: %d %d  sortedt: %d %d\n",(int)roRects[i].topLeft.x,(int)roRects[i].topLeft.y,(int)sortedRects[i].topLeft.x,(int)sortedRects[i].topLeft.y);
	  	}
    }*/
  //__syncthreads();

  if(threadIdx.x == (params->THREAD_PER_BLOCK - 1)) //ostatni watek, bo w ostatnim warpie mamy finalne dane, jeden blokowy watek ustala dzieci, ich indeksy, itd.
    {
        int nodesSumAtLevel = params->nodesCountAtLevel(node.getLevel() + 1);
        int nodesAtLevel = powf(NODES_NUMBER,node.getLevel());

	d_QuadTree* startNodeAtLevel = &nodes[nodesAtLevel]; // wskaznik na pierwszy wezel w tym poziomie
	int childCount = params->QUAD_TREE_CHILD_NUM;
	int childIndex = childCount * nodeId;
	const d_Rect& bounds = node.getBounds();

	//printf("SET CHLD:id: %d sumAtLvl %d ndsAtlvl %d childId %d\n",node.getId(),nodesSumAtLevel,nodesAtLevel,childIndex);

#pragma unroll
	for(int i = 0; i < params->QUAD_TREE_CHILD_NUM; i++)
	  {
	    startNodeAtLevel[childIndex + i].setId(nodesSumAtLevel + childIndex + i);
	    startNodeAtLevel[childIndex + i].setLevel(node.getLevel() + 1);
	    startNodeAtLevel[childIndex + i].setOwnRectOff(rectsCountNode[i][warpId]);
	    node.setChild(nodesSumAtLevel + childIndex + i,i);
	  }

	startNodeAtLevel[childIndex + NODE_ID::UP_LEFT].setBounds(d_Rect(bounds.topLeft,center));
	startNodeAtLevel[childIndex + NODE_ID::DOWN_RIGHT].setBounds(d_Rect(center,bounds.bottomRight));
	startNodeAtLevel[childIndex + NODE_ID::UP_RIGHT].setBounds(d_Rect(center.x,bounds.topLeft.y,
	                                                                bounds.bottomRight.x,center.y));
	startNodeAtLevel[childIndex + NODE_ID::DOWN_LEFT].setBounds(d_Rect(bounds.topLeft.x,center.y,
	                                                                   center.x,bounds.bottomRight.y));

	startNodeAtLevel[childIndex + NODE_ID::UP_LEFT].setOff(node.startRectOff(),rectsCountNode[0][warpId]);
	startNodeAtLevel[childIndex + NODE_ID::UP_RIGHT].setOff(rectsCountNode[0][warpId],rectsCountNode[1][warpId]);
	startNodeAtLevel[childIndex + NODE_ID::DOWN_LEFT].setOff(rectsCountNode[1][warpId],rectsCountNode[2][warpId]);
	startNodeAtLevel[childIndex + NODE_ID::DOWN_RIGHT].setOff(rectsCountNode[2][warpId],rectsCountNode[3][warpId]);
	node.setOwnRectOff(rectsCountNode[3][warpId]);

	// printf("at %d node: %d %d %d %d lvl: %d count %d\n", nodesAtLevel ,(int)node.getBounds().topLeft.x,(int)node.getBounds().topLeft.y,
	//						(int)node.getBounds().bottmRight.x,(int)node.getBounds().bottmRight.y,node.getLevel(),rectCount);
	  /*   printf("ch lvl %d i: 0 id %d  start %d end %d\n",startNodeAtLevel[childIndex + 0].getLevel(),startNodeAtLevel[childIndex + 0].getId(),node.startRectOff(), rectsCountNode[0][warpId]);
	     printf("ch lvl %d i: 1 id %d  start %d end %d\n",startNodeAtLevel[childIndex + 1].getLevel(),startNodeAtLevel[childIndex + 1].getId(),rectsCountNode[0][warpId], rectsCountNode[1][warpId]);
	     printf("ch lvl %d i: 2 id %d  start %d end %d\n",startNodeAtLevel[childIndex + 2].getLevel(),startNodeAtLevel[childIndex + 2].getId(),rectsCountNode[1][warpId], rectsCountNode[2][warpId]);
	     printf("ch lvl %d i: 3 id %d  start %d end %d\n",startNodeAtLevel[childIndex + 3].getLevel(),startNodeAtLevel[childIndex + 3].getId(),rectsCountNode[2][warpId], rectsCountNode[3][warpId]);
	     printf("own  start %d end %d\n",rectsCountNode[3][warpId],node.endRectOff());
*/
	   /*  for(int i = 0; i < 4 ; i++)
	       {
		     printf("ch lvl %d i: %d id %d  x %d y %d\n",startNodeAtLevel[childIndex + i].getLevel(),i,startNodeAtLevel[childIndex + i].getId(),(int)startNodeAtLevel[childIndex + i].getBounds().topLeft.x,(int)startNodeAtLevel[childIndex + i].getBounds().topLeft.y);
	       }*/
/*
       for(int i = 0; i < params->TOTAL_RECT; i++)
	{
	  printf("   rect: %d %d\n",(int)sortedRects[i].topLeft.x,(int)sortedRects[i].topLeft.y);
	}*/
	//printf("wywolanie %d %d nodeid %d\n",node.ownRectOff(),node.endRectOff(),nodeId);
	createQuadTreeKernel<<<childCount,params->THREAD_PER_BLOCK ,
	(childCount + 1) * params->THREAD_PER_BLOCK * sizeof(int)>>>(startNodeAtLevel,rects,params,nodeId);
    }
  __syncthreads();

  if((node.getLevel() % 2) == 0) // musimy skopiowac do wlasciwej tablicy wyniki rectow ktore pozostaly w tym nodzie na koniec
  {
     int it = node.ownRectOff(), end = node.endRectOff();
     int total = params->TOTAL_RECT;
     int threadsNum = params->THREAD_PER_BLOCK;
     for (it += threadIdx.x ; it < end ; it += threadsNum)
       {
	 if (it < end)
	   rects[it] = rects[total + it];
       }
  }
}

bool checkQuadTree(const d_QuadTree *nodes,int idx,d_Rect *rects, int& count)
{
    Params params;
    const d_QuadTree* node = &nodes[idx];
    int rectCount = node->rectCount();

   // printf("%d \n",node->endRectOff() - node->ownRectOff());

 //   printf("node: %d %d %d %d lvl: %d count %d\n", (int)node->getBounds().topLeft.x,(int)node->getBounds().topLeft.y,
//								  (int)node->getBounds().bottmRight.x,(int)node->getBounds().bottmRight.y,node->getLevel(),rectCount);
    if (node->getLevel() < params.MAX_LEVEL && node->rectCount() > params.MIN_RECT_IN_NODE)
    {
	int rectInNode = 0;
	for(int i = 0; i < params.QUAD_TREE_CHILD_NUM; i++)
	  {
	  //  printf("%d \n",node->child(i));
	    rectInNode += nodes[node->child(i)].rectCount();
	  }
	rectInNode += node->endRectOff() - node->ownRectOff();
	//printf("\ninnode %d own %d    s  %d e  %d\n",rectInNode,node->ownRectOff(),node->startRectOff(),node->endRectOff());
	if(rectInNode != rectCount)
	  {
            char error[100];
            sprintf(error,"node: %d %d %d %d : ilosc dzieci nie zgadza sie %d %d \n",(int)node->getBounds().topLeft.x,(int)node->getBounds().topLeft.y,
                    (int)node->getBounds().bottomRight.x,(int)node->getBounds().bottomRight.y,rectInNode,rectCount);
            ErrorLogger::getInstance() >> "CreateTree: blad w">> error >> "\n";
	    return false;
	  }

        return checkQuadTree(nodes,node->child(0), rects,count) &&
               checkQuadTree(nodes,node->child(1), rects,count) &&
               checkQuadTree(nodes,node->child(2), rects,count) &&
               checkQuadTree(nodes,node->child(3), rects,count);
    }

    rectCount += node->startRectOff();
    for (int it = node->startRectOff() ; it < node->endRectOff() ; ++it)
    {
        if (it >= rectCount)
          {
            char error[100];
            sprintf(error,"node: %d %d %d %d : it != rectCount\n",(int)node->getBounds().topLeft.x,(int)node->getBounds().topLeft.y,
								  (int)node->getBounds().bottomRight.x,(int)node->getBounds().bottomRight.y);
            ErrorLogger::getInstance() >> "CreateTree: blad w" >>error>>"\n";
            return false;
          }
	//  printf("it %d rect %d %d %d %d \n",it,(int)rects[it].topLeft.x,(int)rects[it].topLeft.y,
	//			    (int)rects[it].bottmRight.x,(int)rects[it].bottmRight.y);
        if (!node->getBounds().contains(rects[it]))
          {
            char error[100];
            sprintf(error," node: lvl %d %d %d %d %d : nie zawiera rect id %d\n",node->getLevel(),(int)node->getBounds().topLeft.x,(int)node->getBounds().topLeft.y,
                    (int)node->getBounds().bottomRight.x,(int)node->getBounds().bottomRight.y,it);
            ErrorLogger::getInstance() >>"CreateTree: blad w" >>error >>"\n";
            return false;
          }
    }

    return true;
}
