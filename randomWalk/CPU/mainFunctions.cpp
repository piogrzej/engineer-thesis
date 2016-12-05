#include <fstream>
#include "mainFunctions.h"
#include "../utils/Logger.h"
#include "../utils/Timer.h"
#include "getAvgPathLen.h"
#include "../Cuda/mainkernels.h"
#include <iostream>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

void runRandomWalk(char* path, int ITER_NUM, int RECT_ID, bool GPU_FLAG,bool measure,int layer)
{
	auto t1 = Clock::now();
	floatingPoint result;
	std::string name;
	if(GPU_FLAG)
	{
		name = "[GPU]";
		result  = getAvgPathLenCUDA(path,ITER_NUM,RECT_ID,measure,layer);
	}
	else
	{
		name = "[CPU]";
		result = getAvgPathLen(path,ITER_NUM,RECT_ID,measure,layer);
	}
	auto t2 = Clock::now();
	printf("%s Ile sciezek trafiło do innego elementu: %f\%\n",name.c_str(), result * 100.);
	auto timeResult = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "Czas wykonania: " <<  timeResult  << " ms" << std::endl;
}

void createTree(Tree * mainTree, Layer const& layer){
    for(RectHost const& rect : layer)
    {
        mainTree->insert(rect);
    }
}


RectHost RandomWalk(RectHost const& R, Tree* mainTree, int& pointCount,RandGen& gen,int iterId)
{   
    RectHost output;
    point p;
    floatingPoint r;
    int index;
    bool isCollison;
    UINT32_t Nsample = NSAMPLE;
    pointCount = 0;

#ifdef _WIN32
    rng_init(3);//inicjalizacja genaeratora
#elif __linux__
    rng_init(1);//inicjalizacja genaeratora
#endif

    RectHost square = mainTree->creatGaussianSurfFrom(R, 1.5);

    bool broken = false;

    do
    {
        r = myrand() / (floatingPoint)(MY_RAND_MAX);
        index = gen.nextIndexRand(r);
        p = square.getPointFromNindex(index, NSAMPLE);
        //printf("%f %f %f\n",r,p.x,p.y);
        if(false == mainTree->isInBounds(p))
        {
            broken = true;
            break;
        }
#ifdef DEBUG_MODE
        ErrorLogger::getInstance() << p.x << "," << p.y << "\n";
#endif

        square = mainTree->drawBiggestSquareAtPoint(p);
        isCollison = mainTree->checkCollisons(p, output);

        pointCount++;
    }
    while (false == isCollison);

    ErrorLogger::getInstance() << "Number of path's points: " << pointCount << "\n";

    if (false == broken)
        ErrorLogger::getInstance() << "Ending: " << output << "\n";

    else
        ErrorLogger::getInstance() << "Random walk is out of the bounds!\n";
    
    return output;
}

void printList(std::list<RectHost> input)
{
    int i=0;
    for(std::list<RectHost>::iterator iter = input.begin(); iter != input.end(); ++iter){
        ++i;
        std::cout<<i<<" "<< iter->topLeft.x<<" "<<iter->topLeft.y<<" "<<iter->bottomRight.x<<" "<<iter->bottomRight.y<<std::endl;
    }
}
