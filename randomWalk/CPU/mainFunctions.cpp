#include <fstream>
#include "mainFunctions.h"
#include "../utils/Logger.h"
#include "../utils/Timer.h"
#include "getAvgPathLen.h"
#include "../parallelFunc.h"
#include <iostream>

//#define MEASURE_MODE

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

void runRandomWalk(char* path, int ITER_NUM, int RECT_ID)
{
	auto t1 = Clock::now();
	if(GPU_FLAG)
		printf("%f\n",getAvgPathLenCUDA(path,ITER_NUM,RECT_ID));
	else
		printf("%f\n",getAvgPathLen(path,ITER_NUM,RECT_ID));
	auto t2 = Clock::now();
	std::cout << "Execution time: "
	        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	        << " ms" << std::endl;
}

void createTree(Tree * mainTree, Layer const& layer){
    for(RectHost const& rect : layer)
    {
        mainTree->insert(rect);
    }
}

int getIndex(REAL64_t intg[NSAMPLE + 1], floatingPoint rand){
    for (int i = 0; i <= NSAMPLE; ++i)
    {
        if (intg[i] <= rand && intg[i + 1] > rand) 
            return i;
    }
}

int getDistanceRomTwoPoints(point p1, point p2)
{
    return (int)sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}

RectHost RandomWalk(RectHost const& R, Tree* mainTree, int& pointCount,REAL64_t intg[NSAMPLE + 1])
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

    ErrorLogger::getInstance() << "Starting: " << R << "\n";


#ifdef MEASURE_MODE
    RectHost square = Timer::getInstance().measure("createGaussianSurface", *mainTree, 
                                               &Tree::creatGaussianSurfFrom, R, floatingPoint(1.5));
#else
    RectHost square = mainTree->creatGaussianSurfFrom(R, 1.5);
#endif

    bool broken = false;

    do
    {
#ifdef MEASURE_MODE
        r = ((floatingPoint)Timer::getInstance().measure("myrand",myrand))/(floatingPoint)(MY_RAND_MAX);
#else
        r = myrand() / (floatingPoint)(MY_RAND_MAX);
#endif
        

#ifdef MEASURE_MODE
        p = Timer::getInstance().measure("getPointFromNindex",square,
                                         &RectHost::getPointFromNindex, getIndex(intg, r), NSAMPLE);
#else
        p = square.getPointFromNindex(getIndex(intg, r), NSAMPLE);
#endif
        if(false == mainTree->isInBounds(p))
        {
            broken = true;
            break;
        }
#ifdef DEBUG_MODE
        ErrorLogger::getInstance() << p.x << "," << p.y << "\n";
#endif

#ifdef MEASURE_MODE
        square = Timer::getInstance().measure("drawBiggestSquareAtPoint", *mainTree, 
                                              &Tree::drawBiggestSquareAtPoint, point(p));
        isCollison = Timer::getInstance().measure("checkCollisons", *mainTree, 
                                              &Tree::checkCollisons, point(p), output);
#else
        square = mainTree->drawBiggestSquareAtPoint(p);
        isCollison = mainTree->checkCollisons(p, output);
#endif

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
