#include "mainFunctions.h"
#include "ErrorHandler.h"
#include "Timer.h"


void createTree(Tree * mainTree, Layer const& layer){
    for(Rect const& rect : layer)
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

Rect RandomWalk(Rect R, Tree* mainTree, int& pointCount)
{   
    Rect output;
    point p;
    floatingPoint r;
    int index;
    bool isCollison;
    REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE + 1];
    UINT32_t Nsample = NSAMPLE;
    pointCount = 0;

    precompute_unit_square_green(g, dgdx, dgdy, intg, Nsample);//wyliczanie funkcji greena

#ifdef _WIN32
    rng_init(3);//inicjalizacja genaeratora
#elif __linux__
    rng_init(1);//inicjalizacja genaeratora
#endif

    ErrorHandler::getInstance() << "Starting: " << R << "\n";


#ifdef MEASURE_MODE
    Rect square = Timer::getInstance().measure("createGaussianSurface", *mainTree, 
                                               &Tree::creatGaussianSurfFrom, R, 1.5);
#else
    Rect square = mainTree->creatGaussianSurfFrom(R, 1.5);
#endif

    bool broken = false;

    do
    {
        r = myrand() / (floatingPoint)(MY_RAND_MAX);

#ifdef MEASURE_MODE
        p = Timer::getInstance().measure("getPointFromNindex",square,
                                         &Rect::getPointFromNindex, getIndex(intg, r), NSAMPLE);
#else
        p = square.getPointFromNindex(getIndex(intg, r), NSAMPLE);
#endif

        if(false == mainTree->isInBounds(p))
        {
            broken = SPECIAL_VALUE_BOOLEAN;
            SPECIAL_ACTION;
        }
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

    ErrorHandler::getInstance() << "Number of path's points: " << pointCount << "\n";

    if (false == broken)
        ErrorHandler::getInstance() << "Ending: " << output << "\n";

    else
        ErrorHandler::getInstance() << "Random walk is out of the bounds!\n";
    
    return output;
}

void printList(std::list<Rect> input)
{
    int i=0;
    for(std::list<Rect>::iterator iter = input.begin(); iter != input.end(); ++iter){
        ++i;
        std::cout<<i<<" "<< iter->topLeft.x<<" "<<iter->topLeft.y<<" "<<iter->bottomRight.x<<" "<<iter->bottomRight.y<<std::endl;
    }
}