#include "mainFunctions.h"
#include "ErrorHandler.h"
#include "Timer.h"


void createTree(Tree * mainTree, Layer const& layer){
    for(Rect const& rect : layer)
    {
        mainTree->insert(rect);
    }
}

int getIndex(REAL64_t intg[NSAMPLE + 1], double rand){
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
    point p;
    double r;
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
    Timer::getInstance().start("createGaussianSurface");
    Rect output, square = mainTree->creatGaussianSurfFrom(R, 1.5);
    Timer::getInstance().stop("createGaussianSurface");
    bool broken = false;

    do
    {
        r = myrand() / (double)(MY_RAND_MAX);
        index = getIndex(intg, r);
        Timer::getInstance().start("getPointFromNindex");
        p = square.getPointFromNindex(index, NSAMPLE);
        Timer::getInstance().stop("getPointFromNindex");
        if(false == mainTree->isInBounds(p))
        {
            broken = SPECIAL_VALUE_BOOLEAN;
            SPECIAL_ACTION;
        }
        Timer::getInstance().start("drawBiggestSquareAtPoint");
        square = mainTree->drawBiggestSquareAtPoint(p);
        Timer::getInstance().stop("drawBiggestSquareAtPoint");

        Timer::getInstance().start("checkCollisons");
        isCollison = mainTree->checkCollisons(p, output);
        Timer::getInstance().stop("checkCollisons");
        pointCount++;
    }
    while (false == isCollison);

    if (false == broken)
        ErrorHandler::getInstance() << "Ending: " << output << "\n";

    else
    {
        output.topLeft = point(-1,-1);
        output.bottomRight = point(-1,-1);
        ErrorHandler::getInstance() << "Random walk is out of the bounds!\n";
    }
    ErrorHandler::getInstance() << "Number of path's points: "  << pointCount << "\n";

    return output;
}

void printList(std::list<Rect> input)
{
    int i=0;
    for(std::list<Rect>::iterator iter = input.begin(); iter != input.end(); ++iter){
        i++;
        std::cout<<i<<" "<< iter->topLeft.x<<" "<<iter->topLeft.y<<" "<<iter->bottomRight.x<<" "<<iter->bottomRight.y<<std::endl;
     }
}