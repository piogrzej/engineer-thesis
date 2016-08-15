#include "ErrorHandler.h"
#include "tests.h"
#include "green.h"
#include "mainFunctions.h"
#include "Timer.h"

#include <math.h>

#define NSAMPLE 200
#define ITER 50

void randomIndexTest()
{
    REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE + 1];
    UINT32_t Nsample = NSAMPLE;
    int results[ITER];

    precompute_unit_square_green(g, dgdx, dgdy, intg, Nsample);//wyliczanie funkcji greena

    rng_init(3);//inicjalizacja genaeratora

    double s = 0,s2 = 0 ,d = 0, sig = 0, odchylenie = 0;
    int index, index2;
    for (int i = 0; i < ITER; i++)
    {
        double randNum = myrand() / (double)(MY_RAND_MAX);
        index = getIndex(intg, randNum);
        s += index;
        std::cout << index << " ";
    }
    s = s / ITER;
    std::cout << "\n";
    std::cout << "srednia=" << s << "\n";
}

int getRectIt(Layer const& layer, Rect const& rect)
{
    int i = 0;
    for (Rect const& rIt : layer)
    {
        if (rIt == rect)
            return i;

        i++;
    }
    return -1;
}

void randomWalkTest(char* path, int ITER_NUM, int RECT_ID)
{
    ErrorHandler::getInstance() >> "RandomWalk \nTest: " >> path >> "\n";

    Parser parser(path, "<<");
    Layer layer = parser.getLayerAt(0);
    Rect const& spaceSize = parser.getLayerSize(0);

    if (layer.size() <= RECT_ID || RECT_ID < 0 || ITER_NUM <= 0)
    {
        ErrorHandler::getInstance() >> "Incorrect args!" >> "\n";
        exit(0);
    }

    Rect start = layer.at(RECT_ID);
    Tree *mainTree = new Tree(0, layer.size(), spaceSize);//start Tree
    int pos, sumPointCount = 0;
    int* foundedRectCount = new int[layer.size()+1];
    std::fill(foundedRectCount, foundedRectCount + layer.size()+1, 0);

    createTree(mainTree, layer);
    mainTree->printTree("ROOT");

    Timer::getInstance().start("_TotalTime:");
 
    int errors = 0;
    for (int i = 0; i < ITER_NUM; i++)
    {
        int counter;
        Rect founded = RandomWalk(start, mainTree, counter);
        if(-1==founded.topLeft.x
                &&-1==founded.topLeft.y
                &&-1==founded.bottomRight.x
                &&-1==founded.bottomRight.y)
            ++foundedRectCount[layer.size()];
        else
        {
            pos = getRectIt(layer,founded);
            if (pos != -1)
                foundedRectCount[pos] += 1;
            else
                errors++;
        }
        sumPointCount += counter;
    }
    Timer::getInstance().stop("_TotalTime:");

    int i = 0;
    for (Rect const& rect : layer)
    {
        ErrorHandler::getInstance() >> rect >> " founded: " >> foundedRectCount[i++] >> " times\n";
    }
    ErrorHandler::getInstance() >> "Out of bounds case founded: " >> foundedRectCount[i++] >> " times\n";
    ErrorHandler::getInstance() >> "Number of errors: " >> errors >> "\n";
    ErrorHandler::getInstance() >> "Avarage number of path's points: " >> sumPointCount / ITER_NUM >> "\n";
    Timer::getInstance().printResults();
    ErrorHandler::getInstance() >> "END OF TEST!\n";
    delete foundedRectCount;
}