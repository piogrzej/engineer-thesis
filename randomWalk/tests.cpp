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
        cout << index << " ";
    }
    s = s / ITER;
    cout << endl;
    cout << "srednia=" << s << endl;
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
    ErrorHandler::getInstance() >> "TEST RandomWalk\n";
    Timer time;
    Parser parser(path, "<<");
    Layer layer = parser.getLayerAt(0);
    Rect const& spaceSize = parser.getLayerSize(0);
    Rect start = layer.at(RECT_ID);
    Tree *mainTree = new Tree(0, layer.size(), spaceSize);//start Tree
    int pos;
    int* foundedRectCount = new int[layer.size()];
    std::fill(foundedRectCount, foundedRectCount + layer.size(), 0);

    createTree(mainTree, layer);
    mainTree->printTree("ROOT");

    time.start();
 
    int errors = 0;
    for (int i = 0; i < ITER_NUM; i++)
    {
        Rect founded = RandomWalk(start, mainTree);
        pos = getRectIt(layer,founded);
        if (pos != -1)
            foundedRectCount[pos] += 1;
        else
            errors++;
    }
    time.stop("RandomWalks: ");

    int i = 0;
    for (Rect const& rect : layer)
    {
        ErrorHandler::getInstance() >> rect >> "    znaleziony: " >> foundedRectCount[i++] >> "razy\n";
    }
    ErrorHandler::getInstance() >> "Liczba bledow: " >> errors >> "razy\n";
    ErrorHandler::getInstance() >> "KONIEC TESTU!\n";
    delete foundedRectCount;
}