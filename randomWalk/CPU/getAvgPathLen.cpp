#include "../utils/Logger.h"
#include "getAvgPathLen.h"
#include "../green/green.h"
#include "mainFunctions.h"
#include "../utils/Timer.h"
#include "../utils/RandGen.h"

#include <math.h>

//#define MEASURE_MODE

int getRectIt(Layer const& layer, RectHost const& rect)
{
    int i = 0;
    for (RectHost const& rIt : layer)
    {
        if (rIt == rect)
            return i;

        i++;
    }
    return -1;
}

floatingPoint getAvgPathLen(char* path, int ITER_NUM, int RECT_ID, bool measure)
{
	Parser parser("<<");
	if(true==measure)
	{
		TimeLogger::getInstance() << "RandomWalk \nTest: " << path << "\n";
		Timer::getInstance().start("TotalTime");
		Timer::getInstance().start("_Parser");
		parser.parse(path);
		Timer::getInstance().stop("_Parser");
	}
	else
	{
		parser.parse(path);
	}
    ErrorLogger::getInstance() >> "RandomWalk \nTest: " >> path >> "\n";

    const Layer layer = parser.getLayerAt(0);

    RectHost const& spaceSize = parser.getLayerSize(0);

    if (layer.size() <= RECT_ID || RECT_ID < 0 || ITER_NUM <= 0)
    {
        ErrorLogger::getInstance() >> "Incorrect args!" >> "\n";
        exit(0);
    }
  
    RectHost start = layer.at(RECT_ID);
    Tree *mainTree = new Tree(0, layer.size(), spaceSize);//start Tree
    int pos, sumPointCount = 0;
    int* foundedRectCount = new int[layer.size()+1];
    std::fill(foundedRectCount, foundedRectCount + layer.size()+1, 0);

    if(true==measure)
	{
		Timer::getInstance().start("Create Tree");
		createTree(mainTree, layer);
		Timer::getInstance().stop("Create Tree");
	}
    else
    {
    	createTree(mainTree, layer);
    }

#ifdef DEBUG_MODE 
    mainTree->printTree("ROOT");
#endif

    RandGen gen;
    gen.initDeterm(ITER_NUM);
    gen.initPtrs();
    if(true==measure)
	{
    	Timer::getInstance().start("_RandomWalk Total");
	}

    int errors = 0;
    for (int i = 0; i < ITER_NUM; i++)
    {
        int counter;
        RectHost founded = RandomWalk(start, mainTree, counter,gen,i);
        if(-1 == founded.topLeft.x &&
           -1 == founded.topLeft.y &&
           -1 == founded.bottomRight.x &&
           -1 == founded.bottomRight.y)
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
       // std::cout << founded << std::endl;
    }

    if(true==measure)
    {
    	Timer::getInstance().stop("_RandomWalk Total");
    }

    int i = 0;

#ifdef DEBUG_MODE 
    for (RectHost const& rect : layer)
    {
        ErrorLogger::getInstance() << rect << " founded: " << foundedRectCount[i++] << " times\n";
    }
    ErrorLogger::getInstance() << "Out of bounds case founded: " << foundedRectCount[i++] << " times\n";
    
    ErrorLogger::getInstance() << "Number of errors: " << errors << "\n";
    ErrorLogger::getInstance() << "Avarage number of path's points: " << sumPointCount / ITER_NUM << "\n";
#endif

    if(true==measure)
    {
    	Timer::getInstance().stop("TotalTime");
    	Timer::getInstance().printResults();
    }
    delete foundedRectCount;
    ErrorLogger::getInstance() >> "END OF TEST!\n";
    return (floatingPoint)sumPointCount / (floatingPoint)ITER_NUM;
}
