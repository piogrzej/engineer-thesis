#include "../utils/Logger.h"
#include "getAvgPathLen.h"
#include "../green/green.h"
#include "mainFunctions.h"
#include "../utils/Timer.h"
#include "../utils/RandGen.h"

#include <math.h>
#include <map>
#include <iostream>
#include <string>

void saveOutput(std::map<RectHost,int> m,int ITER_NUM)
{
	std::string timestamp = std::to_string((unsigned long)time(NULL));
	std::string filename = timestamp+"wyniki.txt";
	std::ofstream out(filename);
	out << "[CPU]\n";
	for(auto const &entry : m)
	{
		if(-1==entry.first.topLeft.x && -1== entry.first.topLeft.y && -1== entry.first.bottomRight.x && -1== entry.first.bottomRight.y)
		{
			out << "Ile scieżek wyszło poza granice warstwy: "<<(float)entry.second/(float)ITER_NUM*100 << "%\n";
		}
		else
		{
			out << entry.first.topLeft.x <<" "<< entry.first.topLeft.y <<" "<<
					entry.first.bottomRight.x<<" "<<entry.first.bottomRight.y <<": "<<
					((float)entry.second/(float)ITER_NUM)*100<<"%\n";
		}
	}
	printf("Wynik zapisany do %s\n",filename.c_str());
}

floatingPoint getAvgPathLen(char* path, int ITER_NUM, int RECT_ID, bool measure,int layerID)
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

    const Layer layer = parser.getLayerAt(layerID);

    RectHost const& spaceSize = parser.getLayerSize(layerID);

    if (layer.size() <= RECT_ID || RECT_ID < 0 || ITER_NUM <= 0)
    {
        ErrorLogger::getInstance() >> "Incorrect args!" >> "\n";
        exit(0);
    }
  
    RectHost start = layer.at(RECT_ID);
    Tree *mainTree = new Tree(0, layer.size(), spaceSize);//start Tree
    int pos, sumPointCount = 0;
    unsigned long int sumOfOtherRects=0;
    std::map<RectHost,int> foundedMap;

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

    REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE + 1];
	precompute_unit_square_green(g,dgdx,dgdy,intg,NSAMPLE);// niech lidzy zeby bylo wiarygodnie
    if(true==measure)
	{
    	Timer::getInstance().start("_RandomWalk Total");
	}

    int errors = 0;
    for (int i = 0; i < ITER_NUM; i++)
    {
        int counter;
        RectHost founded = RandomWalk(start, mainTree, counter,intg,i);
        foundedMap[founded]++;
        if(!(founded == start)) sumOfOtherRects++;
        sumPointCount += counter;
    }

    if(true==measure)
    {
    	Timer::getInstance().stop("_RandomWalk Total");
    }

    saveOutput(foundedMap,ITER_NUM);

    if(true==measure)
    {
    	Timer::getInstance().stop("TotalTime");
    	Timer::getInstance().printResults();
    }
    ErrorLogger::getInstance() >> "END OF TEST!\n";
    return (floatingPoint)sumOfOtherRects / (floatingPoint)ITER_NUM;
}
