//V05-URUCHOMIONE FUNKCJE GREENA

#include "mainFunctions.h"
#include "Parser.h"
#include "ErrorHandler.h"
#include "Timer.h"
#include "tests.h"
#include <iostream>

int debug=0;

inline bool checkFile(char* name) 
{
	ifstream f(name);
	return f.good();
}

int main(int argc, char *argv[])
{
    QuadTree *mainTree;
    char* path;
    char inputPath[300];//E:\\programowanie\\quadtree\\sigfill_chunk_x.mag
                        //C:\Users\Marcin\Documents\inzynierka\sigfill_chunk_x.gk
    if (argc > 1)
    {
            path = argv[1];
    }
    else
    {
            scanf("%s", inputPath);//wejscie
            path = inputPath;
    }

    if (false == checkFile(path))
    {
            ErrorHandler::getInstance() << "Nie ma takiego pliku!";
            return 0;
    }
    Timer time;
    time.start();
    Parser parser(path, "<<",5);
    time.stop("Parser: ");
    Rect const& spaceSize = parser.getLayerSize(0);

    mainTree = new QuadTree(0, spaceSize);//start Tree
    time.start();
    createTree(mainTree, parser.getLayerAt(00));
    time.stop("Create tree: ");

    mainTree->printTree("ROOT");

//    Rect start = parser.getLayerAt(0).at(10);
//    time.start();
//    Rect founded = RandomWalk(start, mainTree);
//    time.stop("Random Walk: ");
//    ErrorHandler::getInstance() >> "Poczatkowy: " >> start;
//    ErrorHandler::getInstance() >> "Znaleziony: " >> founded;
    randomWalkTest("../tests/test1",100);

    mainTree->clear();

    return 0;
}