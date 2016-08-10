#include "mainFunctions.h"
#include "Parser.h"
#include "ErrorHandler.h"
#include "Timer.h"
#include "tests.h"
#include <iostream>

inline bool checkFile(char* name) 
{
    ifstream f(name);
    return f.good();
}

int main(int argc, char *argv[])
{
    Tree *mainTree;
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
    Layer const& layer = parser.getLayerAt(0);
    Rect const& spaceSize = parser.getLayerSize(0);

    mainTree = new Tree(0,layer.size(), spaceSize);//start Tree
    time.start();
    createTree(mainTree,layer);
    time.stop("Create tree: ");

    mainTree->printTree("ROOT");

//    Rect start = parser.getLayerAt(0).at(10);
//    time.start();
//    Rect founded = RandomWalk(start, mainTree);
//    time.stop("Random Walk: ");
//    ErrorHandler::getInstance() >> "Poczatkowy: " >> start;
//    ErrorHandler::getInstance() >> "Znaleziony: " >> founded;
//    randomWalkTest("../tests/test2",100,0);
    randomWalkTest("../tests/test2",100,0);

    mainTree->clear();

    return 0;
}