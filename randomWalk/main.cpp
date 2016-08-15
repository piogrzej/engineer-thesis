#include "mainFunctions.h"
#include "Parser.h"
#include "ErrorHandler.h"
#include "Timer.h"
#include "tests.h"
#include <iostream>

inline bool checkFile(char* name) 
{
    std::ifstream f(name);
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

    Timer::getInstance().start("parser");
    Parser parser(path, "<<",5);
    Timer::getInstance().stop("parser");
    Layer const& layer = parser.getLayerAt(0);
    Rect const& spaceSize = parser.getLayerSize(0);

    mainTree = new Tree(0,layer.size(), spaceSize);//start Tree
    Timer::getInstance().start("createTree");
    createTree(mainTree,layer);
    Timer::getInstance().stop("createTree");

    mainTree->printTree("ROOT");

    randomWalkTest("../tests/test",100,10);

    mainTree->clear();

    return 0;
}