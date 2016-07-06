//V05-URUCHOMIONE FUNKCJE GREENA

#include "mainFunctions.h"
#include "Parser.h"
#include "ErrorHandler.h"
#include "Timer.h"

#include <iostream>

int debug=0;

inline bool checkFile(char* name) 
{
	ifstream f(name);
	return f.good();
}

//E:\\programowanie\\quadtree\\sigfill_chunk_x.mag
//C:\Users\Marcin\Documents\inzynierka\sigfill_chunk_x.gk
int main(int argc, char *argv[])
{
	QuadTree *mainTree;
	char* path;
	char inputPath[300];
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
	createTree(mainTree, parser.getLayerAt(0));
	time.stop("Create tree: ");
	mainTree->debugFunction();

	Rect founded = RandomWalk(parser.getLayerAt(0).at(0), mainTree);
	ErrorHandler::getInstance() << founded;

	mainTree->clear();

	return 0;
}