//V05-URUCHOMIONE FUNKCJE GREENA

#include "mainFunctions.h"
#include <ctime>
#include <iostream>

int debug=0;

int main()
{
	QuadTree *mainTree = NULL;
	Rect spaceSize;
	FILE *fileIter;

	double duration;
	
	char adresWejscia[300] = "D:\\programowanie\\repositories\\surface.test";

	//C:\Users\Marcin\Documents\inzynierka\sigfill_chunk_x.mag

	/*---------------------------------debug section-----------------------*/

	//debugFunction();

	/*---------------------------end of debug section---------------------*/
	//scanf("%s", adresWejscia);//wejscie

	std::clock_t startT;//start timera

	fileIter = fopen(adresWejscia, "r");
	if (fileIter == NULL) return -1;
	spaceSize = layerSpaceSize(fileIter);
	cout << spaceSize;
	fseek(fileIter, 0, SEEK_SET); // przestawia wskaŸnik na pocz¹tek
		
	startT = std::clock();
	mainTree = new QuadTree(0, spaceSize);//start Tree
	createTree(mainTree, fileIter);
	//Rect founded = RandomWalk(spaceSize, mainTree); <-this function arguments are tree and starting rect, not whloe surface
	Rect start(43, 29, 131, 71);//starting rect for test file "surface.test" -> 43 29 131 71
	Rect founded = RandomWalk(start, mainTree);
	cout << founded;

	duration = (std::clock() - startT) / (double)CLOCKS_PER_SEC;
	fclose(fileIter);

	printf("time: %lf\n", duration);

	mainTree->debugFunction();

	printf("debug=%d\n", debug);

	mainTree->clear();

	return 0;
}