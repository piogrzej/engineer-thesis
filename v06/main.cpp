//V05-URUCHOMIONE FUNKCJE GREENA

#include "mainFunctions.h"
#include <ctime>
#include <iostream>

int debug=0;

int main()
{
	QuadTree *mainTree;
	Rect spaceSize;
	FILE *fileIter;

	double duration;
	
	char adresWejscia[300];//E:\\programowanie\\quadtree\\sigfill_chunk_x.mag

	//C:\Users\Marcin\Documents\inzynierka\sigfill_chunk_x.mag

	/*---------------------------------debug section-----------------------*/

	debugFunction();

	/*---------------------------end of debug section---------------------*/
	scanf("%s", adresWejscia);//wejscie

	std::clock_t startT;//start timera

	fileIter = fopen(adresWejscia, "r");
	if (fileIter == NULL) return -1;
	spaceSize = layerSpaceSize(fileIter);
	cout << spaceSize;
	fseek(fileIter, 0, SEEK_SET); // przestawia wskaŸnik na pocz¹tek
		
	startT = std::clock();
	mainTree = new QuadTree(0, spaceSize);//start Tree
	createTree(mainTree, fileIter);
	Rect founded = RandomWalk(spaceSize, mainTree);
	cout << founded;

	duration = (std::clock() - startT) / (double)CLOCKS_PER_SEC;
	fclose(fileIter);

	printf("time: %lf\n", duration);

	mainTree->debugFunction();

	printf("debug=%d\n", debug);

	mainTree->clear();

	return 0;
}