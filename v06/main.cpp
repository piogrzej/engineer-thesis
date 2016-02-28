//V05-URUCHOMIONE FUNKCJE GREENA

#include "mainFunctions.h"
#include <ctime>
#include <iostream>

int debug=0;

int main(){

	quadTree *mainTree;
	Rect spaceSize;
	FILE *fileIter;

	double duration;
	
	char adresWejscia[300];//E:\\programowanie\\quadtree\\sigfill_chunk_x.mag

	

	/*---------------------------------debug section-----------------------*/

	debugFunction();

	/*---------------------------end of debug section---------------------*/
	scanf("%s", adresWejscia);//wejscie

	std::clock_t startT;//start timera

	fileIter = fopen(adresWejscia, "r");
	if (fileIter == NULL) return -1;
	spaceSize = layerSpaceSize(fileIter);
	printf("TOP: %d LEFT: %d BOTTOM: %d RIGHT %d\n", spaceSize.topLeft.y, spaceSize.topLeft.x, spaceSize.bottomRight.y, spaceSize.bottomRight.x);
	
	fseek(fileIter, 0, SEEK_SET); // przestawia wskaŸnik na pocz¹tek
		
	startT = std::clock();
	mainTree = new quadTree(0, spaceSize);//start Tree
	createTree(mainTree, fileIter);
	duration = (std::clock() - startT) / (double)CLOCKS_PER_SEC;
	fclose(fileIter);

	printf("time: %lf\n", duration);

	mainTree->debugFunction();

	printf("debug=%d\n", debug);

	mainTree->clear();

	return 0;
}