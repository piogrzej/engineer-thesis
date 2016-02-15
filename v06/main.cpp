//V05-URUCHOMIONE FUNKCJE GREENA

#include "mainFunctions.h"
#include "green.h"

int debug=0;

int main(){

	quadTree *mainTree;
	rect spaceSize;
	FILE *fileIter;

	double duration;
	REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE+1];
	UINT32_t Nsample=NSAMPLE;
	char adresWejscia[300];//E:\\programowanie\\quadtree\\sigfill_chunk_x.mag

	precompute_unit_square_green(g, dgdx, dgdy, intg, Nsample);//wyliczanie funkcji greena

	rng_init(1);//inicjalizacja genaeratora

	/*---------------------------------debug section-----------------------*/

	/*REAL64_t suma = 0;

	for (int i = 0; i < Nsample + 1; ++i){
		printf("%lf\n", intg[i]);
		suma += intg[i];
	}
	printf("%lf\n", suma);
	*/

	/*for (int i = 0; i < 1000; ++i){
		std::cout << (double)myrand() / (double)MY_RAND_MAX << std::endl;
	}*/

	/*---------------------------end of debug section---------------------*/

	scanf("%s", adresWejscia);//wejscie

	std::clock_t startT;//start timera

	fileIter = fopen(adresWejscia, "r");
	if (fileIter == NULL) return -1;
	spaceSize = layerSpaceSize(fileIter);
	printf("TOP: %d LEFT: %d BOTTOM: %d RIGHT %d\n", spaceSize.top_left.y, spaceSize.top_left.x, spaceSize.bottom_right.y, spaceSize.bottom_right.x);
	
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