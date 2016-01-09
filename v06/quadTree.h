#ifndef QUADTREE_H
#define QUADTREE_H

#define NUMBER_OF_NODES 4
#define MAX_OBJECTS 4
#define MAX_LEVELS 10

#include "struktury.h"
#include "listFunctions.h"
#include "rectFunctions.h"

extern int debug;

/*definicja klasy quadTree*/
/*NODES
+---+---+
| 0	| 1 |
+---+---+
| 3	| 2 |
+---+---+
*/
class quadTree{
private:
	int level;
	list objects;
	rect bounds;
	quadTree *UL;
	quadTree *UR;
	quadTree *LR;
	quadTree *LL;
	void split();
public:
	bool contains(rect r);
	quadTree();
	quadTree(int pLevel, rect bounds);//konstruktor
	void clear();
	bool insert(rect r);
	int getObjectSize();
	rect getObjectAtIndex(int index);
	rect removeAndReturnObjectAtIndex(int index);
	void addToObjects(rect r);
	void deleteObjects();
	quadTree findRect(rect r);//jezeli nie jestes pewien czy r napewno znajduje sie w bounds obiektu, wywlaj obiekt->contains()! JESZCZE NIE PRZETESTOWANE!
	void retrieve(list *returnedRecs, rect r);//zwraca wszytskie rect kolidujace z r w formie listy
	void debugFunction();//do usuneicia
	void getCollisionObjs(list *returnedRects, rect r);//ddoaje do listy wszytskie rects z listy objects ktore koliduja z r w danym obiekcie
	bool checkCollisionObjs(point p);//sprawdza czy p nie koliduje z jakims obiektyem z listy objects
	bool checkCollisons(point p);//sprawdza czy punkt p nie koliduje z jakims prostokatem
	rect drawBiggestRectAtPoint(point p);
};

#endif