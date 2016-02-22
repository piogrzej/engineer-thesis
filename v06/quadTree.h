#ifndef QUADTREE_H
#define QUADTREE_H

#define NUMBER_OF_NODES 4
#define MAX_OBJECTS 4
#define MAX_LEVELS 10
#define GAUSS_SURFACE_MULTIPLIER 1.1

#include "listFunctions.h"

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
	Rect bounds;
	quadTree *UL;
	quadTree *UR;
	quadTree *LR;
	quadTree *LL;
	void split();
public:
	bool contains(Rect r);
	quadTree(int pLevel, Rect bounds);//konstruktor
	void clear();
	bool insert(Rect r);
	int getObjectSize();
	Rect getObjectAtIndex(int index);
	Rect removeAndReturnObjectAtIndex(int index);
	void addToObjects(Rect r);
	void deleteObjects();
	quadTree findRect(Rect r);//jezeli nie jestes pewien czy r napewno znajduje sie w bounds obiektu, wywlaj obiekt->contains()! JESZCZE NIE PRZETESTOWANE!
	void retrieve(list *returnedRecs, Rect r);//zwraca wszytskie Rect kolidujace z r w formie listy
	void debugFunction();//do usuneicia
	void getCollisionObjs(list *returnedRects, Rect r);//ddoaje do listy wszytskie Rects z listy objects ktore koliduja z r w danym obiekcie
	bool checkCollisionObjs(point p);//sprawdza czy p nie koliduje z jakims obiektyem z listy objects
	bool checkCollisons(point p);//sprawdza czy punkt p nie koliduje z jakims prostokatem
	Rect drawBiggestSquareAtPoint(point p);
};

#endif