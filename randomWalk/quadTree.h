#ifndef QUADTREE_H
#define QUADTREE_H

#define NUMBER_OF_NODES 4
#define MAX_OBJECTS 4
#define MAX_LEVELS 10
#define GAUSS_SURFACE_MULTIPLIER 1.1

#include <list>
#include "Rect.h"

/*definicja klasy QuadTree*/
/*NODES
+---+---+
| 0 | 1 |
+---+---+
| 3 | 2 |
+---+---+
*/
class QuadTree{

private:
	int		level;
	std::list<Rect> objects;
	Rect		bounds;
	QuadTree*	UL;
	QuadTree*	UR;
	QuadTree*	LR;
	QuadTree*	LL;
	bool		isSplited;

	void		split();

public:
                QuadTree(int pLevel, Rect const& bounds);//konstruktor
	bool	isInBounds(point const& p);
        bool	isInBounds(Rect const& r);
	void	clear();
	bool	insert(Rect const& r);
	Rect	getObjectAtIndex(int index);
	void	addToObjects(Rect const& r);
	void	deleteObjects();
	void	retrieve(std::list<Rect> *returnedRecs, Rect const& r);//zwraca wszytskie Rect kolidujace z r w formie listy
	void	getCollisionObjs(std::list<Rect> *returnedRects, Rect const&  r);//ddoaje do listy wszytskie Rects z listy objects ktore koliduja z r w danym obiekcie
	bool	checkCollisionObjs(point p, Rect &r);//sprawdza czy p nie koliduje z jakims obiektyem z listy objects, zwraca ten obiekt jako r
	bool	checkCollisons(point p, Rect& r);//sprawdza czy punkt p nie koliduje z jakims prostokatem, zwraca ten prostokat jako r
	Rect	drawBiggestSquareAtPoint(point p);
	void	printTree(std::string const & name);
};

#endif