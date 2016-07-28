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
            QuadTree(int pLevel, Rect const& bounds);
    bool    isInBounds(point const& p);
    bool    isInBounds(Rect const& r);
    void    clear();
    bool	insert(Rect const& r);
    Rect	getObjectAtIndex(int index);
    void	addToObjects(Rect const& r);
    void	deleteObjects();
    bool	checkCollisions(Rect const& r, const Rect &ignore = Rect());
    bool	getCollisionObjs(Rect const& r, const Rect &ignore);
    bool	checkCollisionObjs(point p, Rect &r);
    bool	checkCollisons(point p, Rect& r);
    Rect	drawBiggestSquareAtPoint(point p);
    void	printTree(std::string const & name);
    Rect    creatGaussianSurfFrom(Rect const & r,double const factor, double& resultFactor);
};

#endif