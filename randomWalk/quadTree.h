#ifndef QUADTREE_H
#define QUADTREE_H

#define NUMBER_OF_NODES 4
#define MAX_OBJECTS 4
#define MAX_LEVELS 10
#define GAUSS_SURFACE_MULTIPLIER 1.1
#define BOUNDS_MUL_FACTOR 0.05 // percent of whole space size that the space will be increased

#include <list>
#include "Rect.h"

/*definicja klasy QuadTree*/
/*NODES
+---+---+
| 0 | 1 |
+---+---+
| 3 | 2 |
+---+---+

+---+---+
|UL |UR |
+---+---+
|LL |LR |
+---+---+
*/
class Tree;
typedef Tree* TreePtr;
typedef unsigned short int ushort;

enum FACTOR_TYPE
{
    FACTOR_X,
    FACTOR_Y
};

class Tree
{
private:
    int             level;
    std::list<Rect> objects;
    Rect            bounds;
    TreePtr         nodes[NUMBER_OF_NODES];
    bool            isSplited;
    int             nodeCount;
    void            split();
    floatingPoint          getAdjustedGaussianFactor(Rect const& r, floatingPoint const factor, FACTOR_TYPE type);
   
public:
            Tree(int pLevel, int nodeSCount, Rect const& bounds);

    bool    isInBounds(point const& p);
    bool    isInBounds(Rect const& r);
    void    clear();
    bool    insert(Rect const& r);
    Rect    getObjectAtIndex(int index);
    void    addToObjects(Rect const& r);
    void    deleteObjects();
    bool    checkCollisions(Rect const& r, const Rect &ignore = Rect());
    bool    checkCollisionsWithObjs(Rect const& r, const Rect &ignore);
    bool    checkCollisionObjs(point p, Rect &r);
    bool    checkCollisons(point p, Rect& r);
    Rect    drawBiggestSquareAtPoint(point p);
    void    printTree(std::string const & name);
    Rect    creatGaussianSurfFrom(Rect const & r,floatingPoint const factor);

    // HELPER
    void    addNodesToStack(TreePtr* stackPtr, Tree* except, bool collisions[]);
    bool    checkIsAnyCollision(bool collisions[]);

};

#endif