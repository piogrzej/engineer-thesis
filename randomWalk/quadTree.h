#ifndef QUADTREE_H
#define QUADTREE_H

#define NUMBER_OF_NODES 4
#define MAX_OBJECTS 4
#define MAX_LEVELS 20
#define GAUSS_SURFACE_MULTIPLIER 1.1
#define BOUNDS_MUL_FACTOR 0.01 // percent of whole space size that the space will be increased

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
    std::list<RectHost> objects;
    RectHost            bounds;
    TreePtr         nodes[NUMBER_OF_NODES];
    bool            isSplited;
    int             nodeCount;
    void            split();
    floatingPoint   getAdjustedGaussianFactor(RectHost const& r, floatingPoint const factor, FACTOR_TYPE type);
   
public:
            Tree(int pLevel, int nodeSCount, RectHost const& bounds);

    bool    isInBounds(point const& p);
    bool    isInBounds(RectHost const& r);
    void    clear();
    bool    insert(RectHost const& r);
    RectHost    getObjectAtIndex(int index);
    void    addToObjects(RectHost const& r);
    void    deleteObjects();
    bool    checkCollisions(RectHost const& r, const RectHost &ignore = RectHost());
    bool    checkCollisionsWithObjs(RectHost const& r, const RectHost &ignore);
    bool    checkCollisionObjs(point p, RectHost &r);
    bool    checkCollisons(point p, RectHost& r);
    RectHost    drawBiggestSquareAtPoint(point p);
    void    printTree(std::string const & name);
    RectHost    creatGaussianSurfFrom(RectHost const & r,floatingPoint const factor);

    // HELPER
    void    addNodesToStack(TreePtr* stackPtr, Tree* except, bool collisions[]);
    bool    checkIsAnyCollision(bool collisions[]);

};

#endif
