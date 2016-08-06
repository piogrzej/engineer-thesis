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
    TreePtr     nodes[NUMBER_OF_NODES];
    bool            isSplited;

    void            split();
    double          getAdjustedGaussianFactor(Rect const& r, double const factor, FACTOR_TYPE type);
   
public:
    static int      nodeCount;

            Tree(int pLevel, Rect const& bounds);
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
    Rect    creatGaussianSurfFrom(Rect const & r,double const factor);

    // HELPERS
    void    addNodesToStack(TreePtr* stackPtr, Tree* except, bool isUL, bool isUR, bool isLR, bool isLL);
};

#endif