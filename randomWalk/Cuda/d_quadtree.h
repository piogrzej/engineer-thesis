/*
 * QuadTree.h
 *
 *  Created on: 22 wrz 2016
 *      Author: mknap
 */

#ifndef QUADTREE_H_
#define QUADTREE_H_

#include "d_Rect.h"
#include "params.h"
#include "../green/green.h"
#include "../defines.h"
#include <stdio.h>

class d_QuadTree;
typedef d_QuadTree* dTreePtr;
typedef unsigned short int ushort;

enum NODE_ID
{
    UP_LEFT=0,
    UP_RIGHT=1,
    DOWN_LEFT=2,
    DOWN_RIGHT=3
};

enum D_FACTOR_TYPE
{
    D_FACTOR_X,
    D_FACTOR_Y
};


struct QuadTreeManager
{
	d_Rect	start;
    d_Rect* rects;
    d_QuadTree* nodes;
    d_QuadTree* root;
    REAL64_t* d_intg;

    int rectsCount;
    int maxlevel;
};

//czyli od rect[startOff] do rect[endOff] sa rect dla danego noda
//a w tablicy nodow to node[chldPtrs[TOP_LEFT]] to top_left dziecko danego noda

class d_QuadTree
{
public:
    __host__ __device__ d_QuadTree()
    : id(0), startOff(0), endOff(0),level(0), startOwnOff(0),treeManager(NULL),splitFlag(false){}

    __host__ __device__ d_QuadTree(int idP, int start, int end)
    : id(idP), startOff(start), endOff(end), level(0),startOwnOff(0),treeManager(NULL) {}

    __host__ __device__ void  setStack(dTreePtr** stck)  { stack = stck;  }
    __host__ __device__ void  setNotSplited()  { splitFlag = false;}
    __host__ __device__ void  setSplited()  { splitFlag = true;}
    __host__ __device__ int   getId() const { return id;}
    __host__ __device__ void  setId(int newId)  { id = newId;}
    __host__ __device__ int   getLevel() const { return level;}
    __host__ __device__ void  setLevel(int lvl)  { level = lvl;}
    __host__ __device__ void  setChild(int child, int index)  { children[index] = child; }
    __host__ __device__ void  setBounds(d_Rect const& rect)  { bounds = rect;}
    __host__ __device__ void  setOwnRectOff(int ownOff)  {  startOwnOff = ownOff; }
    __host__ __device__ d_Rect getBounds() const { return bounds; }
    __host__ __device__ int   rectCount() const{  return endOff - startOff;}
    __host__ __device__ void  setOff(int start, int end) { startOff = start;endOff = end;}
    __host__ __device__ int   startRectOff() const{  return startOff; }
    __host__ __device__ int   ownRectOff() const{  return startOwnOff; }
    __host__ __device__ int   endRectOff() const{  return endOff;}
    __host__ __device__ QuadTreeManager* getTreeManager() const{ return treeManager; };
    __host__ __device__ void setTreeManager(QuadTreeManager* manager) {this->treeManager = manager;};
    __host__ __device__ int   child(const int index) const { return children[index]; }
    __host__ __device__ bool isSplited() const {return splitFlag; }
    __host__ __device__ __forceinline__ point2 getCenter()
    {
        floatingPoint centerX =  bounds.topLeft.x + (bounds.bottomRight.x - bounds.topLeft.x) / 2.;
        floatingPoint centerY =  bounds.topLeft.y + (bounds.bottomRight.y - bounds.topLeft.y) / 2.;
        return make_point2(centerX,centerY);
    }
    __device__ void createStack(int id,int size) { stack[id] = new dTreePtr[size]; } // lazy fetch
    __device__ void freeStack(int id) { delete stack[id]; }
    __device__ bool isInBounds(point2 const& p);
    __device__ bool isInBounds(d_Rect const& r);
    __device__ bool checkCollisions(d_Rect const& r, const d_Rect &ignore = d_Rect());
    __device__ bool checkCollisionsWithObjs(d_Rect const& r, const d_Rect &ignore);
    __device__ bool checkCollisons(point2 p, d_Rect& r);
    __device__ d_Rect drawBiggestSquareAtPoint(point2 p);
    __device__ d_Rect createGaussianSurfFrom(d_Rect const & r, floatingPoint const factor);
    __device__ floatingPoint getAdjustedGaussianFactor(d_Rect const& r, floatingPoint const factor, D_FACTOR_TYPE type);
    __device__ int getChildren(ushort i) { return children[i]; }

private:
    int                 id; // indeks to globalnej tablicy węzłów
    int                 children[NODES_NUMBER];//w tej tablicy znajduja sie numery indeksow dzieci w treeManager->nodes[]
    int                 startOff; // indesky do tablicy gdzie rozpoczynaja sie obiekty znajdujace sie w tym dzieciach tego wezla
    int                 startOwnOff; // tu zaczynaja sie recty tego obiektu
    int	                endOff;
    int	                level;
    bool				splitFlag;
    d_Rect              bounds;
    QuadTreeManager*    treeManager;
    dTreePtr**			stack;
    __device__ bool checkCollisionObjs(point2 p, d_Rect &r);
    __device__ bool checkIsAnyCollision(bool collisions[]);

};

#endif /* QUADTREE_H_ */
