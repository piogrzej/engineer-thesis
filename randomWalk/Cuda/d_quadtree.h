/*
 * QuadTree.h
 *
 *  Created on: 22 wrz 2016
 *      Author: mknap
 */

#ifndef QUADTREE_H_
#define QUADTREE_H_

#include "d_vector.h"
#include "../Rect.h"
#include "params.h"

enum class NODE_ID
{
  UP_LEFT,
  UP_RIGHT,
  DOWN_LEFT,
  DOWN_RIGHT
};

class d_QuadTree
{
public:
  __host__ __device__ d_QuadTree()
  : id(0), startOff(0), endOff(0),level(0) {}

  __host__ __device__ d_QuadTree(int idP, int start, int end)
  : id(idP), startOff(start), endOff(end), level(0) {}

  __host__ __device__ int 	getId() const { return id;}
  __host__ __device__ void 	setId(int newId)  { id = newId;}
  __host__ __device__ int 	getLevel() const { return level;}
  __host__ __device__ void 	setLevel(int lvl)  { level = lvl;}
  __host__ __device__ RectCuda  getBounds() { return bounds; }
  __host__ __device__ int 	rectCount() const{  return startOff - endOff;}
  __host__ __device__ int 	startRectOff() const{  return startOff; }
  __host__ __device__ int 	endRectOff() const{  return endOff;}
  __host__ __device__ int	operator[](const int index) { return chldPtrs[index]; }
  __host__ __device__ __forceinline__
		      point2    getCenter()
  {
    floatingPoint centerX = (bounds.bottmRight.x - bounds.topLeft.x) / 2;
    floatingPoint centerY = (bounds.bottmRight.y - bounds.topLeft.y) / 2;
    return make_point2(centerX,centerY);
  }

private:
  int        id; // Wskaźnik to globalnej tablicy węzłów
  int        chldPtrs[NODES_NUMBER];
  int        startOff; // wskaźnik do tablicy gdzie rozpoczynaja sie obiekty znajdujace sie w tym węźle
  int	     endOff;
  int	     level;
  RectCuda   bounds;

};



#endif /* QUADTREE_H_ */
