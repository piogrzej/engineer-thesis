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

enum NODE_ID
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
  : id(0), startOff(0), endOff(0),level(0), startOwnOff(0){}

  __host__ __device__ d_QuadTree(int idP, int start, int end)
  : id(idP), startOff(start), endOff(end), level(0),startOwnOff(0) {}

  __host__ __device__ int 	getId() const { return id;}
  __host__ __device__ void 	setId(int newId)  { id = newId;}
  __host__ __device__ int 	getLevel() const { return level;}
  __host__ __device__ void 	setLevel(int lvl)  { level = lvl;}
  __host__ __device__ void 	setChild(int child, int index)  { chlildren[index] = child; }
  __host__ __device__ void 	setLBounds(RectCuda rect)  { bounds = rect;}
  __host__ __device__ void 	setOwnRectOff(int ownOff)  {  startOwnOff = ownOff; }
  __host__ __device__ RectCuda  getBounds() const { return bounds; }
  __host__ __device__ int 	rectCount() const{  return endOff - startOff;}
  __host__ __device__ void 	setOff(int start, int end) { startOff = start;endOff = end;}
  __host__ __device__ int 	startRectOff() const{  return startOff; }
  __host__ __device__ int 	ownRectOff() const{  return startOwnOff; }
  __host__ __device__ int 	endRectOff() const{  return endOff;}
  __host__ __device__ int       child(const int index) const { return chlildren[index]; }
  __host__ __device__ __forceinline__
		      float2    getCenter()
  {
    float centerX =  bounds.topLeft.x + (bounds.bottmRight.x - bounds.topLeft.x) / 2;
    float centerY =  bounds.topLeft.y + (bounds.bottmRight.y - bounds.topLeft.y) / 2;
    return make_float2(centerX,centerY);
  }

private:
  int        id; // indeks to globalnej tablicy węzłów
  int        chlildren[NODES_NUMBER];
  int        startOff; // indesky do tablicy gdzie rozpoczynaja sie obiekty znajdujace sie w tym dzieciach tego wezla
  int 	     startOwnOff; // tu zaczynaja sie recty tego obiektu
  int	     endOff;
  int	     level;
  RectCuda   bounds;

};



#endif /* QUADTREE_H_ */
