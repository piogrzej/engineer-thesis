#ifndef MAINKERNELS_H
#define MAINKERNELS_H

#include "d_quadtree.h"
#include <vector>

class RandGen;

void randomWalkCudaWrapper(int dimThread,QuadTreeManager* quadTree,unsigned int *output,d_Rect* d_rectOutput,unsigned long long randomSeed);
QuadTreeManager* createQuadTree(const std::vector<d_Rect>& layer,d_Rect const& spaceSize,int RECT_ID,bool doCheck);
QuadTreeManager* randomWalkCudaInit(char* path,bool measure,int RECT_ID,int layerID);
bool initCuda(int argc, char **argv);
void freeQuadTreeManager(QuadTreeManager* qtm);
floatingPoint getAvgPathLenCUDA(char* path, int ITER_NUM,int RECT_ID,bool measure,int layerID);
floatingPoint countAvg(unsigned int output[],int ITER_NUM);

#endif
