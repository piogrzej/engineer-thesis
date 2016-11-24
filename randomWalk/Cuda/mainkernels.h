#ifndef MAINKERNELS_H
#define MAINKERNELS_H

#include "d_quadtree.h"
#include <vector>

void randomWalkCudaWrapper(int dimBlck,int dimThread,QuadTreeManager* quadTree, int RECT_ID,unsigned int *output,unsigned long long randomSeed);
QuadTreeManager* createQuadTree(const std::vector<d_Rect>& layer,d_Rect const& spaceSize,bool doCheck);
QuadTreeManager* randomWalkCudaInit(char* path,bool measure,int layer_id);
bool initCuda(int argc, char **argv);
void freeQuadTreeManager(QuadTreeManager* qtm);
floatingPoint getAvgPathLenCUDA(char* path, int ITER_NUM,int RECT_ID,bool measure,int layer_id);
floatingPoint countAvg(unsigned int output[],int ITER_NUM);

#endif
