#include "d_quadtree.h"
#include <vector>

QuadTreeManager* createQuadTree(const std::vector<d_Rect>& layer,d_Rect const& spaceSize,bool doCheck);
QuadTreeManager* randomWalkCudaInit(char* path, int ITER_NUM, int RECT_ID);
bool initCuda(int argc, char **argv);
void freeQuadTreeManager(QuadTreeManager* qtm);
__global__ void randomWalkCuda(QuadTreeManager* quadTree, int RECT_ID,floatingPoint *output);
