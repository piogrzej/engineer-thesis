#include "d_quadtree.h"
#include <vector>

#ifdef IGNORE_OUT_OF_BOUNDS_CASE
#define SPECIAL_ACTION continue
#define SPECIAL_VALUE_BOOLEAN false
#else
#define SPECIAL_ACTION break
#define SPECIAL_VALUE_BOOLEAN true
#endif

__global__ void randomWalkCuda(QuadTreeManager* quadTree, int RECT_ID,floatingPoint *output,unsigned int randomSeed=time(NULL));
QuadTreeManager* createQuadTree(const std::vector<d_Rect>& layer,d_Rect const& spaceSize,bool doCheck);
QuadTreeManager* randomWalkCudaInit(char* path);
bool initCuda(int argc, char **argv);
void freeQuadTreeManager(QuadTreeManager* qtm);
floatingPoint getAvgPathLen(char* path, int ITER_NUM,int RECT_ID);
