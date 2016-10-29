#include "mainkernels.h"

void freeQuadTreeManager(QuadTreeManager* qtm)
{
    cudaFree(qtm->d_intg);
    cudaFree(qtm->nodes);
    cudaFree(qtm->rects);
    cudaFree(qtm->root);
    cudaFree(qtm);
}
