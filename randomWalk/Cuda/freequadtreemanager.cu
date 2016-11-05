#include "mainkernels.h"

void freeQuadTreeManager(QuadTreeManager* qtm)
{
	QuadTreeManager *hostQTM= new QuadTreeManager();
	cudaMemcpy(hostQTM,qtm,sizeof(QuadTreeManager),cudaMemcpyDeviceToHost);
    cudaFree(hostQTM->d_intg);
    cudaFree(hostQTM->nodes);
    cudaFree(hostQTM->rects);
    cudaFree(hostQTM->root);
    cudaFree(qtm);
}
