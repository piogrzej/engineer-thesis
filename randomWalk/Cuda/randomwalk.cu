#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <vector>

#include "createquadtree.h"
#include "../Parser.h"
#include "../Logger.h"
#include "../Timer.h"
// TO DO: brzydkie kopiowanie, trzeba poprawić
// TO DO: wykrywanie ilosci threadow, thread/block, (cudaDeviceProp)
void randomWalkCUDA(char* path, int ITER_NUM, int RECT_ID)
{
    ErrorLogger::getInstance() >> "Random Walk CUDA\n";
    Timer::getInstance().start("Parser");
    Parser parser(path, "<<");
    const std::vector<RectHost>& layer = parser.getLayerAt(0); // na razie 0 warstwa hardcode
    RectHost const& spaceSize = parser.getLayerSize(0);
    Timer::getInstance().stop("Parser");
    QuadTreeManager treeMng = createQuadTree(layer,spaceSize,false);


    Timer::getInstance().printResults();
}
