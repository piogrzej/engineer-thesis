
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Parser.h"

#include <helper_cuda.h>
#include <stdio.h>


bool initCuda(int argc, char **argv)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return false;
    }
    int cuda_device = findCudaDevice(argc, (const char **)argv);
        cudaDeviceProp deviceProps;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
    int cdpCapable = (deviceProps.major == 3 && deviceProps.minor >= 5) || deviceProps.major >=4;

    printf("GPU: %s ma (SM %d.%d)\n", deviceProps.name, deviceProps.major, deviceProps.minor);

    if (!cdpCapable)
    {
        std::cerr << "RandomWalk potrzebuje SM 3.5 lub wyzszej do CUDA Dynamic Parallelism.\n" << std::endl;
        return false;
    }
    return true;
}


void randomWalkCUDA(char* path, int ITER_NUM, int RECT_ID)
{
    Parser parser(path, "<<");
    const Layer layer = parser.getLayerAt(0);
    Rect const& spaceSize = parser.getLayerSize(0);
}
