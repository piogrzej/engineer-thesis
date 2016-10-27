#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <helper_cuda.h>

#include "Logger.h"

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

    printf("GPU: %s (SM %d.%d)\n", deviceProps.name, deviceProps.major, deviceProps.minor);

    if (!cdpCapable)
    {
        ErrorLogger::getInstance() >> "RandomWalk potrzebuje SM 3.5 lub wyzszej do CUDA Dynamic Parallelism.\n";
        return false;
    }
    return true;
}
