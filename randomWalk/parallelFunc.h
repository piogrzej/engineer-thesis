#ifndef CUDA_FUNC
#define CUDA_FUNC

extern bool initCuda(int argc, char **argv);
extern floatingPoint getAvgPathLenCUDA(char* path, int ITER_NUM,int RECT_ID);

#endif
