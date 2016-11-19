#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../defines.h"
#include "../green/green.h"

#define GEN_COUNT 10000

typedef float  floatingPoint;

class RandGen
{

public:
    __host__ __device__
                  RandGen() : count(1){}
             void initOnceDeterm(); // Powinno zostać wykonane tylko raz przed wszystkimi testami
             void initDeterm(int thC = 1); // Powinno zostać wykonane raz przed randomwalk gdy deterministycznie
             void initRand(); // raz przed randomWalk gdy niedeterministyczne
__host__ __device__
             void initPtrs()
             {
                 indexPtrs = new int[count];
                 for(int i = 0; i < count;i++)
                     indexPtrs[i] = i;
             }
__host__ __device__
            void freeStck() { delete indexPtrs; }
    __device__
    __host__ void resetIndex();
    __device__
    __host__ int  nextIndex(int id = 0);
    __device__
    __host__ int  nextIndexRand(floatingPoint rand);

    int           indecies[GEN_COUNT];

private:
    int           count;
    int*          indexPtrs;
    REAL64_t      intg[NSAMPLE + 1];

};
