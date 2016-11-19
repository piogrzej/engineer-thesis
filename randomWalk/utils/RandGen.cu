#include "RandGen.h"
#include "Logger.h"

#include <fstream>


__device__
__host__ int RandGen::nextIndexRand(floatingPoint rand)
{
    for (int i = 0; i <= NSAMPLE; ++i)
    {
        if (intg[i] <= rand && intg[i + 1] > rand)
            return i;
    }
}

__device__
__host__ int RandGen::nextIndex(int thC)
{
    if(indexPtrs[thC] >= GEN_COUNT)
        indexPtrs[thC] = 0;
    return indecies[indexPtrs[thC]++];
}

__device__
__host__ void RandGen::resetIndex()
{
    for(int i = 0; i < count; i++)
        indexPtrs[i] = 0;
}
void RandGen::initOnceDeterm()
{
#ifdef _WIN32
    rng_init(3);//inicjalizacja genaeratora
#elif __linux__
    rng_init(1);//inicjalizacja genaeratora
#endif
    initRand();
    std::fstream randFile;
    randFile.open("rand.data", std::ios::out);
    floatingPoint r;
    int index;
    for(int i = 0; i < GEN_COUNT; i++)
    {
        r     = myrand() / (floatingPoint)(MY_RAND_MAX);
        index = nextIndex(r);
        randFile << index << " ";
    }
    randFile.close();
    printf("Utworzono pomyslnie\n");

}

__host__ void RandGen::initDeterm(int thC)
{
    std::fstream randFile;
    randFile.open("rand.data", std::ios::in);
    count = thC;
    if(!randFile.is_open())
    {
        ErrorLogger::getInstance() >> "Nie ma pliku rand.data!\n";
        exit(0);
    }
    int index;
    for(int i = 0; i < GEN_COUNT; i++)
       {
           randFile >> index;
           indecies[i++] = index;
       }

}

__host__ void RandGen::initRand()
{
    REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE];
    precompute_unit_square_green(g, dgdx, dgdy, intg, NSAMPLE);
}
