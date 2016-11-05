#include "CPU/mainFunctions.h"
#include "CPU/Parser.h"
#include "utils/Logger.h"
#include "utils/Timer.h"
#include "parallelFunc.h"
#include "utils/TestGenerator.h"
#include "utils/PerformanceComparer.h"


#include <iostream>

//#define DEFAULT_PATH "../generatedtests/test1000.txt"
#define DEFAULT_PATH "../tests/test"
#define DEFAULT_RECT 10
#define DEFAULT_ITERATION 10

enum PARAMS
{
    PARAM_NAME,
    PARAM_PATH,
    PARAM_RECT,
    PARAM_ITERATIONS,
    PARAMS_COUNT
};

inline bool checkFile(char* name) 
{
    std::ifstream f(name);
    return f.good();
}

int main(int argc, char *argv[])
{
   Tree *mainTree;
    char* path;
    char inputPath[300];//E:\\programowanie\\quadtree\\sigfill_chunk_x.mag
                        //C:\Users\Marcin\Documents\inzynierka\sigfill_chunk_x.gk

    if(GPU_FLAG && !initCuda(argc,argv))
    	return 0;

    if (argc == PARAMS_COUNT)
    {
        path = argv[PARAM_PATH];

        if (false == checkFile(path))
        {
            ErrorLogger::getInstance() >> "No such file!";
            return 0;
        }

        try 
        {
            int rectNum = std::stoi(std::string(argv[PARAM_RECT]));
            int iterNum = std::stoi(std::string(argv[PARAM_ITERATIONS]));

            runRandomWalk(path, iterNum, rectNum);
        }
        catch (const std::invalid_argument& ia)
        {
            ErrorLogger::getInstance() >> "Invalid argument: " >> ia.what() >> '\n';
            return 0;
        }      
    }
    else if (argc > 1)
    {
        ErrorLogger::getInstance() >> "Incorrect number of args!\n";
    }
    else
    {
    	runRandomWalk(DEFAULT_PATH, DEFAULT_ITERATION, DEFAULT_RECT);
    }

    //std::vector<unsigned int> testsSizes =
    //{
       //100,1000,10000,100000,200000,300000,/*400000,500000,
       //600000,700000,800000,900000,1000000*/
    //};

    /*TestGenerator gen(testsSizes);
    if(gen.generate())
    {
        ErrorLogger::getInstance() >> "Stworzono testy pomyslnie\n";
        PerformanceComparer comparer(gen.getTestsPaths());
        comparer.compareCreatingTree();
        comparer.printResults();
    }
    else
    {
        ErrorLogger::getInstance() >> "Błąd przy tworzeniu testów\n";

    }*/

    return 0;
}
