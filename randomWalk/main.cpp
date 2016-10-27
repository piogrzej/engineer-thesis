#include "mainFunctions.h"
#include "Parser.h"
#include "Logger.h"
#include "Timer.h"
#include "parallelFunc.h"

#include <iostream>

#define DEFAULT_PATH "../generatedtests/test1000000.txt"
#define DEFAULT_RECT 10
#define DEFAULT_ITERATION 1000

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

    if(GPU_FLAG &&       !initCuda(argc,argv))
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

	//generateTestFile(RectHost(point(0.,0.),point(10000000.,10000000. )),300000);
    return 0;
}
