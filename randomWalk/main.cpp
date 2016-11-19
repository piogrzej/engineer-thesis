#include "randomwalk.h"
#include "defines.h"
#include "utils/RandGen.h"

#include <iostream>



inline bool checkFile(char* name) 
{
    std::ifstream f(name);
    return f.good();
}

int main(int argc, char *argv[])
{
    Tree *mainTree;
    char* path=		DEFAULT_PATH;
    bool GPU_FLAG=	DEFAULT_GPU_USAGE;
    int rectNum = 	DEFAULT_RECT;
    int iterNum = 	DEFAULT_ITERATION;
    bool measure =	DEFAULT_MEASURE;

    if(argc>0)
    {
    	for(int i=0; i < argc; ++i)
    	{
    		std::string option(argv[i]);
    		if(option == "--help")
    		{
    			printf("%s",HELP_TEXT);
    			if(argc==2)
    				return 0;
    		}
    		else if(option == "-G" || option == "--GPU")
    		{
    			GPU_FLAG = true;
    		}
    		else if(option == "-S" || option == "--source")
    		{
    			if(i+1 < argc)
    			{
    				path = argv[++i];
    			}
    			else
    			{
    				printf("--source option requires one argument.");
    				return -1;
    			}
    		}
    		else if(option == "-I" || option == "--iterations")
			{
				if(i+1 < argc)
				{
					try
					{
						iterNum = std::stoi(std::string(argv[++i]));
					}
					catch (const std::invalid_argument& ia)
					{
						ErrorLogger::getInstance() >> "Invalid argument: " >> ia.what() >> '\n';
						return 0;
					}
				}
				else
				{
					printf("--iterations option requires one argument.");
					return -1;
				}
			}
    		else if(option == "-O" || option == "--object")
			{
				if(i+1 < argc)
				{
					try
					{
						rectNum = std::stoi(std::string(argv[++i]));
					}
					catch (const std::invalid_argument& ia)
					{
						ErrorLogger::getInstance() >> "Invalid argument: " >> ia.what() >> '\n';
						return 0;
					}
				}
				else
				{
					printf("--object option requires one argument.");
					return -1;
				}
			}
    		else if(option == "-M" || option == "--measure")
    		{
    			measure = true;
    		}
    	}
    }

    if(GPU_FLAG && !initCuda(argc,argv))
    	return 0;


	if (false == checkFile(path))
	{
		ErrorLogger::getInstance() >> "No such file!";
		return 0;
	}

    runRandomWalk(path, iterNum, rectNum,GPU_FLAG,measure);

    return 0;
}

/*    std::vector<unsigned int> testsSizes;

    for(int i = 5000; i <= 100000; i += 5000)
        testsSizes.push_back(i);*/
 /*   for(int i = 100000; i < 1000000; i += 100000)
        testsSizes.push_back(i);*/
    //testsSizes.push_back(99000);

/*    TestGenerator gen(testsSizes);
    if(gen.generate())
    {
        ErrorLogger::getInstance() >> "Stworzono testy pomyslnie\n";
        PerformanceComparer comparer(gen.getTestsPaths());
        comparer.compareRandomWalk(1000);
        comparer.printResults();
    }
    else
    {
        ErrorLogger::getInstance() >> "Błąd przy tworzeniu testów\n";

    }*/
