#include "randomwalk.h"

#include <iostream>

#define HELP_TEXT "\t-S\t--source\t\tsource file path\
\n\t-M\t--measure\t\tturns on measure mode\
\n\t-O\t--object\t\tobject ID from [0-(n-1)]\
\n\t-I\t--iterations\t\tnumber of iterations\
\n\t-G\t--GPU\t\t\trun CUDA verison of random walk\
\n\t\t--help\t\t\tdisplay this information\n"

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
