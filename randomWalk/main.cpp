#include "randomwalk.h"

#include <iostream>

#define HELP_TEXT "\t-S\t--source\t\tsource file path\
\n\t-M\t--measure\t\tturns on measure mode\
\n\t-O\t--object\t\tobject ID\
\n\t-I\t--iterations\t\tnumber of iterations\
\n\t-G\t--GPU\t\t\trun CUDA verison of random walk\
\n\t-L\t--layer\t\t\tlayer numeber\
\n\t\t--help\t\t\tdisplay this information\
\n\tYou don't need to specify any options\
\n\tin that case program will be lunched\
\n\twith default options, wich are:\
\n\t-G -I 1000 -O 10 -S ../tests/test -L 0\n"

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
    //GPU_FLAG = false;
    int rectNum = 	DEFAULT_RECT;
    int iterNum = 	DEFAULT_ITERATION;
    bool measure =	DEFAULT_MEASURE;
    int layer = 	DEFAULT_LAYER;

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
						return -1;
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
    		else if(option == "-L" || option == "--layer")
			{
    			if(i+1 < argc)
				{
					try
					{
						layer = std::stoi(std::string(argv[++i]));
						if(layer<0)
						{
							ErrorLogger::getInstance() >> "Invalid argument: layer, can't be lower then 0!\n'";
							return -1;
						}
					}
					catch (const std::invalid_argument& ia)
					{
						ErrorLogger::getInstance() >> "Invalid argument: " >> ia.what() >> '\n';
						return -1;
					}
				}
				else
				{
					printf("--object option requires one argument.");
					return -1;
				}
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

    runRandomWalk(path, iterNum, rectNum,GPU_FLAG,measure,layer);

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

    return 0;
}
