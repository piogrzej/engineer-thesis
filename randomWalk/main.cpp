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
    //GPU_FLAG = true;
    int rectNum = 	DEFAULT_RECT;
    int iterNum = 	DEFAULT_ITERATION;
    bool measure =	DEFAULT_MEASURE;
    int layer =		DEFAULT_LAYER;

    if(argc>1)
    {
    	for(int i=1; i < argc; ++i)
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
    				printf("--source wymaga jednego argumentu.");
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
						printf("Niepoprawy argument, to nie jest liczba! \n");
						ErrorLogger::getInstance() >> "Invalid argument: " >> ia.what() >> '\n';
						return 0;
					}
				}
				else
				{
					printf("--iterations wymaga jednego argumentu.");
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
						printf("Niepoprawy argument, to nie jest liczba! \n");
						ErrorLogger::getInstance() >> "Invalid argument: " >> ia.what() >> '\n';
						return 0;
					}
				}
				else
				{
					printf("--object wymaga jednego argumentu.");
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
					}
					catch (const std::invalid_argument& ia)
					{
						printf("Niepoprawy argument, to nie jest liczba! \n");
						ErrorLogger::getInstance() >> "Invalid argument: " >> ia.what() >> '\n';
						return 0;
					}
				}
				else
				{
					printf("--layer wymaga jednego argumentu.");
					return -1;
				}
    		}
    		else
    		{
    			printf("%s",HELP_TEXT);
    			printf("Niepoprawny argument: %s\n",option.c_str());
    			return -1;
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


    return 0;
}
