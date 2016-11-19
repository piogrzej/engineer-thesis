#ifndef DEFINES_H
#define DEFINES_H

//main.cpp defines
#define DEFAULT_PATH "../../tests/test"		//DOMYSLNA SCIEZKA DO PLIKU
#define DEFAULT_RECT 10						//DOMYSLNY ID OBIEKTU
#define DEFAULT_ITERATION 1000				//DOMYSLNA ILOSC ITERACJI
//loger.h defines
#define LOG_FILE_NAME "errorLog.txt"
#define TIME_LOG_NAME "timeLog.txt"
#define COMPARER "compareLog.txt"
#define INITIAL_TEXT_CONSOLE(NAME) "Log zostal zapisany do " NAME
#define INITIAL_TEXT_LOG "Praca Inzynierska \nRandom Walk\nAutorzy: Piotr Grzejszczyk, Marcin Knap \n\n"
//parser defines
#define MAX_LINE_SIZE 50
#define LINE_HEADER "rect"
#define BOUNDS_MUL_FACTOR 0.01				//O ILE POWIEKSZAMY PRZESTRZEN NA STARCIE
//quadTree defines
#define NODES_NUMBER 4						//ILOSC DZIECI WEZLA
#define MAX_LEVELS 10
#define BIGGEST_SQUARE_INIT_FACTOR 0.05		//O ILE POWIEKSZAMY RECT STARTOWY PRZY STARCIE BLADZENIA
#define GAUSSIAN_ACCURACY 10
#define MAX_OBJECTS 16						//W WERSJI CPU QUADTREE ILOSC OBKETOW W NODZIE
//randomWalk defines
#define NSAMPLE 200							//NA ILE CZESCI JEST DZIELONY OBWOD ELEMENTU
#define DEFAULT_GPU_USAGE 0					//CZY DOMYSLNIE UZYWAMY GPU
#define DEFAULT_MEASURE 0					//CZY DOMYSLNIE UZYWAMY MAUSRE MODE

#define HELP_TEXT "\t-S\t--source\t\tsource file path\
\n\t-M\t--measure\t\tturns on measure mode\
\n\t-O\t--object\t\tobject ID from [0-(n-1)]\
\n\t-I\t--iterations\t\tnumber of iterations\
\n\t-G\t--GPU\t\t\trun CUDA verison of random walk\
\n\t\t--help\t\t\tdisplay this information\
\n\tYou don't need to specify any options\
\n\tin that case program will be lunched\
\n\twith default options, wich are:\
\n\t-G -I 1000 -O 10 -S ../tests/test\n"

#endif //DEFINES_H
