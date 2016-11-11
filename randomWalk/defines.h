#ifndef DEFINES_H
#define DEFINES_H

//main.cpp defines
#define DEFAULT_PATH "../tests/test1"		//DOMYSLNA SCIEZKA DO PLIKU
#define DEFAULT_RECT 5						//DOMYSLNY ID OBIEKTU
#define DEFAULT_ITERATION 10				//DOMYSLNA ILOSC ITERACJI
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
#define MAX_LEVELS 20
#define BIGGEST_SQUARE_INIT_FACTOR 0.05		//O ILE POWIEKSZAMY RECT STARTOWY PRZY STARCIE BLADZENIA
#define GAUSSIAN_ACCURACY 10
#define MAX_OBJECTS 4						//W WERSJI CPU QUADTREE ILOSC OBKETOW W NODZIE
//randomWalk defines
#define NSAMPLE 200							//NA ILE CZESCI JEST DZIELONY OBWOD ELEMENTU
#define GPU_FLAG 1							//FLAGA KTORA OKRESLA CZY DOMYSLNIE UZYWAMY randomWalkCUDA

/* TODO: PORZADEK W TYCH DEFINACH
 * getAvgPathLen.cpp:9 #define MEASURE_MODE
 * mianFunctions.cpp:9 #define MEASURE_MODE
 */

#endif //DEFINES_H