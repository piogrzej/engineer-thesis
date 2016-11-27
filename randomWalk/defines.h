#ifndef DEFINES_H
#define DEFINES_H

//main.cpp defines
#define DEFAULT_PATH "../tests/test"		//DOMYSLNA SCIEZKA DO PLIKU
#define DEFAULT_RECT 10						//DOMYSLNY ID OBIEKTU
#define DEFAULT_ITERATION 372				//DOMYSLNA ILOSC ITERACJI
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
#define DEFAULT_LAYER 0						//NUMER DOMYSLNEJ WARSTWY

#define HELP_TEXT "\t-S\t--source\t\tścieżka do pliku wejściowego\
\n\t-M\t--measure\t\turuchamia tryb pomiaru\
\n\t-O\t--object\t\tID obiektu\
\n\t-I\t--iterations\t\tliczba iteracji\
\n\t-G\t--GPU\t\t\turuchamia werjse CUDA błądzenia przypadkowego\
\n\t-L\t--layer\t\t\tnumer warstwy\
\n\t\t--help\t\t\twyświetla tą informacje\
\n\tNie musisz podawać argumentów,\
\n\tw takim przypadku program zostanie uruchomiony\
\n\tz domyślnymi opcjami, to znaczy:\
\n\t-G -I 1000 -O 10 -S ../tests/test -L 0\n"

#endif //DEFINES_H
