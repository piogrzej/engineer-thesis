#pragma once

#include <iostream>
#include <fstream>

#define LOG_FILE_NAME        "errorLog.txt"
#define TIME_LOG_NAME        "timeLog.txt"
#define INITIAL_TEXT_CONSOLE(NAME) "Log zostal zapisany do " NAME
#define INITIAL_TEXT_LOG "Praca Inzynierska \nRandom Walk\nAutorzy: Piotr Grzejszczyk, Marcin Knap \n\n"

class AbstractLogger
{
public:
    template< typename T>
    AbstractLogger & operator << (const T & itemToLog);
    template< typename T>
    AbstractLogger & operator >> (const T & itemToLog);


protected:
    std::fstream logFile;

             AbstractLogger(std::string name, std::ios_base::openmode  mode);
    virtual ~AbstractLogger();
};

template<typename T>
inline AbstractLogger & AbstractLogger::operator<<(const T & itemToLog)
{
    logFile << itemToLog;
    return *this;
}

template<typename T>
inline AbstractLogger & AbstractLogger::operator>>(const T & itemToConsole)
{
    std::cout << itemToConsole;
    logFile << itemToConsole;
    return *this;
}

class ErrorLogger : public AbstractLogger
{
public:
    static ErrorLogger& getInstance();

private:
    std::fstream logFile;

    ErrorLogger();
};

class TimeLogger : public AbstractLogger
{
public:
    static TimeLogger& getInstance();

private:
    std::fstream logFile;

    TimeLogger();
};
