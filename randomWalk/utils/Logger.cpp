#include "Logger.h"
#include "../defines.h"



AbstractLogger::AbstractLogger(std::string name, std::ios_base::openmode  mode)
{
    logFile.open(name.c_str(), mode);
    logFile << INITIAL_TEXT_LOG;
}

AbstractLogger::~AbstractLogger()
{
    logFile << "                                      "
               "KONIEC TESTU \n\n";
    logFile.close();
}


ErrorLogger & ErrorLogger::getInstance()
{
    static ErrorLogger instance;
    return instance;
}

ErrorLogger::ErrorLogger() : AbstractLogger(LOG_FILE_NAME,std::ios::out)
{
    std::cout << INITIAL_TEXT_CONSOLE(LOG_FILE_NAME) << std::endl;
}

TimeLogger & TimeLogger::getInstance()
{
    static TimeLogger instance;
    return instance;
}

TimeLogger::TimeLogger() : AbstractLogger(TIME_LOG_NAME, std::ios::app)
{
    std::cout << INITIAL_TEXT_CONSOLE(TIME_LOG_NAME) << std::endl;
}

CompareLogger &CompareLogger::getInstance()
{
    static CompareLogger instance;
    return instance;
}

CompareLogger::CompareLogger() : AbstractLogger(COMPARER, std::ios::app)
{
    std::cout << INITIAL_TEXT_CONSOLE(COMPARER) << std::endl;
}
