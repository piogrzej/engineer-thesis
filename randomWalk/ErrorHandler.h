#pragma once

#include <iostream>
#include <fstream>

#define LOG_FILE_NAME        "log.txt"
#define INITIAL_TEXT_CONSOLE "Log zostal zapisany do " LOG_FILE_NAME
#define INITIAL_TEXT_LOG "Praca Inzynierska \n Autorzy: Piotr Grzejszczyk, Marcin Knap \n\n" \
					     "Random Walk ver. 0.2 - LOG: \n\n\n"

class ErrorHandler
{
public:
	template< typename T>
	ErrorHandler & operator << (const T & itemToLog);
	template< typename T>
	ErrorHandler & operator >> (const T & itemToLog);

	static ErrorHandler& getInstance();

private:
	std::fstream logFile;

	ErrorHandler();
	~ErrorHandler();
};

template<typename T>
inline ErrorHandler & ErrorHandler::operator<<(const T & itemToLog)
{
	logFile << itemToLog;
	return *this;
}

template<typename T>
inline ErrorHandler & ErrorHandler::operator>>(const T & itemToConsole)
{
	std::cout << itemToConsole;
	logFile << itemToConsole;
	return *this;
}
