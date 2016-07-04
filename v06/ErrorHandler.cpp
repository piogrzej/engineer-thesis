#include "ErrorHandler.h"


ErrorHandler & ErrorHandler::getInstance()
{
	static ErrorHandler instance;
	return instance;
}

ErrorHandler::ErrorHandler()
{
	std::cout << INITIAL_TEXT_CONSOLE << std::endl;
	logFile.open(LOG_FILE_NAME, std::ios::out);

}

ErrorHandler::~ErrorHandler()
{
	logFile.close();
}
