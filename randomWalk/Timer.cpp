#include "Timer.h"


#define timeNow() std::chrono::high_resolution_clock::now()
#define duration(a) std::chrono::duration_cast<std::chrono::microseconds>(a).count()

Timer::Timer()
{
}

Timer::~Timer()
{
}

void Timer::start()
{
	startTime = timeNow();
}

void Timer::stop(std::string const& title)
{
	ErrorHandler::getInstance() << title.c_str() << duration(timeNow() - startTime) << " micros \n";
}
