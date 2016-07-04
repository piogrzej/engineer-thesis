#pragma once

#include <chrono>

#include "ErrorHandler.h"

typedef std::chrono::high_resolution_clock::time_point TimePoint;

class Timer
{

public:
	Timer();
	~Timer();

	void start();
	void stop(std::string const& title);

private:
	TimePoint startTime;

};

