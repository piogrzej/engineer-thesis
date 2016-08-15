#pragma once

#include <chrono>
#include <utility>
#include <map>

#include "ErrorHandler.h"

struct ResultsData;
typedef std::chrono::high_resolution_clock::time_point TimePoint;
typedef std::map<std::string, ResultsData> ResultMap;
typedef ResultMap::iterator ResultMapIt;

struct ResultsData
{
    long      count;
    long long resultsSum;
    TimePoint startPoint;

    ResultsData() {}
    ResultsData(long count, TimePoint startPoint) : count(count), startPoint(startPoint) { resultsSum = 0;  }
};

class Timer
{

public:
	Timer();
	~Timer();

    static Timer&      getInstance();
	void               start(std::string const& title);
	void               stop(std::string const& title);
    long long          stop();
    void               printResults();

    template<typename ObjectType, typename RetType, typename ...Args>
    RetType     measure(std::string name, ObjectType& object, RetType (ObjectType::*method)(Args...), Args &&... args);

private:
	TimePoint startTime;
    ResultMap resultMap;
};

