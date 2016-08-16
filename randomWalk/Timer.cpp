#include "Timer.h"

Timer::Timer()
{
}

Timer::~Timer()
{
}

Timer &Timer::getInstance()
{
    static Timer instance;
    return instance;
}

void Timer::start(std::string const& name)
{
    ResultMapIt it = resultMap.find(name);

    if (it == resultMap.end())
        resultMap[name] = ResultsData(0, timeNow());
    else
    {
        resultMap[name].startPoint = timeNow();
    }
}

void Timer::stop(std::string const& name)
{
    ResultMapIt it = resultMap.find(name);

    if (it != resultMap.end())
    {
        resultMap[name].resultsSum += duration(timeNow() - resultMap[name].startPoint);
        resultMap[name].count++;
    }
}

long long Timer::stop()
{
    return duration(timeNow() - startTime);
}


void Timer::printResults()
{
    for (ResultMapIt it = resultMap.begin(); it != resultMap.end(); ++it)
    {
        long long avarage = it->second.resultsSum / it->second.count;
        ErrorHandler::getInstance() << it->first.c_str() << ":\n" <<
            "    Avarage execution time: " << avarage << "us\n" <<
            "    Total execution time: " << it->second.resultsSum << "us\n" <<
            "    Executions count: " << it->second.count << "\n";
    }
}

void Timer::updateMap(std::string const name, long long value)
{
    ResultMapIt it = resultMap.find(name);

    if (it == resultMap.end())
        resultMap[name] = ResultsData(1, timeNow());
    else
    {
        resultMap[name].count++;
        resultMap[name].resultsSum += value;
    }
}
