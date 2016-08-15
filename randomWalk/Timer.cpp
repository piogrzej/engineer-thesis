#include "Timer.h"

#define timeNow() std::chrono::high_resolution_clock::now()
#define duration(a) std::chrono::duration_cast<std::chrono::microseconds>(a).count()

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

template<typename ObjectType, typename RetType,typename ...Args>
RetType Timer::measure(std::string name, ObjectType& object, RetType (ObjectType::*method)(Args...), Args &&...args)
{
    startTime = timeNow();

    auto ret = (object.*method)(std::forward<Args>(args)...);

    long long restult = stop();

    ResultMapIt it = resultMap.find(name);

    if (it == resultMap.end())
        resultMap[name] = ResultPair(0, result);
    else
    {
        resultMap[name].count++;
        resultMap[name] += result;
    }

    return ret;
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
