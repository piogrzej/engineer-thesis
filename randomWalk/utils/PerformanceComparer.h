/*
 * PerformanceComparer.h
 *
 *  Created on: 4 lis 2016
 *      Author: mknap
 */

#ifndef PERFORMANCECOMPARER_H_
#define PERFORMANCECOMPARER_H_

#include <vector>
#include <list>
#include <map>

#include "../CPU/Parser.h"
#include "../Cuda/d_parser.h"

enum class Device
{
    Gpu,
    Cpu
};
enum class Component
{
    CreateTree,
    RandomWalk,
    //....
};
typedef std::map<int,long long>    ResultsMap;

class PerformanceComparer
{
public:
    PerformanceComparer(std::vector<std::string> const& testsPaths);

    void compareCreatingTree();
    void compareRandomWalk(int numOfIteratins);
    void printResults();

private:
    int                  EXEC_PER_TEST = 10;
    std::vector<std::string>
                         testsPaths;
    ResultsMap           resultsGpu;
    ResultsMap           resultsCpu;
    Parser               parser;
    d_Parser             dParser;

    void  initInfo(Component comp);

    void runCreateTreeCpu(int layerId,std::string const& name);
    void runCreateTreeGpu(int layerId,std::string const& name);
    void runRandomWalkCpu(int layerId,std::string const& name, int RECT_ID, int ITER_NUM);
	void runRandomWalkGpu(int layerId,std::string const& name, int RECT_ID, int ITER_NUM);
};

#endif /* PERFORMANCECOMPARER_H_ */
