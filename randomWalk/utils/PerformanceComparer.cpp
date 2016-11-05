/*
 * PerformanceComparer.cpp
 *
 *  Created on: 4 lis 2016
 *      Author: mknap
 */

#include "PerformanceComparer.h"
#include "Logger.h"
#include "Timer.h"
#include "../CPU/mainFunctions.h"
#include "../Cuda/mainkernels.h"

#include <string>

std::map<Device,std::string> deviceMap =
{
        { Device::Cpu, "CPU" },
        { Device::Gpu, "GPU" },
};

std::map<Component,std::string> componentMap =
{
        { Component::CreateTree, "CreateTree" }
};


PerformanceComparer::PerformanceComparer(std::vector<std::string> const& testsPaths)
: testsPaths(testsPaths), parser("<<"), dParser("<<")
{
    for(auto const& path : testsPaths)
    {
        parser.parse(path);
        dParser.parse(path);
    }
    ErrorLogger::getInstance() >> "Rozmiar parserów: " >> parser.getLayerCount() >> " " >>
                                                          parser.getLayerCount() >> "\n";
}

void PerformanceComparer::compareCreatingTree()
{
    initInfo(Component::CreateTree);
    std::string cpuName = deviceMap[Device::Cpu] + componentMap[Component::CreateTree];
    std::string gpuName = deviceMap[Device::Gpu] + componentMap[Component::CreateTree];

    for(int i = 0; i < testsPaths.size(); i++)
    {
        runCreateTreeCpu(i,cpuName);
        runCreateTreeGpu(i,gpuName);
    }

}

void PerformanceComparer::runCreateTreeCpu(int layerId,std::string const& name)
{
    Layer const& layer = parser.getLayerAt(layerId);
    ErrorLogger::getInstance() >> name >> "  " >> layer.size()>> "\n";
    for(int i = 0; i < EXEC_PER_TEST; i++)
    {
        Tree* root = new Tree(0,layer.size(),parser.getLayerSize(layerId));
        Timer::getInstance().start(name);
        createTree(root,layer);
        Timer::getInstance().stop(name);
        root->clear();
        delete root;
    }
    resultsCpu[layer.size()] = Timer::getInstance().getAvgResult(name);
    Timer::getInstance().clear();
}

void PerformanceComparer::runCreateTreeGpu(int layerId,std::string const& name)
{
    d_Layer const& layer = dParser.getLayerAt(layerId);
    ErrorLogger::getInstance() >> name >> "  " >> layer.size() >> "\n";

    for(int i = 0; i < EXEC_PER_TEST; i++)
      {
          Timer::getInstance().start(name);
          QuadTreeManager* qtm = createQuadTree(layer,dParser.getLayerSize(layerId),false);
          Timer::getInstance().stop(name);
          // tu nie iwem o co chodzi ...
        //  freeQuadTreeManager(qtm);
      }
    resultsGpu[layer.size()] = Timer::getInstance().getAvgResult(name);
    Timer::getInstance().clear();
}

void PerformanceComparer::initInfo(Component comp)
{
    CompareLogger::getInstance() << "Porównanie komponentu: " + componentMap[comp] << "\n";
}

void PerformanceComparer::printResults()
{
    std::string header = "TestSize " +
            deviceMap[Device::Cpu] + " " +
            deviceMap[Device::Gpu] + "\n";

    CompareLogger::getInstance() << header;

    for(auto pair : resultsCpu)
    {
        CompareLogger::getInstance() << pair.first << " "  <<
                                        pair.second << " " <<
                                        resultsGpu[pair.first] << "\n";
    }
}
