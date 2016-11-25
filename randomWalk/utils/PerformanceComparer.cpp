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
#include "RandGen.h"
#include <stdio.h>

#include <string>

std::map<Device,std::string> deviceMap =
{
        { Device::Cpu, "CPU" },
        { Device::Gpu, "GPU" },
};

std::map<Component,std::string> componentMap =
{
        { Component::CreateTree, "CreateTree" },
        { Component::RandomWalk, "RandomWalk" }
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

void PerformanceComparer::compareRandomWalk(int numOfIteratins)
{
	initInfo(Component::RandomWalk);
	std::string cpuName = deviceMap[Device::Cpu] + componentMap[Component::RandomWalk];
	std::string gpuName = deviceMap[Device::Gpu] + componentMap[Component::RandomWalk];
	for(int i=0; i<this->testsPaths.size(); ++i)
	{
		runRandomWalkCpu(i,cpuName,parser.getLayerAt(i).size() / 2,numOfIteratins);
		runRandomWalkGpu(i,gpuName,parser.getLayerAt(i).size() / 2,numOfIteratins);
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
    QuadTreeManager* qtm;
    for(int i = 0; i < EXEC_PER_TEST; i++)
      {
          cudaDeviceSynchronize();
          Timer::getInstance().start(name);
          qtm = createQuadTree(layer,dParser.getLayerSize(layerId),0,false);
          Timer::getInstance().stop(name);
          cudaDeviceReset();
          // tu nie iwem o co chodzi ...
      }
    resultsGpu[layer.size()] = Timer::getInstance().getAvgResult(name);
    Timer::getInstance().clear();
}

void PerformanceComparer::runRandomWalkCpu(int layerId,std::string const& name, int RECT_ID, int ITER_NUM)
{
    RandGen gen;
    gen.initDeterm( ITER_NUM);
    gen.initPtrs();
    Layer const& layer = parser.getLayerAt(layerId);
    ErrorLogger::getInstance() >> name >> "  " >> layer.size()>> "\n";
    for(int i = 0; i < EXEC_PER_TEST; i++)
    {
        Tree* root = new Tree(0,layer.size(),parser.getLayerSize(layerId));
        createTree(root,layer);
        RectHost start = layer.at(RECT_ID);
        REAL64_t g[NSAMPLE], dgdx[NSAMPLE], dgdy[NSAMPLE], intg[NSAMPLE + 1];
        precompute_unit_square_green(g,dgdx,dgdy,intg,NSAMPLE);// niech lidzy zeby bylo wiarygodnie
        int pos, sumPointCount = 0;
        int* foundedRectCount = new int[layer.size()+1];
		std::fill(foundedRectCount, foundedRectCount + layer.size()+1, 0);
        Timer::getInstance().start(name);
        int errors = 0;
		for (int i = 0; i < ITER_NUM; i++)
		{
			int counter;
			RectHost founded = RandomWalk(start, root, counter,gen,i);
			if(-1 == founded.topLeft.x &&
			   -1 == founded.topLeft.y &&
			   -1 == founded.bottomRight.x &&
			   -1 == founded.bottomRight.y)
				++foundedRectCount[layer.size()];
			else
			{
				pos = getRectIt(layer,founded);
				if (pos != -1)
					foundedRectCount[pos] += 1;
				else
					errors++;
			}
			sumPointCount += counter;
		}
        Timer::getInstance().stop(name);
        root->clear();
        delete root;
        gen.resetIndex();// resetujemy zeby sciezka byla ta sama
    }
    resultsCpu[layer.size()] = Timer::getInstance().getAvgResult(name);
    Timer::getInstance().clear();
}

void PerformanceComparer::runRandomWalkGpu(int layerId,std::string const& name, int RECT_ID, int ITER_NUM)
{
    d_Layer const& layer = dParser.getLayerAt(layerId);
    ErrorLogger::getInstance() >> name >> "  " >> layer.size() >> "\n";
    QuadTreeManager* qtm;
    RandGen gen;
    gen.initDeterm(ITER_NUM);
    for(int i = 0; i < EXEC_PER_TEST; i++)
	{
		  cudaDeviceSynchronize();
		  qtm = createQuadTree(layer,dParser.getLayerSize(layerId),RECT_ID,false);
		  Timer::getInstance().start(name);
		  unsigned int output[ITER_NUM];
		  unsigned int* d_output;
		  unsigned int outputSize = ITER_NUM * sizeof(unsigned int);
		  cudaMalloc((void **)&d_output,outputSize);
		  randomWalkCudaWrapper(ITER_NUM,qtm,d_output,gen,time(NULL));
		  cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost);
		  freeQuadTreeManager(qtm);
		  cudaFree(d_output);
		  cudaDeviceReset();
		  countAvg(output,ITER_NUM);
		  Timer::getInstance().stop(name);
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
    for(auto const& file : testsPaths)
    {
        //if(0 == remove(file.c_str()))
          //  ErrorLogger::getInstance() >> "Usunieto " >> file >> "\n";
    }
}
