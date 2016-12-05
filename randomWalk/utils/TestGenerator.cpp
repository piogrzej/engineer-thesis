/*
 * TestGenerator.cpp
 *
 *  Created on: 4 lis 2016
 *      Author: mknap
 */

#include "TestGenerator.h"
#include "../utils/Logger.h"
#include "../CPU/rectHost.h"
#include "../CPU/quadTree.h"
#include "../green/randgen.h"
#include <fstream>

TestGenerator::TestGenerator(std::vector<unsigned int>  const& rectsCounts)
: rectsCounts(rectsCounts)
{
    defaultPath = "";
    fileName = "test";
    extension = ".txt";
}
bool TestGenerator::generate(std::string path)
{
    if(path.empty())
        path = defaultPath;

    genFilesPaths.clear();
    for(auto count : rectsCounts)
    {
        std::string fullPath = path + fileName + std::to_string(count) + extension;
        ErrorLogger::getInstance() >> fullPath >> "\n";
        std::ifstream f(fullPath.c_str());
        if(f.good())
        {
            genFilesPaths.push_back(fullPath);
        	continue;
        }

        if(!generateTestFile(fullPath,count))
            return false;

        genFilesPaths.push_back(fullPath);
    }
    return true;
}

bool TestGenerator::generateTestFile(std::string path,unsigned int rectsCount)
{
    RectHost maxSpace = RectHost(point(0,0),point(rectsCount * RECTS_GAP,rectsCount * RECTS_GAP));
    const int MIN_X = maxSpace.topLeft.x;
    const int MIN_Y = maxSpace.topLeft.y;
    const int MAX_X = maxSpace.bottomRight.x;
    const int MAX_Y = maxSpace.bottomRight.y;
    const unsigned int L = MAX_X - MIN_X;
    const unsigned int H = MAX_Y - MIN_Y;
    const int MAX_L = (L / rectsCount) * RECTS_IN_LAYER;
    const int MAX_H = (H / rectsCount) * RECTS_IN_LAYER;

    Tree *mainTree = new Tree(0, rectsCount, maxSpace);
    std::ofstream output(path,std::ofstream::out);

    if(!output.is_open())
        return false;

    output << "magic\ntech mayukh\ntimestamp 536610539\n<< metal3 >>\n";

    RectHost random;

#ifdef _WIN32
    rng_init(3);//inicjalizacja genaeratora
#elif __linux__
    rng_init(1);//inicjalizacja genaeratora
#endif

    for(unsigned long long i=0; i < rectsCount; ++i)
    {
        do
        {
            random.topLeft.x = MIN_X + myrand() % L;

            if(random.topLeft.x>MAX_L)
                random.topLeft.x - MAX_L;

            random.topLeft.y = MIN_Y + myrand() % H;

            if(random.topLeft.y>MAX_H)
                random.topLeft.y - MAX_H;

            do
            {
                random.bottomRight.x = random.topLeft.x + myrand() % MAX_L;
                random.bottomRight.y = random.topLeft.y + myrand() % MAX_H;
            }
            while((int)(random.bottomRight.x) == (int)(random.topLeft.x) ||
                  (int)(random.bottomRight.y) == (int)(random.topLeft.y));

        }
        while(true == mainTree->checkCollisions(random) &&
                0 != (int)random.getHeigth() &&
                0 != (int)random.getWidth());

        mainTree->insert(random);
        output<<"rect "<<(int)random.topLeft.x<<" "<<(int)random.topLeft.y<<" "<<
                         (int)random.bottomRight.x<<" "<<(int)random.bottomRight.y<<"\n";
    }
    output << "<< metal3 >>";
    printf("generateTestFile: end\n");
    output.close();
    return true;
}
