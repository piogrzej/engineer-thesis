/*
 * TestGenerator.h
 *
 *  Created on: 4 lis 2016
 *      Author: mknap
 */

#ifndef TESTGENERATOR_H_
#define TESTGENERATOR_H_

#include <vector>
#include <string>

class TestGenerator
{

public:
    TestGenerator(std::vector<unsigned int>  const& rectsCounts);

    bool                       generate(std::string path = "");
    std::vector<std::string>   getTestsPaths() { return genFilesPaths; }

private:
   const unsigned int RECTS_GAP = 100;
        double        RECTS_IN_LAYER = 0.25;
   std::string        extension;
   std::string        fileName;
   std::string        defaultPath;
   std::vector<unsigned int>
                      rectsCounts;
   std::vector<std::string>
                      genFilesPaths;

   bool                       generateTestFile(std::string path,unsigned int rectsCount);
};

#endif /* TESTGENERATOR_H_ */
