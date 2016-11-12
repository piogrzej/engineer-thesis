#ifndef MAINFUNCTIONS_H
#define MAINFUNCTIONS_H

#include<stdlib.h>//atoi
#include<string.h>//strcmp
#include<iostream>
#include <string>

#include "quadTree.h"
#include "../green/green.h"
#include "Parser.h"
#include "../Cuda/mainkernels.h"

void createTree(Tree * mainTree,Layer const& layer);
RectHost RandomWalk(RectHost const& R, Tree* mainTree, int& pointCount,REAL64_t intg[NSAMPLE + 1]);
int getIndex(REAL64_t intg[NSAMPLE + 1], floatingPoint rand);
void runRandomWalk(char* path, int ITER_NUM, int RECT_ID, bool GPU_FLAG,bool measure);
void printList(std::list<RectHost> input);

#endif
