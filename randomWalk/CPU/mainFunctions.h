#ifndef MAINFUNCTIONS_H
#define MAINFUNCTIONS_H

#ifdef IGNORE_OUT_OF_BOUNDS_CASE
#define SPECIAL_ACTION continue
#define SPECIAL_VALUE_BOOLEAN false
#else
#define SPECIAL_ACTION break
#define SPECIAL_VALUE_BOOLEAN true
#endif
#define GPU_FLAG 1

#include<stdlib.h>//atoi
#include<string.h>//strcmp
#include<iostream>

#include "quadTree.h"
#include "../green/green.h"
#include "Parser.h"

#define _CRT_SECURE_NO_WARNINGS
#define LINE_BUFFOR_SIZE 40
#define NSAMPLE 200
#define GAUSSIAN_ACCURACY 10
#define BIGGEST_SQUARE_INIT_FACTOR 0.05
#define GEN_TEST_FILE_PERCET_OF_RECTS_IN_LAYER 0.25;                            


void createTree(Tree * mainTree,Layer const& layer);
RectHost RandomWalk(RectHost const& R, Tree* mainTree, int& pointCount);
int getIndex(REAL64_t intg[NSAMPLE + 1], floatingPoint rand);
void runRandomWalk(char* path, int ITER_NUM, int RECT_ID);
void printList(std::list<RectHost> input);
void generateTestFile(RectHost space, unsigned long long numOfRects);               //przykadlowe uzycie: ilosc elementow 100, rozmair przestrzeni 10000

#endif
