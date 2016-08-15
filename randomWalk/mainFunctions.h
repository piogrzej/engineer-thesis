#ifndef MAINFUNCTIONS_H
#define MAINFUNCTIONS_H

#ifdef IGNORE_OUT_OF_BOUNDS_CASE
#define SPECIAL_ACTION continue
#define SPECIAL_VALUE_BOOLEAN false
#else
#define SPECIAL_ACTION break
#define SPECIAL_VALUE_BOOLEAN true
#endif

#include<stdlib.h>//atoi
#include<string.h>//strcmp
#include<iostream>

#include "quadTree.h"
#include "green.h"
#include "Parser.h"

#define _CRT_SECURE_NO_WARNINGS
#define LINE_BUFFOR_SIZE 40
#define NSAMPLE 200
#define GAUSSIAN_ACCURACY 10
#define BIGGEST_SQUARE_INIT_FACTOR 0.05


void createTree(Tree * mainTree,Layer const& layer);
Rect RandomWalk(Rect R, Tree* mainTree, int& pointCount);
int getIndex(REAL64_t intg[NSAMPLE + 1], double rand);
void printList(std::list<Rect> input);

#endif