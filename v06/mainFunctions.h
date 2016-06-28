#ifndef MAINFUNCTIONS_H
#define MAINFUNCTIONS_H

#include<stdlib.h>//atoi
#include<string.h>//strcmp
#include<iostream>

#include "quadTree.h"
#include "green.h"

#define _CRT_SECURE_NO_WARNINGS
#define LINE_BUFFOR_SIZE 40
#define NSAMPLE 200

//funkcje "glowne" przetwarzanie itd
void pointsFormLine(point * topLeft, point * bottomRight, char * line);
void createTree(QuadTree * mainTree, FILE * pFile);
Rect layerSpaceSize(FILE * pFile);
Rect RandomWalk(Rect R, QuadTree* mainTree);
int getIndex(REAL64_t intg[NSAMPLE + 1], double rand);
void debugFunction();

#endif