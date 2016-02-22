#ifndef MAINFUNCTIONS_H
#define MAINFUNCTIONS_H

#include<stdlib.h>//atoi
#include<string.h>//strcmp
#include<iostream>

#include "quadTree.h"

#define _CRT_SECURE_NO_WARNINGS
#define LINE_BUFFOR_SIZE 40
#define NSAMPLE 200

//funkcje "glowne" przetwarzanie itd
void pointsFormLine(point * topLeft, point * bottomRight, char * line);
void createTree(quadTree * mainTree, FILE * pFile);
Rect layerSpaceSize(FILE * pFile);

#endif