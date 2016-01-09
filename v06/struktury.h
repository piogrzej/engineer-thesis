#ifndef STRUKTURY_H
#define STRUKTURY_H

#include<stdio.h>//printf,File, NULL etc

struct point{
	int x;
	int y;
};

struct rect{
	point top_left;
	point bottom_right;
};

struct list{
	list* next = NULL;
	list* prev = NULL;
	rect value;
	bool isValueSet = false;
};

#endif