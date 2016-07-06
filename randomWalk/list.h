#ifndef STRUKTURY_H
#define STRUKTURY_H

#include<stdio.h>

#include "Rect.h"

struct list{
	list* next = NULL;
	list* prev = NULL;
	Rect value;
	bool isValueSet = false;
};

#endif