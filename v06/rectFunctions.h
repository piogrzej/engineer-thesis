#ifndef RECTFUNCTIONS_H
#define RECTFUNCTIONS_H

#include<ctime>//timer

#include "struktury.h"

//funkcje rect
int getWidth(rect r);
int getHeigth(rect r);
bool cmpRects(rect r1, rect r2); // true -> r1==r2
bool rectsCollision(rect r1, rect r2);//sprawdza kolizje miedzy r1 i r2
bool rectContains(rect r,point p);//sprwadza czy p znajduje sie w r

#endif