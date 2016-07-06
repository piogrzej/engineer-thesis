#ifndef LISTFUNCTIONS_H
#define LISTFUNCTIONS_H

#include "list.h"

//funkcje listy
int listSize(list *l);
void printAllObjects(list *l);
void addToList(list *l, Rect r);
void deleteList(list *l);
bool listContains(list *l, Rect r);

#endif