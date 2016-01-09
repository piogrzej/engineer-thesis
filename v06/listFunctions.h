#ifndef LISTFUNCTIONS_H
#define LISTFUNCTIONS_H

#include "struktury.h"

//funkcje listy
int listSize(list *l);
void addToList(list *l, rect r);
void deleteList(list *l);
bool listContains(list *l, rect r);

#endif