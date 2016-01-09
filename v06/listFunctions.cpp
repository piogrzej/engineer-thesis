#include "listFunctions.h"

int listSize(list *l){
	list *tmp = l;
	int i = 0;
	if (tmp != NULL){
		while (tmp->isValueSet){
			i++;
			if (tmp->next == NULL) break;
			else tmp = tmp->next;
		}
	}
	return i;
}

void addToList(list *l, rect r){
	if (l == NULL) l = new list();
	list *tmp = l;
	while (l->next != NULL){
		l = l->next;
	}
	if (l->isValueSet){
		l->next = new list();
		l->next->prev = l;
		l->next->value = r;
		l->next->isValueSet = true;
	}
	else{
		l->isValueSet = true;
		l->value = r;
	}
}

void deleteList(list *l){//usuwanie listy
	list *tmp = l;
	if (tmp->next != NULL){
		tmp = tmp->next;
		while (tmp->next != NULL){
			tmp = tmp->next;
			delete(tmp->prev);
			tmp->prev = NULL;
		}
		delete(tmp);
		tmp = NULL;
	}
}

/*bool listContains(list *l, rect r){
	if (l != NULL && l->isValueSet){
		if (cmpRects(l->value, r)) return true;
		list *tmp = l;
		while (tmp->next != NULL){
			if (cmpRects(tmp->value, r)) return true;
			tmp = tmp->next;
		}
		if (cmpRects(tmp->value, r)) return true;
	}
	else
		return false;
}*/