#include "quadTree.h"

void quadTree::deleteObjects(){
	list *tmp = &(this->objects);
	tmp->isValueSet = false;
	if (tmp->next != NULL){
		tmp = tmp->next;
		tmp->prev->next = NULL;
		while (tmp->next != NULL){
			tmp = tmp->next;
			delete(tmp->prev);
		}
		delete(tmp);
	}
}

void quadTree::addToObjects(rect r){
	list *l = &(this->objects);
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

rect quadTree::getObjectAtIndex(int index){
	list *tmp = &(this->objects);
	int i = 0;
	while (i<index/*&& tmp->next!=null*/){//takie lekkie zabezpieczenie->wykomentowane, jak sie tu spynie to cos nie tak
		tmp = tmp->next;
		i++;
	}
	return tmp->value;
}

//nie jest zalecane uzywanie tej funkcji z argumentem 0 jezeli lista ma wiecej niz 1 element
rect quadTree::removeAndReturnObjectAtIndex(int index){
	rect r;
	list *tmp = &(this->objects);
	int i = 0;
	while (i<index/*&& tmp->next!=null*/){//takie lekkie zabezpieczenie->wykomentowane, jak sie tu spynie to cos nie tak
		tmp = tmp->next;
		i++;
	}
	r = tmp->value;
	if (i != 0){
		tmp->prev->next = tmp->next;
		if (tmp->next != NULL)tmp->next->prev = tmp->prev;//jezeli nei jest ostatni elementem
		delete(tmp);
		return r;
	}
	else{
		if (this->objects.next != NULL) this->objects.next->prev = NULL;
		this->objects.isValueSet = false;
		return r;
	}
}

int quadTree::getObjectSize(){
	list *tmp = &(this->objects);
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

//checks if rect is inside quadTree bounds
bool quadTree::contains(rect r){
	if (r.top_left.y > this->bounds.top_left.y &&
		r.top_left.x > this->bounds.top_left.x &&
		r.bottom_right.y < this->bounds.bottom_right.y &&
		r.bottom_right.x < this->bounds.bottom_right.x) return true;
	else return false;
}

quadTree::quadTree(int level, rect bounds){
	this->level = level;
	this->bounds = bounds;
	this->UL = NULL;
	this->UR = NULL;
	this->LR = NULL;
	this->LL = NULL;
	this->objects.isValueSet = false;
}


void quadTree::clear(){//czyszczenie struktury
	if (this != NULL){
		if (this->objects.isValueSet != false) this->deleteObjects();
		this->UL->clear();
		this->UL = NULL;
		this->UR->clear();
		this->UR = NULL;
		this->LR->clear();
		this->LR = NULL;
		this->LL->clear();
		this->LL = NULL;
	}
}

void quadTree::split(){
	int subWidth = this->bounds.top_left.x + (int)(getWidth(this->bounds) / 2);
	int subHeigth = this->bounds.top_left.y + (int)(getHeigth(this->bounds) / 2);
	rect ULbound, URbound, LRbound, LLbound;
	//UL
	ULbound.top_left = this->bounds.top_left;
	ULbound.bottom_right.x = subWidth;
	ULbound.bottom_right.y = subHeigth;
	//UR
	URbound.bottom_right.x = this->bounds.bottom_right.x;
	URbound.bottom_right.y = subHeigth;
	URbound.top_left.x = subWidth;
	URbound.top_left.y = this->bounds.top_left.y;
	//LR
	LRbound.top_left.x = subWidth;
	LRbound.top_left.y = subHeigth;
	LRbound.bottom_right = this->bounds.bottom_right;
	//LL
	LLbound.top_left.x = this->bounds.top_left.x;
	LLbound.top_left.y = subHeigth;
	LLbound.bottom_right.x = subWidth;
	LLbound.bottom_right.y = this->bounds.bottom_right.y;
	//"SPLIT"
	this->UL = new quadTree(this->level + 1, ULbound);
	this->UR = new quadTree(this->level + 1, URbound);
	this->LR = new quadTree(this->level + 1, LRbound);
	this->LL = new quadTree(this->level + 1, LLbound);
}

//"wkladanie" elementu na drzewo
bool quadTree::insert(rect r){
	int counter = 0;
	rect tmp;
	if (!this->contains(r)) return false;
	if (this->getObjectSize() < MAX_OBJECTS && this->UL == NULL){//JEZELI W LISCIE OBIEKTOW JEST JESZCZE MIEJSCE I NIE BYLO PODZIALU
		this->addToObjects(r);
		return true;
	}
	else if (this->getObjectSize() >= MAX_OBJECTS && this->UL == NULL){//JEZELI W LISCIE OBIEKTOW NIE MA MIEJSCE I NIE BYLO PODZIALU
		//DZIELIMY I obiekty z listy wrzucamy do odpowiednich kwadratow
		this->split();//podzial
		counter = this->getObjectSize();
		while (this->getObjectSize() && counter){//powtarzamy dopoki jest cos w liscie//wykomentowane oki, cel->debug
			tmp = this->removeAndReturnObjectAtIndex(counter - 1);
			if (this->UL->insert(tmp)){
				counter--;
				continue;
			}
			if (this->UR->insert(tmp)){
				counter--;
				continue;
			}
			if (this->LR->insert(tmp)){
				counter--;
				continue;
			}
			if (this->LL->insert(tmp)){
				counter--;
				continue;
			}
			//jezeli powyzsze niespelnione to jest nachodzi na bisectory line i powinno zostac
			//debug++;
			this->addToObjects(tmp);
			counter--;
		}
	}
	if (/*this->level < MAX_LEVELS && this->UL!=NULL*/this->level < 100 && this->UL != NULL){//wykomentowane jest oki, tu test//JEZELI ODPOWIEDNI poziom i byl split

		if (this->UL->insert(r))
			return true;
		if (this->UR->insert(r))
			return true;
		if (this->LR->insert(r))
			return true;
		if (this->LL->insert(r))
			return true;
		this->addToObjects(r);//jezeli powyzsze nie spelnione to musi byc na bisectory lines, czyli dodajemy do listy rodzica
		return true;//lezy na lini przeciecia wiec dodajemy do listy rodzica (@up), udalo sie dodac wiec true
	}
	return false;//nigdy nie powinno do tego dojsc//jedyne wytlumacznie max level lub obszar o bardzo malym rozmiarze//nie jestem pewien, do sprawdzenia!
}

/*quadTree quadTree::findRect(rect r){//NIEPRZETESTOWANE!
	if(listContains(&(this->objects), r)) return *this;//sprawdzamy czy rect r zawiera sie w liscie obiektow
	if (this->UL != NULL){
		if (this->UL->contains(r))
			return this->UL->findRect(r);
		if (this->UR->contains(r))
			return this->UR->findRect(r);
		if (this->LR->contains(r))
			return this->LR->findRect(r);
		if (this->LL->contains(r))
			return this->LR->findRect(r);
	}
	else{
		assert(0);
		return *this;
	}
}*/

void quadTree::retrieve(list *returnedRects, rect r){
	if (this->UL != NULL){
		//TE ELSE IF SA NIEKONIECZNE
		if (this->UL->contains(r)){
			this->UL->retrieve(returnedRects, r);
		}
		/*else if (rectsCollision(this->UL->bounds, r)){
			//TODO sprwadz wszytskie pozostale
		}*/
		if (this->UR->contains(r)){
			this->UR->retrieve(returnedRects, r);
		}
		/*else if (rectsCollision(this->UR->bounds, r)){
			//TODO sprwadz wszytskie pozostale
		}*/
		if (this->LR->contains(r)){
			this->LR->retrieve(returnedRects, r);
		}
		/*else if (rectsCollision(this->LR->bounds, r)){
			//TODO sprwadz wszytskie pozostale
		}*/
		if (this->LL->contains(r)){
			this->LL->retrieve(returnedRects, r);
		}
		/*else if (rectsCollision(this->LL->bounds, r)){
			//TODO sprwadz wszytskie pozostale
		}*/
	}
	//tutaj dla kazdego sprawdzenie bisectory lines
	this->getCollisionObjs(returnedRects, r);

}


void quadTree::getCollisionObjs(list *returnedRects, rect r){
	if (this->objects.isValueSet){
		if (rectsCollision(this->objects.value, r)){
			addToList(returnedRects, this->objects.value);
		}
		if (this->objects.next != NULL){
			if (this->objects.next->isValueSet){
				list * tmp = this->objects.next;
				if (rectsCollision(tmp->value, r)){
					addToList(returnedRects, tmp->value);
				}
				while (tmp->next != NULL){
					if (tmp->next->isValueSet){
						tmp = tmp->next;
						if (rectsCollision(tmp->value, r)){
							addToList(returnedRects, tmp->value);
						}
					}
				}
			}
		}
	}
}

bool quadTree::checkCollisionObjs(point p){
	if (this->objects.isValueSet){
		if (rectContains(this->objects.value, p)){
			return true;//KOLIZJA!
		}
		if (this->objects.next != NULL){
			if (this->objects.next->isValueSet){
				list * tmp = this->objects.next;
				if (rectContains(tmp->value, p)){
					return true;//KOLIZJA!
				}
				while (tmp->next != NULL){
					if (tmp->next->isValueSet){
						tmp = tmp->next;
						if (rectContains(tmp->value, p)){
							return true;//KOLIZJA!
						}
					}
				}
			}
		}
	}
}

bool quadTree::checkCollisons(point p){
	if (this->UL != NULL){
		//TE ELSE IF SA NIEKONIECZNE
		if (rectContains(this->UL->bounds,p)){
			this->UL->checkCollisons(p);
		}
		/*else if (rectsCollision(this->UL->bounds, r)){
		//TODO sprwadz wszytskie pozostale
		}*/
		if (rectContains(this->UR->bounds, p)){
			this->UR->checkCollisons(p);
		}
		/*else if (rectsCollision(this->UR->bounds, r)){
		//TODO sprwadz wszytskie pozostale
		}*/
		if (rectContains(this->LR->bounds, p)){
			this->LR->checkCollisons(p);
		}
		/*else if (rectsCollision(this->LR->bounds, r)){
		//TODO sprwadz wszytskie pozostale
		}*/
		if (rectContains(this->LL->bounds, p)){
			this->LL->checkCollisons(p);
		}
		/*else if (rectsCollision(this->LL->bounds, r)){
		//TODO sprwadz wszytskie pozostale
		}*/
	}
	//tutaj dla kazdego sprawdzenie bisectory lines
	if (this->checkCollisionObjs(p)) return true;//KOLIZJA
	else if (this->UL == NULL) return false;//DOSZEDLEM DO KONCA BRAK KOLIZJI
}

rect quadTree::drawBiggestRectAtPoint(point p){
	unsigned int left=p.x-1, right=p.y+1, top=p.y-1, bottom = p.y+1;
	point tmp;
	bool continueFlag = true;
	//pierwszy obieg na sztywno, ustalamy czy w ogole jest jakies mijsce
	tmp.x = left;
	for (int i = 0; i < 3; ++i){
		tmp.y = top;
		if (checkCollisons(tmp)){
			continueFlag = false;
			break;
		}

		tmp.y = bottom;
		if (checkCollisons(tmp)){
			continueFlag = false;
			break;
		}
		tmp.x++;
	}
	if (!continueFlag){
		//TODO
	}
	//TODO
}

//do usuniecia pozniej
void quadTree::debugFunction(){
	/*quadTree *tmp = this;
	while (tmp->UL != NULL){
		printf("%d\n",listSize(&(tmp->objects)));
		tmp = tmp->UL;
	}*/
	if (this->UL != NULL){
		this->UL->debugFunction();
		this->UR->debugFunction();
		this->LR->debugFunction();
		this->LL->debugFunction();
	}
	debug += listSize(&(this->objects));
}