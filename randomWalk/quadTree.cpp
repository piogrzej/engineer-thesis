#include "quadTree.h"

#include <string>

QuadTree::QuadTree(int level, Rect const& bounds) {
	this->level = level;
	this->bounds = bounds;
	this->UL = nullptr;
	this->UR = nullptr;
	this->LR = nullptr;
	this->LL = nullptr;
	isSplited = false;
}

void QuadTree::deleteObjects(){
    objects.clear();
}

void QuadTree::addToObjects(Rect const&  r)
{
    objects.push_back(r);
}

Rect QuadTree::getObjectAtIndex(int index)
{
    int counter=0;
    std::list<Rect>::iterator i;
    for(i=objects.begin(); i != objects.end(); ++i){
        counter++;
        if(counter==index) return *i;
    }
}

//checks if Rect is inside QuadTree bounds
bool QuadTree::isInBounds(Rect const&  r)
{
	if (r.topLeft.y     > this->bounds.topLeft.y     &&
		r.topLeft.x     > this->bounds.topLeft.x     &&
		r.bottomRight.y < this->bounds.bottomRight.y &&
		r.bottomRight.x < this->bounds.bottomRight.x) 
		return true;
	else 
		return false;
}

void QuadTree::clear()
{
	if (this != nullptr)
	{
            this->objects.clear();
            this->UL->clear();
            this->UL = nullptr;
            this->UR->clear();
            this->UR = nullptr;
            this->LR->clear();
            this->LR = nullptr;
            this->LL->clear();
            this->LL = nullptr;
	}
}

void QuadTree::split()
{
	int subWidth = this->bounds.topLeft.x + (int)(bounds.getWidth() / 2);
	int subHeigth = this->bounds.topLeft.y + (int)(bounds.getHeigth() / 2);
	Rect ULbound, URbound, LRbound, LLbound;
	//UL
	ULbound.topLeft = this->bounds.topLeft;
	ULbound.bottomRight.x = subWidth;
	ULbound.bottomRight.y = subHeigth;
	//UR
	URbound.bottomRight.x = this->bounds.bottomRight.x;
	URbound.bottomRight.y = subHeigth;
	URbound.topLeft.x = subWidth;
	URbound.topLeft.y = this->bounds.topLeft.y;
	//LR
	LRbound.topLeft.x = subWidth;
	LRbound.topLeft.y = subHeigth;
	LRbound.bottomRight = this->bounds.bottomRight;
	//LL
	LLbound.topLeft.x = this->bounds.topLeft.x;
	LLbound.topLeft.y = subHeigth;
	LLbound.bottomRight.x = subWidth;
	LLbound.bottomRight.y = this->bounds.bottomRight.y;
	//"SPLIT"
	this->UL = new QuadTree(this->level + 1, ULbound);
	this->UR = new QuadTree(this->level + 1, URbound);
	this->LR = new QuadTree(this->level + 1, LRbound);
	this->LL = new QuadTree(this->level + 1, LLbound);
	isSplited = true;
}

//"wkladanie" elementu na drzewo
bool QuadTree::insert(Rect const&  r)
{
	int counter = 0;
	Rect tmp;

	if (!this->isInBounds(r)) 
		return false;

	if (this->objects.size() < MAX_OBJECTS && false == isSplited) 
	{
		this->addToObjects(r);
		return true;
	}
	else if (this->objects.size() >= MAX_OBJECTS && false == isSplited)
	{
            //DZIELIMY I obiekty z listy wrzucamy do odpowiednich kwadratow
            this->split();//podzial
            counter = this->objects.size();
            for(int i=0;i<counter;++i)
            {
                tmp = objects.front();
                this->objects.pop_front();
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
                this->addToObjects(tmp);
                counter--;
            }
	}
	if (this->level < MAX_LEVELS && isSplited)
	{
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

/*QuadTree QuadTree::findRect(Rect r){//NIEPRZETESTOWANE!
	if(listContains(&(this->objects), r)) return *this;//sprawdzamy czy Rect r zawiera sie w liscie obiektow
	if (this->UL != NULL){
		if (this->UL->isInBounds(r))
			return this->UL->findRect(r);
		if (this->UR->isInBounds(r))
			return this->UR->findRect(r);
		if (this->LR->isInBounds(r))
			return this->LR->findRect(r);
		if (this->LL->isInBounds(r))
			return this->LR->findRect(r);
	}
	else{
		assert(0);
		return *this;
	}
}*/

void QuadTree::retrieve(std::list<Rect> *returnedRects, Rect const& r)
{
    if (isSplited)
    {
        if (this->UL->bounds.rectsCollision(r))
                this->UL->retrieve(returnedRects, r);

        if (this->UR->bounds.rectsCollision(r))
                this->UR->retrieve(returnedRects, r);		

        if (this->LR->bounds.rectsCollision(r))
                this->LR->retrieve(returnedRects, r);

        if (this->LL->bounds.rectsCollision(r))
                this->LL->retrieve(returnedRects, r);
    }
    //tutaj dla kazdego sprawdzenie bisectory lines
    this->getCollisionObjs(returnedRects, r);
}


void QuadTree::getCollisionObjs(std::list<Rect> *returnedRects, Rect const&  r){
    std::list<Rect>::iterator i;
    for(i=this->objects.begin(); i != this->objects.end(); ++i)
        if(r.rectsCollision(*i)){
            returnedRects->push_front(*i);
        }
}

bool QuadTree::checkCollisionObjs(point p, Rect* r)
{
    std::list<Rect>::iterator i;
    for(i=this->objects.begin(); i != this->objects.end(); ++i)
        if(i->rectContains(p)){
            r = new Rect(*i);
            return true;
        }
    return false;
}

bool QuadTree::checkCollisons(point p, Rect* r)
{
	if (isSplited)
	{
		if (this->UL->bounds.rectContains(p))
			this->UL->checkCollisons(p,r);
		
		if (this->UR->bounds.rectContains(p))
			this->UR->checkCollisons(p,r);
		
		if (this->LR->bounds.rectContains(p))
			this->LR->checkCollisons(p,r);
		
		if (this->LL->bounds.rectContains(p))
			this->LL->checkCollisons(p,r);
	}
	//tutaj dla kazdego sprawdzenie bisectory lines
	if (this->checkCollisionObjs(p, r))//KOLIZJA
		return true;
	else if (false == isSplited) 
		return false;//DOSZEDLEM DO KONCA BRAK KOLIZJI
}

Rect QuadTree::drawBiggestSquareAtPoint(point p){
    Rect output(point(p.x-1,p.y-1),point(p.x+1,p.y+1));
    std::list<Rect> collisions;
    while(true){
        if(false==this->bounds.rectContains(output)){
            this->retrieve(&collisions,output);
            if(false==collisions.empty())
                return output;
            output.topLeft.x--;
            output.topLeft.y--;
            output.bottomRight.x++;
            output.bottomRight.y++;
        }
        return output;//potencjalnie niebezpieczne
    }
}

//Rect QuadTree::drawBiggestSquareAtPoint(point p)
//{
//    //pierwwszy obieg
//    Rect r;//tylko jako argument w funkcji, nie ptorzebne do ncizego
//    unsigned int left=p.x-1, right=p.y+1, top=p.y-1, bottom = p.y+1;
//    point tmp;
//    bool continueFlag = true;
//    bool leftStopFlag = false;
//    bool rightStopFlag = false;
//    bool topStopFlag = false;
//    bool bottomStopFlag = false;
//	
//    while ( !bottomStopFlag && 
//                !topStopFlag    && 
//                !rightStopFlag  && 
//                !bottomStopFlag)
//    {
//        if (!topStopFlag) 
//                for (int i = 0; i<right-left; ++i)
//                {
//                        tmp.x = left + i;
//                        tmp.y = top;
//                        topStopFlag = checkCollisons(tmp,r);
//                }
//
//        if (!bottomStopFlag) 
//                for (int i = 0; i < right - left; ++i)
//                {
//                        tmp.x = left + i;
//                        tmp.y = top;
//                        bottomStopFlag = checkCollisons(tmp,r);
//                }
//
//        if (!rightStopFlag) 
//                for (int i = 0; i < bottom - top; i++)
//                {
//                        tmp.x = right;
//                        tmp.y = top + i;
//                        rightStopFlag = checkCollisons(tmp,r);
//                }
//
//        if (!leftStopFlag) 
//                for (int i = 0; i < bottom - top; i++)
//                {
//                        tmp.x = left;
//                        tmp.y = top + i;
//                        leftStopFlag = checkCollisons(tmp,r);
//                }
//
//        //ustalanie nowych wsp
//        if (p.x-1 > this->bounds.topLeft.x) 
//                left = p.x - 1;
//        else 
//                leftStopFlag = true;
//
//        if (p.x + 1 < this->bounds.bottomRight.x) 
//                right = p.x + 1;
//        else 
//                rightStopFlag = true;
//
//        if (p.y + 1 > this->bounds.topLeft.y) 
//                top = p.y - 1;
//        else 
//                topStopFlag = true;
//
//        if (p.y + 1 < this->bounds.bottomRight.y)
//                bottom = p.y + 1;
//        else 
//                bottomStopFlag = true;
//    }
//
//Rect ret;
//ret.topLeft.y = top;
//ret.topLeft.x = left;
//ret.bottomRight.y = bottom;
//ret.bottomRight.x = right;
//
//return ret;
//}

void QuadTree::printTree(std::string const& name)
{
    std::string lvlSpaceNode = "", lvlSpaceRect = "";
    std::list<Rect>::iterator i;
    //list* iter = &objects;
    for (int i = 0; i < level; i++)
    {
            if (i + 1 == level)
                    lvlSpaceNode += "|--";
            else
                    lvlSpaceNode += "|  ";
            lvlSpaceRect += "|  ";
    }
    std::cout << lvlSpaceNode << name << " objects: " << this->objects.size() << std::endl;
    lvlSpaceRect += "|---";
    for(i=this->objects.begin(); i != this->objects.end(); ++i){
        std::cout << lvlSpaceRect << *i;
    }

    if (isSplited)
    {
            UL->printTree("UL");
            UR->printTree("UR");
            LR->printTree("LR");
            LL->printTree("LL");
    }
}

void QuadTree::debugFunction()
{
	if (isSplited)
	{
		this->UL->debugFunction();
		this->UR->debugFunction();
		this->LR->debugFunction();
		this->LL->debugFunction();
	}
	debug += this->objects.size();
}