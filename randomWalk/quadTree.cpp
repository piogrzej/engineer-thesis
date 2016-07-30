#include "quadTree.h"
#include "ErrorHandler.h"
#include "mainFunctions.h"

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

bool QuadTree::isInBounds(point const&  p)
{
    if (p.x > this->bounds.topLeft.x &&
        p.y > this->bounds.topLeft.y &&
        p.x < this->bounds.bottomRight.x &&
        p.y < this->bounds.bottomRight.y
        )
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

bool QuadTree::checkCollisions(Rect const& r, const Rect &ignore)
{
    if (isSplited)
    {
        if (this->UL->bounds.rectsCollision(r))
                return this->UL->checkCollisions(r, ignore);

        if (this->UR->bounds.rectsCollision(r))
                return this->UR->checkCollisions(r, ignore);

        if (this->LR->bounds.rectsCollision(r))
                return this->LR->checkCollisions(r, ignore);

        if (this->LL->bounds.rectsCollision(r))
                return this->LL->checkCollisions(r, ignore);
    }
    return  this->getCollisionObjs(r, ignore);
}

bool QuadTree::getCollisionObjs(Rect const&  r, const Rect &ignore)
{
        for (Rect const& i : objects)
            if (i != ignore && i.rectsCollision(r))
                return true;
    return false;
}

bool QuadTree::checkCollisionObjs(point p, Rect& r)
{
    std::list<Rect>::iterator i;
    for(i=this->objects.begin(); i != this->objects.end(); ++i)
        if(i->rectContains(p)){
            r = Rect(i->topLeft,i->bottomRight);
            return true;
        }
    return false;
}

bool QuadTree::checkCollisons(point p, Rect& r)
{
    bool returned=false;
    if (isSplited)
    {
        if (this->UL->bounds.rectContains(p))
            returned = this->UL->checkCollisons(p,r);
        
        if (this->UR->bounds.rectContains(p))
            returned = this->UR->checkCollisons(p,r);
        
        if (this->LR->bounds.rectContains(p))
            returned = this->LR->checkCollisons(p,r);
        
        if (this->LL->bounds.rectContains(p))
            returned = this->LL->checkCollisons(p,r);
    }
    //tutaj dla kazdego sprawdzenie bisectory lines
    if (this->checkCollisionObjs(p, r))//KOLIZJA
        return true;
    else
        return returned;
}

Rect QuadTree::drawBiggestSquareAtPoint(point p)
{
    Rect output(point(p.x-1,p.y-1),point(p.x+1,p.y+1));
    std::list<Rect> collisions;
    while(true)
    {
        if(true == bounds.rectContains(output))
        {          
            if(checkCollisions(output))
            {
                return output;
            }
            output.topLeft.x--;
            output.topLeft.y--;
            output.bottomRight.x++;
            output.bottomRight.y++;
        }
        else break;
    }
    return output;//potencjalnie niebezpieczne
}

Rect QuadTree::creatGaussianSurfFrom(Rect const & r, double const factor, double& resultFactor) // bez kolizji
{
    if (factor < 1)
    {
        ErrorHandler::getInstance() >> "CreateGaussian: Nieprawidlowy wspolczynnik!\n";
        return r;
    }

    Rect surface;
    bool isCollision = false;
    bool isDividing = true;
    bool isFirstIt = true;
    double adjustedFactor = factor;
    double leftBound = 1., righBound = factor;

    for (int i = 0; i < GAUSSIAN_ACCURACY; i++)
    {
        surface = r.createGaussianSurface(adjustedFactor);
        isCollision = (!isInBounds(surface) || checkCollisions(surface,r));

        if (isFirstIt && !isCollision)
            break;
        if ((isCollision && !isDividing) ||
            !isCollision &&  isDividing)
        {
            isDividing = !isDividing;
        }

        if (isDividing)
            adjustedFactor = righBound = (leftBound + righBound) / 2.;
        else
            adjustedFactor = leftBound = (leftBound + righBound) / 2.;

        isFirstIt = false;
    }

    resultFactor = adjustedFactor;
    return surface;
}

void QuadTree::printTree(std::string const& name)
{
    std::string lvlSpaceNode = "", lvlSpaceRect = "";
    std::list<Rect>::iterator i;

    for (int i = 0; i < level; i++)
    {
            if (i + 1 == level)
                    lvlSpaceNode += "|--";
            else
                    lvlSpaceNode += "|  ";
            lvlSpaceRect += "|  ";
    }
    ErrorHandler::getInstance() << lvlSpaceNode << name << " objects: " << this->objects.size() << "\n";
    lvlSpaceRect += "|---";

    for(i=this->objects.begin(); i != this->objects.end(); ++i)
    {
        ErrorHandler::getInstance() << lvlSpaceRect << *i << "\n";
    }

    if (isSplited)
    {
            UL->printTree("UL");
            UR->printTree("UR");
            LR->printTree("LR");
            LL->printTree("LL");
    }
}