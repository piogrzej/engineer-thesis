#include "quadTree.h"
#include "ErrorHandler.h"
#include "mainFunctions.h"

#include <string>

int Tree::nodeCount = 0;

Tree::Tree(int level, Rect const& bounds) 
{
    this->level = level;
    this->bounds = bounds;
    for(ushort i=0; i<NUMBER_OF_NODES; ++i)
        this->nodes[i]=nullptr;
    isSplited = false;
    nodeCount++;
}

void Tree::deleteObjects()
{
    objects.clear();
}

void Tree::addToObjects(Rect const&  r)
{
    objects.push_back(r);
}

Rect Tree::getObjectAtIndex(int index)
{
    int counter=0;
    std::list<Rect>::iterator i;

    for(i=objects.begin(); i != objects.end(); ++i)
    {
        counter++;
        if(counter == index) 
            return *i;
    }
}

//checks if Rect is inside QuadTree bounds
bool Tree::isInBounds(Rect const&  r)
{
    if (r.topLeft.y     > this->bounds.topLeft.y     &&
        r.topLeft.x     > this->bounds.topLeft.x     &&
        r.bottomRight.y < this->bounds.bottomRight.y &&
        r.bottomRight.x < this->bounds.bottomRight.x) 
        return true;
    else 
        return false;
}

bool Tree::isInBounds(point const&  p)
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

void Tree::clear()
{
    if (this != nullptr)
    {
        this->objects.clear();
        for(ushort i=0; i<NUMBER_OF_NODES; ++i){
            this->nodes[i]->clear();
            this->nodes[i] = nullptr;
        }
    }
}

void Tree::split()
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
    this->nodes[0] = new Tree(this->level + 1, ULbound);
    this->nodes[1] = new Tree(this->level + 1, URbound);
    this->nodes[2] = new Tree(this->level + 1, LRbound);
    this->nodes[3] = new Tree(this->level + 1, LLbound);
    isSplited = true;
}

//"wkladanie" elementu na drzewo
bool Tree::insert(Rect const&  r)
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
            for(ushort i=0; i<NUMBER_OF_NODES; ++i)
                if(true==this->nodes[i]->insert(tmp)){
                    counter--;
                    break;
                }
            //jezeli powyzsze niespelnione to jest nachodzi na bisectory line i powinno zostac
            this->addToObjects(tmp);
            counter--;
        }
    }
    if (this->level < MAX_LEVELS && true==isSplited)
    {
        for(ushort i=0 ;i<NUMBER_OF_NODES; ++i)
            if (true==this->nodes[i]->insert(r))
                return true;
        this->addToObjects(r);//jezeli powyzsze nie spelnione to musi byc na bisectory lines, czyli dodajemy do listy rodzica
        return true;//lezy na lini przeciecia wiec dodajemy do listy rodzica (@up), udalo sie dodac wiec true
    }
    return false;//nigdy nie powinno do tego dojsc//jedyne wytlumacznie max level lub obszar o bardzo malym rozmiarze//nie jestem pewien, do sprawdzenia!
}


bool Tree::checkCollisions(Rect const& r, const Rect &ignore)
{
    if (isSplited)
    {
        for(ushort i=0; i<NUMBER_OF_NODES;++i)
        if (this->nodes[i]->bounds.rectsCollision(r))
                return this->nodes[i]->checkCollisions(r, ignore);


    }
    return  this->checkCollisionsWithObjs(r, ignore);
}

//bool QuadTree::checkCollisions(Rect const& r, const Rect &ignore)
//{
//    QuadTree* oldNode, *node = this;
//    QuadTreePtr* stack = new QuadTreePtr[nodeCount + 1];
//    QuadTreePtr* stackPtr = stack;
//    bool goUL, goUR, goLR, goLL;
//    *stackPtr++ = nullptr; // koniec p�tli gdy tu trafimy
//
//    while (node != nullptr)
//    {
//        if (node->isSplited)
//        {
//            goUL = node->UL->bounds.rectsCollision(r);
//            goUR = node->UR->bounds.rectsCollision(r);
//            goLR = node->LR->bounds.rectsCollision(r);
//            goLL = node->LL->bounds.rectsCollision(r);
//        }
//        else
//            goUL = goUR = goLR = goLL = false;
//
//        if (!goUL && !goUR && !goLL && !goLR)
//        {
//            if (node->checkCollisionsWithObjs(r, ignore))
//            {
//                delete stack;
//                return true;
//            }
//            node = *--stackPtr;
//        }
//        else
//        {
//            oldNode = node;
//
//            if (goUL)
//                node = node->UL;         
//            else if (goUR)
//                node = node->UR;
//            else if (goLR)
//                node = node->LR;
//            else if (goLL) 
//                node = node->LL;
//
//            oldNode->addNodesToStack(stackPtr, node, goUL, goUR, goLR, goLL);
//        }    
//    }
//    delete stack;
//    return false;
//}
//
//void QuadTree::addNodesToStack(QuadTreePtr* stackPtr,QuadTree* except,bool isUL, bool isUR, bool isLR, bool isLL)
//{
//    if (isUL && except != UL)
//        *stackPtr++ = UL;
//    if (isUR && except != UR)
//        *stackPtr++ = UR;
//    if (isLR && except != LR)
//        *stackPtr++ = LR;
//    if (isLL && except != LL)
//        *stackPtr++ = LL;
//}

bool Tree::checkCollisionsWithObjs(Rect const&  r, const Rect &ignore)
{
    for (Rect const& i : objects)
        if (i != ignore && i.rectsCollision(r))
            return true;

    return false;
}

bool Tree::checkCollisionObjs(point p, Rect& r)
{
    std::list<Rect>::iterator i;
    for(i=this->objects.begin(); i != this->objects.end(); ++i)
        if(i->rectContains(p))
        {
            r = Rect(i->topLeft,i->bottomRight);
            return true;
        }
    return false;
}

bool Tree::checkCollisons(point p, Rect& r)
{
    Tree* current=this,*next;
    while(true){
        if (true==current->isSplited)
        {
            for(ushort i=0; i<NUMBER_OF_NODES; ++i)
                if (true==current->nodes[i]->bounds.rectContains(p))
                {
                    next = current->nodes[i];
                    break;
                }
        }
        //tutaj dla kazdego sprawdzenie bisectory lines
        if (true==current->checkCollisionObjs(p, r))//KOLIZJA
            return true;
        else if(false==current->isSplited)
            return false;
        else
            current=next;
            
    }
}

Rect Tree::drawBiggestSquareAtPoint(point p)
{
    int dist;
    if (bounds.getHeigth() > bounds.getWidth())
        dist = bounds.getHeigth();
    else
        dist = bounds.getWidth();

    dist *= BIGGEST_SQUARE_INIT_FACTOR;

    bool maxReached=false;
    int MIN_MOVE_DIST=3;
    Rect output(point(p.x-1,p.y-1),point(p.x+1,p.y+1));

    while(dist > MIN_MOVE_DIST || false == maxReached)
    {
        if(true == checkCollisions(output))
        {
            maxReached = true;
            dist/=2;
            output.topLeft.x+=dist;
            output.topLeft.y+=dist;
            output.bottomRight.x-=dist;
            output.bottomRight.y-=dist;
        }
        else
        {
            if(true == maxReached)  
                dist /=2;
            output.topLeft.x-=dist;
            output.topLeft.y-=dist;
            output.bottomRight.x+=dist;
            output.bottomRight.y+=dist;
        }
    }
    return output;//potencjalnie niebezpieczne
}

double Tree::getAdjustedGaussianFactor(Rect const& r, double const factor, FACTOR_TYPE type)
{
    bool isCollision = false;
    bool isDividing = true;
    bool isFirstIt = true;
    double adjustedFactor = factor;
    double leftBound = 1., righBound = factor;

    Rect surface;

    for (int i = 0; i < GAUSSIAN_ACCURACY; i++)
    {
        surface = (type == FACTOR_X) ? 
                r.createGaussianSurfaceX(adjustedFactor) :
                r.createGaussianSurfaceY(adjustedFactor);

        isCollision = (!isInBounds(surface) || checkCollisions(surface, r));

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
    return adjustedFactor;
}


Rect Tree::creatGaussianSurfFrom(Rect const & r, double const factor) // bez kolizji
{
    if (factor < 1)
    {
        ErrorHandler::getInstance() >> "CreateGaussian: Nieprawidlowy wspolczynnik!\n";
        return r;
    }

    double factorX = getAdjustedGaussianFactor(r, factor, FACTOR_X);
    double factorY = getAdjustedGaussianFactor(r, factor, FACTOR_Y);
    ErrorHandler::getInstance() >> factorX >> "," >> factorY >> "\n";

    return r.createGaussianSurface(factorX, factorY);
}

void Tree::printTree(std::string const& name)
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
        for(ushort i=0; i<NUMBER_OF_NODES;++i)
            this->nodes[i]->printTree("UL");
    }
}
