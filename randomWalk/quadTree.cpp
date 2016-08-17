#include "quadTree.h"
#include "Logger.h"
#include "mainFunctions.h"

#include <string>

using namespace std;

Tree::Tree(int level, int nodeSCount, Rect const& bounds) :
level(level), 
bounds(bounds), 
nodeCount(nodeSCount),
isSplited(false)
{
    for(ushort i=0; i<NUMBER_OF_NODES; ++i)
        this->nodes[i] = nullptr;
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
    if (r.topLeft.y     >= this->bounds.topLeft.y     &&
        r.topLeft.x     >= this->bounds.topLeft.x     &&
        r.bottomRight.y <= this->bounds.bottomRight.y &&
        r.bottomRight.x <= this->bounds.bottomRight.x) 
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
    this->nodes[0] = new Tree(this->level + 1, nodeCount, ULbound);
    this->nodes[1] = new Tree(this->level + 1, nodeCount, URbound);
    this->nodes[2] = new Tree(this->level + 1, nodeCount, LRbound);
    this->nodes[3] = new Tree(this->level + 1, nodeCount, LLbound);
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
    if (false == isInBounds(r))
        return true;

    Tree* oldNode, *node = this;
    TreePtr* stack = new TreePtr[nodeCount + 1];
    TreePtr* stackPtr = stack;
    bool collisions[NUMBER_OF_NODES];
    *stackPtr++ = nullptr; // koniec petli gdy tu trafimy

    while (node != nullptr)
    {
        if (node->isSplited)
        {
            for (int i = 0; i < NUMBER_OF_NODES; i++)
                collisions[i] = node->nodes[i]->bounds.rectsCollision(r);
        }
        else
            for (int i = 0; i < NUMBER_OF_NODES; i++)
                collisions[i] = false;

        if (false == checkIsAnyCollision(collisions))
        {
            if (node->checkCollisionsWithObjs(r, ignore))
            {
                delete stack;
                return true;
            }
            node = *--stackPtr;
        }
        else
        {
            oldNode = node;

            for (int i = 0; i < NUMBER_OF_NODES; i++)
            {
                if (collisions[i])
                {
                    node = node->nodes[i];
                    break;
                }
            }

            oldNode->addNodesToStack(stackPtr, node, collisions);
        }
    }
    delete stack;
    return false;
}

bool Tree::checkIsAnyCollision(bool collisions[])
{
    for (int i = 0; i < NUMBER_OF_NODES; i++)
    {
        if (collisions[i])
            return true;
    }
    return false;
}

void Tree::addNodesToStack(TreePtr* stackPtr,Tree* except, bool collisions[])
{
    for (int i = 0; i < NUMBER_OF_NODES; i++)
    {
        if (collisions[i] && except != nodes[i])
            *stackPtr++ = nodes[i];
    }
}

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
    bool isCollision = false;
    bool maxReached = false;
    const floatingPoint MIN_DIST = .1;
    floatingPoint dist;

    Rect output(p - 1, p + 1);
    Rect init(p - 2, p + 2);

    if (bounds.getHeigth() > bounds.getWidth())
        dist = bounds.getHeigth();
    else
        dist = bounds.getWidth();

    dist *= BIGGEST_SQUARE_INIT_FACTOR;

    if (checkCollisions(output))
        return output;

    if (checkCollisions(init))
        return init;

    while((isCollision = checkCollisions(output)) || false == maxReached || dist >  MIN_DIST)
    {
        if(isCollision)
        {
            if(dist > 1)
                dist /= 2;
            maxReached = true;
            output.topLeft += dist;
            output.bottomRight -= dist;
        }
        else
        {
            if(maxReached)  
                dist /= 2;
            output.topLeft -= dist;
            output.bottomRight += dist;
        }
    }
     if (p.x == output.bottomRight.x)
         cout << "erroe" << endl;
    return output;
}


floatingPoint Tree::getAdjustedGaussianFactor(Rect const& r, floatingPoint const factor, FACTOR_TYPE type)
{
    bool isCollision = false;
    bool isDividing = true;
    bool isFirstIt = true;
    floatingPoint adjustedFactor = factor;
    floatingPoint leftBound = 1., righBound = factor;

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


Rect Tree::creatGaussianSurfFrom(Rect const & r, floatingPoint const factor) // bez kolizji
{
    if (factor < 1)
    {
        ErrorLogger::getInstance() >> "CreateGaussian: Nieprawidlowy wspolczynnik!\n";
        return r;
    }

    floatingPoint factorX = getAdjustedGaussianFactor(r, factor, FACTOR_X);
    floatingPoint factorY = getAdjustedGaussianFactor(r, factor, FACTOR_Y);

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
    ErrorLogger::getInstance() << lvlSpaceNode << name << " objects: " << this->objects.size() << "\n";
    lvlSpaceRect += "|---";

    for(i=this->objects.begin(); i != this->objects.end(); ++i)
    {
        ErrorLogger::getInstance() << lvlSpaceRect << *i << "\n";
    }

    if (isSplited)
    {
        for(ushort i=0; i < NUMBER_OF_NODES;++i)
            this->nodes[i]->printTree("Node");
    }
}
