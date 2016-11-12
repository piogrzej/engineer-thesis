#include "d_quadtree.h"

__device__ bool d_QuadTree::isInBounds(point2 const&  p)
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

__device__ bool d_QuadTree::isInBounds(d_Rect const&  r)
{
    if (r.topLeft.y     >= this->bounds.topLeft.y     &&
        r.topLeft.x     >= this->bounds.topLeft.x     &&
        r.bottomRight.y <= this->bounds.bottomRight.y &&
        r.bottomRight.x <= this->bounds.bottomRight.x)
        return true;
    else
        return false;
}

/// create tree,.... d_QuadTree root; d_QuadTree* d_nodes; RectCuda* d_rects;
//for(int i = root.startRectOff(); i < root.endRectOff(); ++i)
//  {
//d_rects[i].topLeft.x;
//// .........
//  }
//for(int i  = 0;i < params.QUAD_TREE_CHILD_NUM; ++i)
//  {
// int childIdx = root[i];
// d_nodes[childIdx].startRectOff();
// // more ...
//  }

__device__ bool d_QuadTree::checkCollisons(point2 p, d_Rect& r)
{
	d_QuadTree* nodes = treeManager->nodes;
    d_QuadTree* current = this, *next = nullptr;

    while(true)
    {
        if (current->isSplited())
        {
        	//printf("s: %d o: %d e: %d \n",current->startRectOff(),current->ownRectOff(),current->endRectOff());
            for(int i=0; i < NODES_NUMBER; ++i)
            {
            	//printf("ch %d    %d\n",i,current->getChlidren(i));
                d_QuadTree* node = &nodes[current->getChlidren(i)];
                if (node->bounds.contains(p))
                {
                    next = node;
                    break;
                }
            }
        }
        //tutaj dla kazdego sprawdzenie bisectory lines
        if (current->checkCollisionObjs(p, r))//KOLIZJA
            return true;
        else if(false == current->isSplited() || next == nullptr)
            return false;
        else
            current=next;
    }
}

__device__ bool d_QuadTree::checkCollisionObjs(point2 p, d_Rect &r)
{
	d_Rect* rects = treeManager->rects;
    for(int i = startOwnOff; i < endOff; ++i)
    {
        if(true == rects[i].contains(p))
        {
            r = d_Rect(rects[i].topLeft,
                       rects[i].bottomRight);
            return true;
        }
    }
    return false;
}

__device__ d_Rect d_QuadTree::drawBiggestSquareAtPoint(point2 p)
{
    bool isCollision = false;
    bool maxReached = false;
    const floatingPoint MIN_DIST = .1f;
    floatingPoint dist;

    d_Rect output(p.x -1,p.y -1, p.x +1,p.y +1);
    d_Rect init(p.x -2,p.y -2, p.x +2,p.y +2);

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
            output.topLeft.x += dist;
            output.topLeft.y += dist;
            output.bottomRight.x -= dist;
            output.bottomRight.y -= dist;
        }
        else
        {
            if(maxReached)
                dist /= 2;
            output.topLeft.x -= dist;
            output.topLeft.y -= dist;
            output.bottomRight.x += dist;
            output.bottomRight.y += dist;
        }
    }
     if (p.x == output.bottomRight.x)
         printf("drawBiggestSquareAtPoint: error\n");
    return output;
}

__device__ bool d_QuadTree::checkCollisions(d_Rect const& r, const d_Rect &ignore)//FUNKCJA DO PRZEPISANIA OD NOWA
{
    if (false == isInBounds(r))
        return true;

    d_QuadTree*	nodes = treeManager->nodes;
    d_QuadTree* oldNode, *node = this;
    dTreePtr* stack = new dTreePtr[treeManager->nodesCount +1];
    dTreePtr* stackPtr = stack;
    bool collisions[NODES_NUMBER];
    *stackPtr++ = nullptr; // koniec petli gdy tu trafimy
    //printf("Col: %f %f %f %f\n",r.topLeft.x,r.topLeft.y,r.bottomRight.x,r.bottomRight.y);

    while (node != nullptr)
    {

        if (node->isSplited())
        {
          //  printf("Nod: %d      ch %d %d %d %d\n",node->getId(),node->chlildren[0],node->chlildren[1],node->chlildren[2],node->chlildren[3]);
#pragma unroll
            for (int i = 0; i < NODES_NUMBER; ++i)
            {
                collisions[i] = nodes[node->getChlidren(i)].getBounds().rectsCollision(r);//czy istnieje nodes[node->getChlidren(i)]?
            }
        }
        else
        {
           // printf("Nod: %d   \n",node->getId());
#pragma unroll
            for (int i = 0; i < NODES_NUMBER; ++i)
                collisions[i] = false;
        }

        if (false == checkIsAnyCollision(collisions))
        {
            if (node->checkCollisionsWithObjs(r, ignore))
            {
                delete stack;
                //printf("jest kolizja\n");
                return true;
            }
            node = *--stackPtr;
        }
        else
        {
            oldNode = node;
            for (int i = 0; i < NODES_NUMBER; ++i)
            {
                if (collisions[i])
                {
                    node = &(nodes[node->getChlidren(i)]);
                    break;
                }
            }

           oldNode->addNodesToStack(stackPtr, node, collisions);
          /*  for (int i = 0; i < NODES_NUMBER; ++i)
            {
                if (collisions[i] && node != &(nodes[oldNode->getChlidren(i)]))
                    *stackPtr++ = &(nodes[oldNode->getChlidren(i)]);
            }*/
        }
    }

    delete stack;
    return false;
}

__device__ bool d_QuadTree::checkIsAnyCollision(bool collisions[])//FUNKCJA DO PRZEPISANIA OD NOWA
{
#pragma unroll
    for (int i = 0; i < NODES_NUMBER; ++i)
    {
        if (collisions[i])
            return true;
    }
    return false;
}

__device__ void d_QuadTree::addNodesToStack(dTreePtr* stackPtr,d_QuadTree* except, bool collisions[])//FUNKCJA DO PRZEPISANIA OD NOWA
{
    d_QuadTree* nodes = treeManager->nodes;
 //   printf("s %d o %d e %d\n",startOff,startOwnOff,endOff);
//#pragma unroll
    for (int i = 0; i < NODES_NUMBER; ++i)
    {
    //    printf("ch %d     %d",i,chlildren[i]);
    //	int child = chlildren[i];
   // 	child += 1;
        if (collisions[i] && except != &(nodes[chlildren[i]]))
            *stackPtr++ = &(nodes[chlildren[i]]);
  //      printf("				TTTTTTTTT\n");

    }
}

__device__ bool d_QuadTree::checkCollisionsWithObjs(d_Rect const& r, const d_Rect &ignore)
{
    d_Rect* rects = treeManager->rects;
    for (int i = startOwnOff; i< endOff; ++i)
            if (rects[i] != ignore && rects[i].rectsCollision(r))
                return true;

        return false;
}

__device__ d_Rect d_QuadTree::createGaussianSurfFrom(d_Rect const & r, floatingPoint const factor) // bez kolizji
{
    if (factor < 1)
    {
        //ErrorLogger::getInstance() >> "CreateGaussian: Nieprawidlowy wspolczynnik!\n";
        printf("CreateGaussian: Nieprawidlowy wspolczynnik!\n");
        return r;
    }

    floatingPoint factorX = getAdjustedGaussianFactor(r, factor, D_FACTOR_X);
    floatingPoint factorY = getAdjustedGaussianFactor(r, factor, D_FACTOR_Y);

    return r.createGaussianSurface(factorX, factorY);
}

__device__ floatingPoint d_QuadTree::getAdjustedGaussianFactor(d_Rect const& r, floatingPoint const factor, D_FACTOR_TYPE type)
{
    bool isCollision = false;
    bool isDividing = true;
    bool isFirstIt = true;
    floatingPoint adjustedFactor = factor;
    floatingPoint leftBound = 1., righBound = factor;

    d_Rect surface;

    for (int i = 0; i < GAUSSIAN_ACCURACY; i++)
    {
        surface = (type == D_FACTOR_X) ?
                r.createGaussianSurfaceX(adjustedFactor) :
                r.createGaussianSurfaceY(adjustedFactor);

        isCollision = (!isInBounds(surface) || checkCollisions(surface, r));

        if (isFirstIt && !isCollision)
            break;

        if ((isCollision && !isDividing) ||
            (!isCollision &&  isDividing))
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
