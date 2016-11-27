#include "d_quadtree.h"
#include <device_functions.h>


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
            for(int i=0; i < NODES_NUMBER; ++i)
            {
            	int id = current->getChildren(i);
                d_QuadTree* node = &nodes[id];
                if (node->bounds.contains(p))
                {
                    next = node;
                    break;
                }
            }
            if(next == current)
            {
            	printf("b: lvl: %d %flf %lf %lf %lf\n",current->getLevel(),current->bounds.topLeft.x,current->bounds.topLeft.y,
            								   current->bounds.bottomRight.x,current->bounds.bottomRight.y);
            	  for(int i=0; i < NODES_NUMBER; ++i)
					{
							  int id = current->getChildren(i);
							  d_QuadTree* node = &nodes[id];
							printf("ch: lvl: %d %flf %lf %lf %lf\n",node->getLevel(),node->bounds.topLeft.x,node->bounds.topLeft.y,
									node->bounds.bottomRight.x,node->bounds.bottomRight.y);
					}
            	  return false;
            }
        }
        //tutaj dla kazdego sprawdzenie bisectory lines
        if (current->checkCollisionObjs(p, r))//KOLIZJA
            return true;
        else if(false == current->isSplited() || next == nullptr)
            return false;
        else
            current = next;
    }
}

__device__ bool d_QuadTree::checkCollisionObjs(point2 p, d_Rect &r)
{
	d_Rect* rects = treeManager->rects;
    for(int i = startOwnOff; i < endOff; ++i)
    {
        if(rects[i].contains(p))
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
            if(dist > MIN_DIST)
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

__device__ bool d_QuadTree::checkCollisions(d_Rect const& r, const d_Rect &ignore)
{
    if (false == isInBounds(r))
        return true;

    d_QuadTree*	nodes = treeManager->nodes;
    d_QuadTree* oldNode, *node = this;
    dTreePtr* stackPtr = stack[threadIdx.x];
    bool collisions[NODES_NUMBER];
    *stackPtr++ = nullptr; // koniec petli gdy tu trafimy
    //printf("Col: %f %f %f %f\n",r.topLeft.x,r.topLeft.y,r.bottomRight.x,r.bottomRight.y);

    while (node != nullptr)
    {
        if (node->checkCollisionsWithObjs(r, ignore))
            return true;

        if (node->isSplited())
        {
          //  printf("Nod: %d      ch %d %d %d %d\n",node->getId(),node->chlildren[0],node->chlildren[1],node->chlildren[2],node->chlildren[3]);
#pragma unroll
            for (int i = 0; i < NODES_NUMBER; ++i)
            {
                collisions[i] = nodes[node->getChildren(i)].getBounds().rectsCollision(r);//czy istnieje nodes[node->getChlidren(i)]?
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NODES_NUMBER; ++i)
                collisions[i] = false;
        }

        if (false == checkIsAnyCollision(collisions))
        {
            node = *--stackPtr;
        }
        else
        {
            oldNode = node;
            for (int i = 0; i < NODES_NUMBER; ++i)
            {
                if (collisions[i])
                {
                    node = &(nodes[node->getChildren(i)]);
                    break;
                }
            }
            d_QuadTree* nodes = treeManager->nodes;
        #pragma unroll
            for (int i = 0; i < NODES_NUMBER; ++i)
            {
                if (collisions[i] && node != &(nodes[oldNode->children[i]]))
                    *stackPtr++ = &(nodes[oldNode->children[i]]);
            }
        //   oldNode->addNodesToStack(stackPtr, node, collisions);
        }
    }
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
