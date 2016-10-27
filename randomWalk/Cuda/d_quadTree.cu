#include "d_quadtree.h"

__host__ __device__ bool d_QuadTree::isInBounds(point2 const&  p)
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

__host__ __device__ bool d_QuadTree::isInBounds(d_Rect const&  r)
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

__host__ __device__ bool d_QuadTree::checkCollisons(point2 p, d_Rect& r)
{
    d_QuadTree* current=this,*next;
    while(true)
    {
        if (true==current->isSplited())
        {
            for(ushort i=0; i<NODES_NUMBER; ++i)
            {
                if (true==current->getTreeManager()->nodes[current->getChlidren(i)].bounds.contains(p))
                {
                    next = &(current->getTreeManager()->nodes[current->getChlidren(i)]);
                    break;
                }
            }
        }
        //tutaj dla kazdego sprawdzenie bisectory lines
        if (true==current->checkCollisionObjs(p, r))//KOLIZJA
            return true;
        else if(true==current->isSplited())
            return false;
        else
            current=next;
    }
}

__device__ bool d_QuadTree::checkCollisionObjs(point2 p, d_Rect &r)
{
    for(ushort i = this->startRectOff(); i < this->endRectOff(); ++i)
    {
        if(true == this->getTreeManager()->rects[i].contains(p))
        {
            r = d_Rect(this->getTreeManager()->rects[i].topLeft,
                    this->getTreeManager()->rects[i].bottomRight);
            return true;
        }
    }
    return false;
}

__host__ __device__ d_Rect d_QuadTree::drawBiggestSquareAtPoint(point2 p)
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

__device__ __host__ bool d_QuadTree::checkCollisions(d_Rect const& r, const d_Rect &ignore)//FUNKCJA DO PRZEPISANIA OD NOWA
{
    if (false == isInBounds(r))
        return true;

    d_QuadTree* oldNode, *node = this;
    TreePtr* stack = new TreePtr[this->getTreeManager()->nodes[0].rectCount() + 1];//UWAGA MOZE NIE DZIALAC!!!
    TreePtr* stackPtr = stack;
    bool collisions[NODES_NUMBER];
    *stackPtr++ = nullptr; // koniec petli gdy tu trafimy

    while (node != nullptr)
    {
        if (true==node->isSplited())
        {
            for (int i = 0; i < NODES_NUMBER; ++i)
                collisions[i-node->startRectOff()] = node->getTreeManager()->nodes[node->getChlidren(i)].getBounds().rectsCollision(r);
        }
        else
            for (int i = 0; i < NODES_NUMBER; ++i)
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

            for (int i = 0; i < NODES_NUMBER; ++i)
            {
                if (collisions[i])
                {
                    node = &(node->getTreeManager()->nodes[node->getChlidren(i)]);
                    break;
                }
            }

            oldNode->addNodesToStack(stackPtr, node, collisions);
        }
    }
    delete stack;
    return false;
}

__host__ __device__ bool d_QuadTree::checkIsAnyCollision(bool collisions[])//FUNKCJA DO PRZEPISANIA OD NOWA
{
    for (int i = 0; i < NODES_NUMBER; ++i)
    {
        if (collisions[i])
            return true;
    }
    return false;
}

__host__ __device__ void d_QuadTree::addNodesToStack(TreePtr* stackPtr,d_QuadTree* except, bool collisions[])//FUNKCJA DO PRZEPISANIA OD NOWA
{
    for (int i = 0; i < NODES_NUMBER; ++i)
    {
        if (collisions[i] && except != &(this->getTreeManager()->nodes[this->getChlidren(i)]))
            *stackPtr++ = &(this->getTreeManager()->nodes[this->getChlidren(i)]);
    }
}

__host__ __device__ bool d_QuadTree::checkCollisionsWithObjs(d_Rect const& r, const d_Rect &ignore)
{
    for (int i =this->startOff; i< this->endOff; ++i)
            if (this->getTreeManager()->rects[i] != ignore && this->getTreeManager()->rects[i].rectsCollision(r))
                return true;

        return false;
}