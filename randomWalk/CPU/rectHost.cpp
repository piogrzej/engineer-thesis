#include "rectHost.h"

#include <math.h> //round

RectHost::RectHost(point tLeft, point bRight)
{
	topLeft = tLeft;
	bottomRight = bRight;
}

RectHost::~RectHost()
{
}

void RectHost::changeDirection()
{
    if (getWidth() < 0 || getHeigth() < 0)
    {
        point tmp = topLeft;
        topLeft = bottomRight;
        bottomRight = tmp;
    }
}

floatingPoint RectHost::getWidth() const {
    return (bottomRight.x - topLeft.x);
}

floatingPoint RectHost::getHeigth() const {
    return (bottomRight.y - topLeft.y);
}

bool RectHost::cmpRect(RectHost const& r2) const {
    if (this->bottomRight.x == r2.bottomRight.x &&
            this->bottomRight.y == r2.bottomRight.y &&
            this->topLeft.x == r2.topLeft.x &&
            this->topLeft.y == r2.topLeft.y)
        return true;
    else
        return false;
}

bool RectHost::rectsCollision(RectHost const& r2) const 
{
    if (bottomRight.x >= r2.topLeft.x &&
        r2.bottomRight.x >= topLeft.x    &&
        bottomRight.y >= r2.topLeft.y &&
        r2.bottomRight.y >= topLeft.y)    
        return true;

	return false;
}

bool RectHost::rectContains(point p) const
{
    if ((p.x >= topLeft.x) &&
        (p.y >= topLeft.y) &&
        (p.x <= bottomRight.x) &&
        (p.y <= bottomRight.y))
        return true;
    else
        return false;
}

bool RectHost::rectContains(RectHost r) const {
    if(r.topLeft.x > topLeft.x &&
       r.topLeft.y > topLeft.y &&
       r.bottomRight.x < bottomRight.x &&
       r.bottomRight.y < bottomRight.y)
        return true;
    else
        return false;
}

RectHost  RectHost::createGaussianSurfaceX(floatingPoint factorX) const
{
    return createGaussianSurface(factorX, 1.);
}

RectHost  RectHost::createGaussianSurfaceY(floatingPoint factorY) const
{
    return createGaussianSurface(1., factorY);
}
RectHost RectHost::createGaussianSurface(floatingPoint factorX, floatingPoint factorY) const 
{
    floatingPoint middleX = floatingPoint(topLeft.x + bottomRight.x) / 2.;
    floatingPoint middleY = floatingPoint(topLeft.y + bottomRight.y) / 2.;
    floatingPoint vectorX = floatingPoint(topLeft.x) - middleX;
    floatingPoint vectorY = floatingPoint(topLeft.y) - middleY;
    RectHost gaussSurface;
    vectorX *= factorX;
    vectorY *= factorY;

    gaussSurface.topLeft.x = middleX + vectorX;
    gaussSurface.topLeft.y = middleY + vectorY;
    gaussSurface.bottomRight.x = middleX - vectorX;
    gaussSurface.bottomRight.y = middleY - vectorY;

    return gaussSurface;
}

int RectHost::getPerimeter() const
{
    return (2 * (this->bottomRight.x - this->topLeft.x) + 2 * (this->bottomRight.y - this->topLeft.y));
}

point RectHost::getPointFromNindex(int index, int Nsample) {
	floatingPoint perimeter = this->getPerimeter();
    floatingPoint vector = (floatingPoint)perimeter / (floatingPoint)Nsample;
    floatingPoint heigth = this->getHeigth();
    floatingPoint width = this->getWidth();
    point ret;
    if (index*vector < width){
        ret.x = (int)((index-1)*vector +vector / 2 + this->topLeft.x);
        ret.y = this->topLeft.y;
        return ret;
    }
    else if ((index - 1)*vector < width &&index*vector > width){//JEZELI PRZECHODZI PRZEZ KRAWEDZ TO DAJE WIESZCHO�EK
        ret.x = this->bottomRight.x;
        ret.y = this->topLeft.y;
        return ret;
    }
    else if (index*vector < (width + heigth)){
        ret.x = this->bottomRight.x;
        if ((index - 1)*vector > width){
            ret.y = this->topLeft.y + (index - 1)*vector - width + vector / 2;
        }
        else{
            ret.y = this->topLeft.y + index*vector - width + (index*vector - width)/2;
        }
        return ret;
    }
    else if ((index - 1)*vector < (width + heigth) && vector*index>(width + heigth)){
        return this->bottomRight;
    }
    else if (index*vector < (2 * width + heigth)){
        ret.y = this->bottomRight.y;
        if ((index - 1)*vector > width + heigth){
            ret.x = this->bottomRight.x - ((index - 1)*vector - width - heigth + vector / 2 );
        }
        else{
            ret.x = this->bottomRight.x - (index*vector - width - heigth + (index*vector - width - heigth) / 2);
        }
        return ret;
    }
    else if ((index - 1)*vector < (2 * width + heigth) && vector*index>(2 * width + heigth)){
        ret.x = this->topLeft.x;
        ret.y = this->bottomRight.y;
        return ret;
    }
    else{
        ret.x = this->topLeft.x;
        if ((index - 1)*vector > 2*width + heigth){
            ret.y = this->bottomRight.y - ((index - 1)*vector - 2*width - heigth + vector / 2);
        }
        else{
            ret.y = this->bottomRight.y - (index*vector - 2*width - heigth + (index*vector - 2*width - heigth) / 2);
        }
        return ret;
    }
}
bool RectHost::operator==(const RectHost & r2) const
{
    if (r2.topLeft.x == topLeft.x &&
        r2.topLeft.y == topLeft.y &&
        r2.bottomRight.x == bottomRight.x &&
        r2.bottomRight.y == bottomRight.y)
        return true;
    else
        return false;
}

bool RectHost::operator<(const RectHost & r2) const
{
    if (r2.topLeft.x < topLeft.x &&
        r2.topLeft.y < topLeft.y &&
        r2.bottomRight.x < bottomRight.x &&
        r2.bottomRight.y < bottomRight.y)
        return true;
    else
        return false;
}

std::ostream& operator<< (std::ostream &wyjscie, RectHost const& ex)
{
    wyjscie << "Rect TL: x:" << ex.topLeft.x<<", y:" << ex.topLeft.y << "  BR: x:" << ex.bottomRight.x << ", y:" << ex.bottomRight.y;
    return wyjscie;
}
