#ifndef RECT_H
#define RECT_H

#include <iostream>

typedef double floatingPoint;

struct point 
{
    floatingPoint x;
    floatingPoint y;
    point() {}
    point(floatingPoint pX, floatingPoint pY) { x = pX; y = pY; }
    void  operator+=(floatingPoint scalar) {  x += scalar; y += scalar; }
    void  operator-=(floatingPoint scalar) {  x -= scalar; y -= scalar; }
    point operator+(floatingPoint scalar) { return point(x + scalar, y + scalar); }
    point operator-(floatingPoint scalar) { return point(x - scalar, y - scalar); }
};

class RectHost
{

public:
    RectHost() { bottomRight = topLeft = point(-1, -1);  } // incorrect rect
    RectHost(point tLeft, point bRight);
    ~RectHost();

    point   topLeft;
    point   bottomRight;

    void    changeDirection();
    floatingPoint     getWidth() const;
 	floatingPoint     getHeigth() const;
    bool    cmpRect(RectHost const& r2) const;
    bool    rectsCollision(RectHost const& r2) const;
    bool    rectContains(point p) const;
    bool    rectContains(RectHost r) const;
    RectHost createGaussianSurface(floatingPoint factorX, floatingPoint factorY) const;
    RectHost createGaussianSurfaceX(floatingPoint factorX) const;
    RectHost createGaussianSurfaceY(floatingPoint factorY) const;
    int     getPerimeter() const;
    point   getPointFromNindex(int index, int Nsample);
    friend  std::ostream& operator<< (std::ostream&, RectHost const&);
    bool    operator==(const RectHost& r2) const;
    bool    operator!=(const RectHost& r2) const { return !(*this == r2); }
};



#endif
