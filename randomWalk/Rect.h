#ifndef RECT_H
#define RECT_H

#include <iostream>

typedef float floatingPoint;

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

class Rect
{
public:
    Rect() { bottomRight = topLeft = point(-1, -1);  } // incorrect rect
    Rect(point tLeft, point bRight);
    ~Rect();

    point   topLeft;
    point   bottomRight;

    void    changeDirection();
    int     getWidth() const;
    int     getHeigth() const;
    bool    cmpRect(Rect const& r2) const;
    bool    rectsCollision(Rect const& r2) const;
    bool    rectContains(point p) const;
    bool    rectContains(Rect r) const;
    Rect    createGaussianSurface(floatingPoint factorX, floatingPoint factorY) const;
    Rect    createGaussianSurfaceX(floatingPoint factorX) const;
    Rect    createGaussianSurfaceY(floatingPoint factorY) const;
    int     getPerimeter() const;
    point   getPointFromNindex(int index, int Nsample);
    friend  std::ostream& operator<< (std::ostream&, Rect const&);
    bool    operator==(const Rect& r2) const;
    bool    operator!=(const Rect& r2) const { return !(*this == r2); }
};

#endif