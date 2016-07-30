#ifndef RECT_H
#define RECT_H

#include <iostream>

using namespace std;

struct point 
{
	int x;
	int y;
	point() {}
	point(int pX, int pY) { x = pX; y = pY; }
};

class Rect
{
public:
    Rect() {}
	Rect(point tLeft, point bRight);
	~Rect();

	point	topLeft;
	point	bottomRight;

	int	getWidth() const;
	int	getHeigth() const;
	bool	cmpRect(Rect const& r2) const;
	bool	rectsCollision(Rect const& r2) const;
	bool	rectContains(point p) const;
        bool	rectContains(Rect r) const;
	Rect    createGaussianSurface(double factor) const;
	int     getPerimeter() const;
	point	getPointFromNindex(int index, int Nsample) const;
	friend  ostream& operator<< (ostream&, Rect const&);
        bool    operator==(const Rect& r2) const;
        bool    operator!=(const Rect& r2) const { return !(*this == r2); }
};

#endif