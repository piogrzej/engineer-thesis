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
	Rect();
	Rect(point topLeft,point bottomRight);
	Rect(int topLeftX, int topLeftY, int bootomRightX, int bottomRightY);
	~Rect();

	point	topLeft;
	point	bottomRight;

	int		getWidth() const;
	int		getHeigth() const;
	bool	cmpRect(Rect const& r2) const;
	bool	rectsCollision(Rect const& r2) const;
	bool	rectContains(point p) const;
	Rect	createGaussianSurface(double factor) const;
	int		getPerimeter() const;
	point	getPointFromNindex(int index, int Nsample) const;
	friend ostream& operator<< (ostream&, Rect const&);
};

#endif