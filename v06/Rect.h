#ifndef RECT_H
#define RECT_H

#include <iostream>

using namespace std;

struct point {
	int x;
	int y;
};

class Rect
{
public:
	Rect();
	Rect(point topLeft,point bottomRight);
	Rect(int topLeftX, int topLeftY, int bootomRightX, int bottomRightY);
	~Rect();
	point topLeft;
	point bottomRight;

	int getWidth();
	int getHeigth();
	bool cmpRect(Rect r2);
	bool rectsCollision(Rect r2);
	bool rectContains(point p);
	Rect createGaussianSurface(double factor);
	int getPerimeter();
	point getPointFromNindex(int index, int Nsample);
	friend ostream& operator<< (ostream&, Rect const&);
};

#endif